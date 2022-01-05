import detectron2
from pathlib import Path
import random, cv2, os
import pandas as pd
import matplotlib.pyplot as plt
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg, CfgNode
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.modeling.backbone import fpn_resneSt

import copy
from detectron2.data.detection_utils import read_image
from detectron2.modeling.test_time_augmentation import GeneralizedRCNNWithTTA, DatasetMapperTTA
from detectron2.modeling.roi_heads.fast_rcnn import fast_rcnn_inference_single_image
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.structures import Boxes, Instances
from fvcore.transforms import HFlipTransform, NoOpTransform

from contextlib import contextmanager
from itertools import count
import itertools

import torch

from utils.postprocessing import post_process_output
from utils.metric_by_outputs import calculate_AP

from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description='Some argument')
parser.add_argument('--weights', type=str, default='models/pretrained_models/pseudo_round2_model.pth',
                    help='Path to pth weight')

args = parser.parse_args()

print(args.weights)

ROOT_FOLDER = './'
ANN_DIR = f'{ROOT_FOLDER}/data/annotation_semisupervised_round2/annotations'
DATA_DIR = f'{ROOT_FOLDER}/data/annotation_semisupervised_round2/images'
FOLD = 0
FINAL_THRESH = [0.5, 0.7, 0.8]


def get_config(weight):
    cfg = get_cfg()
    cfg.INPUT.MASK_FORMAT='bitmask'

    cfg.MODEL.RESNETS.RADIX = 1
    cfg.MODEL.RESNETS.DEEP_STEM = False
    cfg.MODEL.RESNETS.AVD = False
    # Apply avg_down to the downsampling layer for residual path 
    cfg.MODEL.RESNETS.AVG_DOWN = False
    cfg.MODEL.RESNETS.BOTTLENECK_WIDTH = 64

    cfg.merge_from_file(f"{ROOT_FOLDER}/configs/mask_rcnn_ResNeSt200.yaml")
    
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  
    
    cfg.MODEL.WEIGHTS = weight
    
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.99
    cfg.CUSTOM = CfgNode()
    cfg.TEST.DETECTIONS_PER_IMAGE = 10000
    
    # img size
    cfg.INPUT.MIN_SIZE_TEST = 1024
    cfg.INPUT.MAX_SIZE_TEST = 2000
    
    cfg.TEST.AUG.MIN_SIZES = (1024, )
    cfg.TEST.AUG.FLIP = False
    
    return cfg

@contextmanager
def turn_off_roi_heads(model, attrs):
        """
        Open a context where some heads in `model.roi_heads` are temporarily turned off.
        Args:
            attr (list[str]): the attribute in `model.roi_heads` which can be used
                to turn off a specific head, e.g., "mask_on", "keypoint_on".
        """
        roi_heads = model.roi_heads
        old = {}
        for attr in attrs:
            try:
                old[attr] = getattr(roi_heads, attr)
            except AttributeError:
                # The head may not be implemented in certain ROIHeads
                pass

        if len(old.keys()) == 0:
            yield
        else:
            for attr in old.keys():
                setattr(roi_heads, attr, False)
            yield
            for attr in old.keys():
                setattr(roi_heads, attr, old[attr])

def model_batch_forward(model, inputs, instances=[None]):
    outputs = model.inference(inputs, 
                   instances if instances[0] is not None else None,
                   do_postprocess=False)
    return outputs

def merge_detections(all_boxes, all_scores, all_classes, shape_hw, num_classes=3, 
                     box_nms_thresh=0.99, max_detection_per_im=10000):
    # select from the union of all results
    num_boxes = len(all_boxes)
    
    # +1 because fast_rcnn_inference expects background scores as well
    all_scores_2d = torch.zeros(num_boxes, num_classes + 1, device=all_boxes.device)
    for idx, cls, score in zip(count(), all_classes, all_scores):
        all_scores_2d[idx, cls] = score

    merged_instances, _ = fast_rcnn_inference_single_image(
        all_boxes,
        all_scores_2d,
        shape_hw,
        1e-8,
        box_nms_thresh,
        max_detection_per_im,
    )
    
    return merged_instances

def rescale_detected_boxes(augmented_inputs, merged_instances, tfms):
    augmented_instances = []
    for input, tfm in zip(augmented_inputs, tfms):
        # Transform the target box to the augmented image's coordinate space
        pred_boxes = merged_instances.pred_boxes.tensor.cpu().numpy()
        pred_boxes = torch.from_numpy(tfm.apply_box(pred_boxes))

        aug_instances = Instances(
            image_size=input["image"].shape[1:3],
            pred_boxes=Boxes(pred_boxes),
            pred_classes=merged_instances.pred_classes,
            scores=merged_instances.scores,
        )
        augmented_instances.append(aug_instances)
    return augmented_instances

def reduce_pred_masks(outputs, tfms):
    # Should apply inverse transforms on masks.
    # We assume only resize & flip are used. pred_masks is a scale-invariant
    # representation, so we handle flip specially
    for output, tfm in zip(outputs, tfms):
        if any(isinstance(t, HFlipTransform) for t in tfm.transforms):
            output.pred_masks = output.pred_masks.flip(dims=[3])
    all_pred_masks = torch.stack([o.pred_masks for o in outputs], dim=0)
    avg_pred_masks = torch.mean(all_pred_masks, dim=0)
    return avg_pred_masks

def ensemble(inp_dict, list_configs, list_predictors, 
            mask_nms_thresh=0.1, max_detection_per_im=10000,
            conf_thresh=[0.3,0.5,0.6]):
    # read image
    inp = copy.copy(inp_dict)
    image = read_image(inp.pop("file_name"), list_predictors[0].model.input_format)
    image = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))  # CHW
    inp["image"] = image
    if "height" not in inp and "width" not in inp:
        inp["height"] = image.shape[1]
        inp["width"] = image.shape[2]
    orig_shape = (inp["height"], inp["width"])
    
    # dataset mapper for each model
    list_tta_mappers = []
    list_augmented_inputs = []
    list_tfms = []
    
    all_boxes = []
    all_scores = []
    all_classes = []
    
    for cfg, predictor in zip(list_configs, list_predictors):
        tta_mapper = DatasetMapperTTA(cfg)
        list_tta_mappers.append(tta_mapper)
        augmented_inputs = tta_mapper(inp)
        
        tfms = [x.pop("transforms") for x in augmented_inputs]
        list_augmented_inputs.append(augmented_inputs)
        list_tfms.append(tfms)
        
#         return augmented_inputs
        with turn_off_roi_heads(predictor.model, ['mask_on', 'keypoint_on']), torch.no_grad():
            outputs = model_batch_forward(predictor.model, augmented_inputs)
            
        for output, tfm in zip(outputs, tfms):
            # Need to inverse the transforms on boxes, to obtain results on original image
            pred_boxes = output.pred_boxes.tensor
            original_pred_boxes = tfm.inverse().apply_box(pred_boxes.cpu().numpy())
            all_boxes.append(torch.from_numpy(original_pred_boxes).to(pred_boxes.device))

            all_scores.extend(output.scores)
            all_classes.extend(output.pred_classes)
            
    all_boxes = torch.cat(all_boxes, dim=0)
    
    merged_instances = merge_detections(all_boxes, all_scores, all_classes, orig_shape)

    # filter by confidence score
    pred_class = torch.mode(merged_instances.pred_classes)[0]
    take = merged_instances.scores >= conf_thresh[pred_class]
    merged_instances = merged_instances[take]
    
    list_outputs = []
    for cfg, predictor, augmented_inputs, tfms in zip(list_configs, list_predictors, list_augmented_inputs, list_tfms):
        augmented_instances = rescale_detected_boxes(
            augmented_inputs, merged_instances, tfms
        )
                  
        with torch.no_grad():
            outputs = model_batch_forward(predictor.model, augmented_inputs, augmented_instances)
        
        del augmented_inputs, augmented_instances
        
        list_outputs.extend(outputs)
        
    list_tfms = itertools.chain.from_iterable(list_tfms)
    
    merged_instances.pred_masks = reduce_pred_masks(list_outputs, list_tfms)
    merged_instances = detector_postprocess(merged_instances, *orig_shape)
    
    final_outputs = {'instances':merged_instances}
    final_outputs = custom_nms(final_outputs, nms_thresh=mask_nms_thresh)
    # final_outputs = post_process_output(final_outputs, mask_nms_thresh=mask_nms_thresh, conf_thresh=conf_thresh)
    
    return final_outputs



dataDir=Path(DATA_DIR)
register_coco_instances('sartorius_val',{}, f'{ANN_DIR}/valid/annotations_valid_{FOLD}.json', dataDir)
metadata = MetadataCatalog.get('sartorius_val')
valid_ds = DatasetCatalog.get('sartorius_val')


list_cfgs = []
list_predictors = []

for weight in args.weight.split(' '):
    cfg = get_config(cfg)
    predictor = DefaultPredictor(cfg)
    list_cfgs.append(cfg)
    list_predictors.append(predictor)


list_APs = []
list_TPs = []
list_FPs = []
list_FNs = []
list_logs = []
list_cell_types = []
list_inst_counts = []
list_im_ids = []

i = 0
for d in tqdm(valid_ds, total=len(valid_ds)):
    default_thresh[CLASS_ID] = thresh
    outputs =  ensemble(d, 
                    list_cfgs, 
                list_predictors,
                    conf_thresh=FINAL_THRESH)
    
    calculate_AP(outputs, d['annotations'])
    AP, TP, FP, FN, log = calculate_AP(outputs, d['annotations'])

    list_APs.append(AP)
    list_logs.append(log)
    list_TPs.append(TP)
    list_FPs.append(FP)
    list_FNs.append(FN)
    
    list_cell_types.append(d['annotations'][0]['category_id'])
    list_inst_counts.append(len(d['annotations']))
    list_im_ids.append(d['image_id'])
    i+=1
    if(i > 3):
        break

import pandas as pd
result_df = pd.DataFrame({'image_id':list_im_ids, 'cell_type':list_cell_types, 'inst_count':list_inst_counts,
                        'AP':list_APs, 'TP':list_TPs, 'FP':list_FPs, 'FN':list_FNs,'log':list_logs})

print('Result by each cell type (average precision IOU@0.5:0.95):')
print(result_df.groupby('cell_type').AP.sum() / len(result_df))

print('Result (average precision IOU@0.5:0.95):')
print(result_df.AP.mean())


