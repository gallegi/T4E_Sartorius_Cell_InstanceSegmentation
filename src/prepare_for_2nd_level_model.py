'''
    Generate pickle files containing output prediction of MaskRCNN models
    in order to training 2nd-level model
'''
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

import pickle
import numpy as np
import torch

from utils.metric_by_outputs import calculate_AP
from utils.ensemble import ensemble

from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description='Some arguments')
parser.add_argument('--image_dir', type=str, default=f'data/images',
                    help='Path to image folder')
parser.add_argument('--annotation_dir', type=str, default=f'data/annotations_semi_supervised_round2',
                    help='Path to annotation folder')
parser.add_argument('--weights', type=str, default='models/pretrained_models/pseudo_round1_model.pth models/pretrained_models/pseudo_round2_model.pth',
                    help='Paths to weight separated by a space')

args = parser.parse_args()

weight_list = args.weights.split(' ')
print(f'Ensemble {len(weight_list)} models:')
print(weight_list)

ANN_DIR = args.annotation_dir
IMAGE_DIR = args.image_dir
FOLD = 0
FINAL_THRESH = [0.5, 0.7, 0.8]
PKL_FOLDER = f'data_for_2nd_level_model'

# =============== Make detectron output raw pixel float mask =================
def paste_masks_in_image(masks, boxes, image_shape, threshold=0.5):
    """
    Copy pasted from detectron2.layers.mask_ops.paste_masks_in_image and deleted thresholding of the mask
    """
    assert masks.shape[-1] == masks.shape[-2], "Only square mask predictions are supported"
    N = len(masks)
    if N == 0:
        return masks.new_empty((0,) + image_shape, dtype=torch.uint8)
    if not isinstance(boxes, torch.Tensor):
        boxes = boxes.tensor
    device = boxes.device
    assert len(boxes) == N, boxes.shape

    img_h, img_w = image_shape

    # The actual implementation split the input into chunks,
    # and paste them chunk by chunk.
    if device.type == "cpu":
        # CPU is most efficient when they are pasted one by one with skip_empty=True
        # so that it performs minimal number of operations.
        num_chunks = N
    else:
        # GPU benefits from parallelism for larger chunks, but may have memory issue
        num_chunks = int(np.ceil(N * img_h * img_w * BYTES_PER_FLOAT / GPU_MEM_LIMIT))
        assert (
            num_chunks <= N
        ), "Default GPU_MEM_LIMIT in mask_ops.py is too small; try increasing it"
    chunks = torch.chunk(torch.arange(N, device=device), num_chunks)

    img_masks = torch.zeros(
        N, img_h, img_w, device=device, dtype=torch.float32
    )
    for inds in chunks:
        masks_chunk, spatial_inds = _do_paste_mask(
            masks[inds, None, :, :], boxes[inds], img_h, img_w, skip_empty=device.type == "cpu"
        )
        img_masks[(inds,) + spatial_inds] = masks_chunk
    return img_masks

from typing import Any, Iterator, List, Union
import numpy as np

def BitMasks__init__(self, tensor: Union[torch.Tensor, np.ndarray]):
    """
    Args:
        tensor: bool Tensor of N,H,W, representing N instances in the image.
    """
    device = tensor.device if isinstance(tensor, torch.Tensor) else torch.device("cpu")
    tensor = torch.as_tensor(tensor, dtype=torch.float32, device=device)
    assert tensor.dim() == 3, tensor.size()
    self.image_size = tensor.shape[1:]
    self.tensor = tensor
    
detectron2.structures.masks.BitMasks.__init__.__code__ = BitMasks__init__.__code__
detectron2.layers.mask_ops.paste_masks_in_image.__code__ = paste_masks_in_image.__code__
# ==================================================================================

# =========== Define functions for config ==========
def get_config(weight):
    cfg = get_cfg()
    cfg.INPUT.MASK_FORMAT='bitmask'

    cfg.MODEL.RESNETS.RADIX = 1
    cfg.MODEL.RESNETS.DEEP_STEM = False
    cfg.MODEL.RESNETS.AVD = False
    # Apply avg_down to the downsampling layer for residual path 
    cfg.MODEL.RESNETS.AVG_DOWN = False
    cfg.MODEL.RESNETS.BOTTLENECK_WIDTH = 64

    cfg.merge_from_file(f"configs/mask_rcnn_ResNeSt200.yaml")
    
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
# =======================================================

# ====== Register datasets =======
dataDir=Path(IMAGE_DIR)
register_coco_instances('sartorius_val',{}, f'{ANN_DIR}/annotations_valid_{FOLD}.json', dataDir)
metadata = MetadataCatalog.get('sartorius_val')
valid_ds = DatasetCatalog.get('sartorius_val')
# ================================

# ========== Load models ===========
list_cfgs = []
list_predictors = []

for weight in args.weights.split(' '):
    cfg = get_config(weight)
    predictor = DefaultPredictor(cfg)
    list_cfgs.append(cfg)
    list_predictors.append(predictor)
# ==================================

# ========== Predict and save to pkl files ===========
i = 0
for d in tqdm(valid_ds, total=len(valid_ds)):
    name = d['file_name'].split('/')[-1]
    outputs =  ensemble(d, 
                    list_cfgs, 
                    list_predictors,
                    conf_thresh=FINAL_THRESH)
    with open(os.path.join(PKL_FOLDER, name[:-4]+'.pkl'), 'wb') as f:
        pickle.dump(outputs, f)
  
    i+=1
# =====================================================



