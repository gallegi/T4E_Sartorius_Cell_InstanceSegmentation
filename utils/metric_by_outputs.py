import numpy as np
import torch
import pycocotools.mask as mask_util
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.logger import setup_logger
from detectron2.evaluation.evaluator import DatasetEvaluator
from detectron2.engine import DefaultTrainer

def precision_at(threshold, iou):
    matches = iou > threshold
    true_positives = np.sum(matches, axis=1) == 1  # Correct objects
    false_negatives = np.sum(matches, axis=0) == 0  # Missed objects
    false_positives = np.sum(matches, axis=1) == 0  # Extra objects
    return np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)

def calculate_AP(pred, targ, resolve_overlap=True):
    pred_masks = pred['instances'].pred_masks.detach().cpu().numpy()
    
    if(resolve_overlap):
        res = []
        used = np.zeros(pred_masks[0].shape, dtype=int)
        for mask in pred_masks:
            mask = mask * (1-used)
            used += mask
            res.append(mask)
        pred_masks = res
        pred_masks = np.array(pred_masks).astype('uint8')
    
    enc_preds = [mask_util.encode(np.asarray(p, order='F')) for p in pred_masks]
    enc_targs = list(map(lambda x:x['segmentation'], targ))
    ious = mask_util.iou(enc_preds, enc_targs, [0]*len(enc_targs))
    prec = []
    TPs, FPs, FNs = [],[],[]
    log_dict = dict()
    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = precision_at(t, ious)
        p = tp / (tp + fp + fn)
        prec.append(p)
        TPs.append(tp)
        FPs.append(fp)
        FNs.append(fn)
        log_dict[round(t,2)] = {'AP':p, 'TP':tp, 'FP':fp, 'FN':fn}
    return np.mean(prec), np.mean(TPs), np.mean(FPs), np.mean(FNs), log_dict

def get_AP_single_model(cfg, predictor, ds):
    valid_APs = []
    for d in tqdm(ds, total=len(ds)):
        im = cv2.imread(d['file_name'])

        outputs = predictor(im)
        outputs = post_process_output(cfg, outputs)

        AP = calculate_AP( outputs, d['annotations'])
        valid_APs.append(AP)
        
    return np.mean(valid_APs)