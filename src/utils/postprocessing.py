import pycocotools.mask as mask_util
import numpy as np
import torch

import pycocotools.mask as mask_util

def custom_nms(pred, nms_thresh=0.5):
    pred_masks = pred['instances'].pred_masks.detach().cpu().numpy()
    scores = pred['instances'].scores.detach().cpu().numpy()
    # print(pred_masks.dtype)
    if(pred_masks.dtype == np.float32):
        pred_masks = (pred_masks >= 0.5 ).astype('bool')
        # print(pred_masks.dtype)

    # calculate IOU
    enc_preds = [mask_util.encode(np.asarray(p, order='F')) for p in pred_masks]
    ious = mask_util.iou(enc_preds, enc_preds, [0]*len(enc_preds))

    orders = np.arange(len(scores))
    keeps = []

    while len(orders) > 0:

        # append the idx of the instance X with highest score to keeps list
        keeps.append(orders[0])
        # remove the idx of the above instance from the remaining list
        orders = orders[1:]

        # check IOU of the instance X with all the instance in the remaining list
        look_up = ious[keeps[-1], orders]

        # filter those having IOU > nms_thresh
        mask = look_up < nms_thresh
        orders = orders[mask]

        if(len(orders)==0):
            break
    
    new_pred = pred.copy()
    new_pred['instances'] = new_pred['instances'][keeps]
    return new_pred

def post_process_output(cfg, outputs):
    # filter by score
    pred_class = torch.mode(outputs['instances'].pred_classes)[0]
    
    # filter by confidence score
    take = outputs['instances'].scores >= cfg.CUSTOM.THRESHOLDS[pred_class]
    outputs['instances'] = outputs['instances'][take]
    
    # nms by mask
    outputs = custom_nms(outputs, nms_thresh=cfg.CUSTOM.NMS_THRESH[pred_class])
    
    return outputs