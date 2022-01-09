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


dataDir=Path(IMAGE_DIR)

register_coco_instances('sartorius_val',{}, f'{ANN_DIR}/annotations_valid_{FOLD}.json', dataDir)
metadata = MetadataCatalog.get('sartorius_val')
valid_ds = DatasetCatalog.get('sartorius_val')

list_cfgs = []
list_predictors = []

for weight in args.weights.split(' '):
    cfg = get_config(weight)
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
    # if(i > 3):
    #     break

import pandas as pd
result_df = pd.DataFrame({'image_id':list_im_ids, 'cell_type':list_cell_types, 'inst_count':list_inst_counts,
                        'AP':list_APs, 'TP':list_TPs, 'FP':list_FPs, 'FN':list_FNs,'log':list_logs})

print('Result by each cell type (average precision IOU@0.5:0.95):')
print(result_df.groupby('cell_type').AP.sum() / len(result_df))

print('\nResult (average precision IOU@0.5:0.95):')
print(result_df.AP.mean())


