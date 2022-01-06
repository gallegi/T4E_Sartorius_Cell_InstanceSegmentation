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

parser = argparse.ArgumentParser(description='Some argument')
parser.add_argument('--weights', type=str, default='models/pretrained_models/pseudo_round1_model.pth models/pretrained_models/pseudo_round2_model.pth',
                    help='Provide paths to weight separated by a space')

args = parser.parse_args()

weight_list = args.weights.split(' ')
print(f'Ensemble {len(weight_list)} models:')
print(weight_list)

ROOT_FOLDER = './'
ANN_DIR = f'{ROOT_FOLDER}/data/annotation_semisupervised_round2/annotations'
DATA_DIR = f'{ROOT_FOLDER}/data/annotation_semisupervised_round2/images'
FOLD = 0
FINAL_THRESH = [0.5, 0.7, 0.8]
PKL_FOLDER = f'{ROOT_FOLDER}/data_for_2nd_level_model'


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


dataDir=Path(DATA_DIR)
register_coco_instances('sartorius_val',{}, f'{ANN_DIR}/valid/annotations_valid_{FOLD}.json', dataDir)
metadata = MetadataCatalog.get('sartorius_val')
valid_ds = DatasetCatalog.get('sartorius_val')


list_cfgs = []
list_predictors = []

for weight in args.weights.split(' '):
    cfg = get_config(weight)
    predictor = DefaultPredictor(cfg)
    list_cfgs.append(cfg)
    list_predictors.append(predictor)


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



