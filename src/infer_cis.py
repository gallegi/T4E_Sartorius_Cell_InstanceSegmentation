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
from detectron2.data import MetadataCatalog, DatasetCatalog, Metadata
from detectron2.data.datasets import register_coco_instances
from detectron2.modeling.backbone import fpn_resneSt

import numpy as np
import torch
from tqdm import tqdm
import argparse

from utils.metric_by_outputs import calculate_AP
from utils.ensemble import ensemble

parser = argparse.ArgumentParser(description='Some arguments')
parser.add_argument('--image', type=str, default='demo_images/7ae19de7bc2a.png',
                    help='Path to input image')
parser.add_argument('--weights', type=str, default='models/pretrained_models/pseudo_round1_model.pth models/pretrained_models/pseudo_round2_model.pth',
                    help='Paths to weight separated by a space')
parser.add_argument('--2nd_level_model', type=str, default='2nd_level_model/catboost.pkl',
                    help='Path to pkl file where 2nd-level model weights are saved')
parser.add_argument('--2nd_level_features', type=str, default='2nd_level_model/features.csv',
                    help='Path to csv files containing features used by 2nd-level model')

args = parser.parse_args()

weight_list = args.weights.split(' ')
print(f'Ensemble {len(weight_list)} models:')
print(weight_list)

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

im = cv2.imread(args.image)
input_dict = {'file_name':args.image, 'height':520, 'width':704}
outputs =  ensemble(input_dict, 
                list_cfgs, 
                list_predictors,
                conf_thresh=FINAL_THRESH)

v = Visualizer(im[:, :, ::-1],
                   metadata = Metadata(thing_classes=['shsy5y', 'astro', 'cort']), 
                   instance_mode=ColorMode.IMAGE_BW
    )
out_pred = v.draw_instance_predictions(outputs["instances"].to("cpu"))

out_name = args.image.split('/')[-1]
out_path = os.path.join('demo_outputs/', out_name)
plt.savefig(out_path)
