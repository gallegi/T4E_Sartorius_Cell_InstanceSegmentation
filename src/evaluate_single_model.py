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

import argparse

parser = argparse.ArgumentParser(description='Some argument')
parser.add_argument('--weight', type=str, default='models/pretrained_models/pseudo_round2_model.pth',
                    help='Path to pth weight')

args = parser.parse_args()

print(args.weight)

cfg = get_cfg()
ROOT_FOLDER = './'

ANN_DIR = f'{ROOT_FOLDER}/data/annotation_semisupervised_round2/annotations'
DATA_DIR = f'{ROOT_FOLDER}/data/annotation_semisupervised_round2/images'
FOLD = 0

dataDir=Path(DATA_DIR)

cfg.INPUT.MASK_FORMAT='bitmask'
register_coco_instances('sartorius_val',{}, f'{ANN_DIR}/valid/annotations_valid_{FOLD}.json', dataDir)
metadata = MetadataCatalog.get('sartorius_val')
valid_ds = DatasetCatalog.get('sartorius_val')

cfg.MODEL.RESNETS.RADIX = 1
cfg.MODEL.RESNETS.DEEP_STEM = False
cfg.MODEL.RESNETS.AVD = False
# Apply avg_down to the downsampling layer for residual path 
cfg.MODEL.RESNETS.AVG_DOWN = False
cfg.MODEL.RESNETS.BOTTLENECK_WIDTH = 64


cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_ResNeSt200.yaml"))
cfg.DATASETS.TRAIN = ("sartorius_train",)
cfg.DATASETS.TEST = ("sartorius_train", "sartorius_val")
cfg.DATALOADER.NUM_WORKERS = 4
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  

# img size
cfg.INPUT.MIN_SIZE_TEST = 1024
cfg.INPUT.MAX_SIZE_TEST = 2000


cfg.MODEL.WEIGHTS = args.weight
cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.99
cfg.CUSTOM = CfgNode()
cfg.CUSTOM.THRESHOLDS = [0.3,0.5,0.6]
cfg.CUSTOM.NMS_THRESH = [0.1,0.1,0.1]
cfg.TEST.DETECTIONS_PER_IMAGE = 10000


predictor = DefaultPredictor(cfg)