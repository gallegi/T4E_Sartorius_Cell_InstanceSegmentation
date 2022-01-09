import torch
import sys
from utils.train_net import Trainer
from utils.general import seed_torch
seed_torch(67)

from pathlib import Path
import random, cv2, os
import matplotlib.pyplot as plt
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg, CfgNode
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.modeling.backbone import fpn_resneSt # import to register resneSt backbone

import argparse

ROOT_FOLDER = './'
FOLD = 0

parser = argparse.ArgumentParser(description='Some arguments')
parser.add_argument('--data_dir', type=str, default=f'{ROOT_FOLDER}/data/annotation_semisupervised_round2',
                    help='Path to data folder')
parser.add_argument('--out_dir', type=str, default=f'{ROOT_FOLDER}/models/maskrcnn_ResNeSt200_pseudo_round2_fold{FOLD}',
                    help='Path to output folder where trained weights will be saved')
args = parser.parse_args()

cfg = get_cfg()
ANN_DIR = f'{args.data_dir}/annotations'
IMAGE_DIR = f'{args.data_dir}/images'
cfg.OUTPUT_DIR = args.out_dir

dataDir=Path(IMAGE_DIR)

# ====== Register datasets =======
cfg.INPUT.MASK_FORMAT='bitmask'
register_coco_instances('sartorius_train',{}, f'{ANN_DIR}/train/annotations_train_{FOLD}.json', dataDir)
register_coco_instances('sartorius_val',{}, f'{ANN_DIR}/valid/annotations_valid_{FOLD}.json', dataDir)
metadata = MetadataCatalog.get('sartorius_train')
train_ds = DatasetCatalog.get('sartorius_train')
valid_ds = DatasetCatalog.get('sartorius_val')
# ================================

# ======= Visualize examples ========
d = train_ds[0]
img = cv2.imread(d["file_name"])
visualizer = Visualizer(img[:, :, ::-1], metadata=metadata)
out = visualizer.draw_dataset_dict(d)
plt.figure(figsize = (20,15))
plt.imshow(out.get_image()[:, :, ::-1])
plt.show()

d = valid_ds[0]
img = cv2.imread(d["file_name"])
visualizer = Visualizer(img[:, :, ::-1], metadata=metadata)
out = visualizer.draw_dataset_dict(d)
plt.figure(figsize = (20,15))
plt.imshow(out.get_image()[:, :, ::-1])
plt.show()
# ==================================

# ======== Configuration =========
cfg.MODEL.RESNETS.RADIX = 1
cfg.MODEL.RESNETS.DEEP_STEM = False
cfg.MODEL.RESNETS.AVD = False
# Apply avg_down to the downsampling layer for residual path 
cfg.MODEL.RESNETS.AVG_DOWN = False
cfg.MODEL.RESNETS.BOTTLENECK_WIDTH = 64

cfg.merge_from_file(f"{ROOT_FOLDER}/configs/mask_rcnn_ResNeSt200.yaml")
cfg.DATASETS.TRAIN = ("sartorius_train",)
cfg.DATASETS.TEST = ("sartorius_train", "sartorius_val")
cfg.DATALOADER.NUM_WORKERS = 4
cfg.MODEL.WEIGHTS = "https://livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021/models/Anchor_based/ALL/LIVECell_anchor_based_model.pth"
cfg.SOLVER.IMS_PER_BATCH = 1
cfg.SOLVER.BASE_LR = 0.003
cfg.SOLVER.MAX_ITER = 30000    
cfg.SOLVER.CHECKPOINT_PERIOD = 1000
cfg.TEST.EVAL_PERIOD = 1000
cfg.SOLVER.STEPS = ( 20000,)        
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 1024   
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  

cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE = 1024
cfg.MODEL.RPN.POSITIVE_FRACTION = 0.7

# augment
cfg.INPUT.CROP.ENABLED = True
cfg.INPUT.CROP.SIZE = [0.8, 0.8]

# img size
cfg.INPUT.MIN_SIZE_TRAIN = (1024, )
cfg.INPUT.MAX_SIZE_TRAIN = (2000, )
cfg.INPUT.MIN_SIZE_TEST = 1024
cfg.INPUT.MAX_SIZE_TEST = 2000

cfg.MODEL.BACKBONE.FREEZE_AT = 0

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

print('Number of iterations in 1 epoch:', len(train_ds) // cfg.SOLVER.IMS_PER_BATCH)
# ==========================

# ======= Training =========
trainer = Trainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()
# ==========================
