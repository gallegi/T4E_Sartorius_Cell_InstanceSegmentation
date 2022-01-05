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

from utils.postprocessing import post_process_output
from utils.metric_by_outputs import calculate_AP

from tqdm import tqdm
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


cfg.merge_from_file(f"{ROOT_FOLDER}/configs/mask_rcnn_ResNeSt200.yaml")
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
cfg.CUSTOM.NMS_THRESH = [0.1,0.1,0.1]
cfg.TEST.DETECTIONS_PER_IMAGE = 10000


predictor = DefaultPredictor(cfg)

cfg.CUSTOM.THRESHOLDS = [0.5,0.7,0.8]

list_APs = []
list_TPs = []
list_FPs = []
list_FNs = []
list_logs = []
list_cell_types = []
list_inst_counts = []
list_im_ids = []

i = 0
# for d in tqdm(train_ds, total=len(train_ds)):
for d in tqdm(valid_ds, total=len(valid_ds)):
    im = cv2.imread(d['file_name'])
    
    outputs = predictor(im)  
    outputs = post_process_output(cfg, outputs)
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
#     break

result_df = pd.DataFrame({'image_id':list_im_ids, 'cell_type':list_cell_types, 'inst_count':list_inst_counts,
                            'AP':list_APs, 'TP':list_TPs, 'FP':list_FPs, 'FN':list_FNs,'log':list_logs})

print('Result by each cell type (average precision IOU@0.5:0.95):')
print(result_df.groupby('cell_type').AP.sum() / len(result_df))

print('Result (average precision IOU@0.5:0.95):')
print(result_df.AP.mean())

# outpath = f'{ROOT_FOLDER}/analysis_log/{cfg.OUTPUT_DIR.split("/")[-1]}/valid_results.csv'
# os.makedirs(os.path.dirname(outpath),exist_ok=True)
# result_df.to_csv(outpath, index=False)