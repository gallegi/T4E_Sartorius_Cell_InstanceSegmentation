import numpy as np
import detectron2
from pathlib import Path
import random, cv2, os
import matplotlib.pyplot as plt
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.structures.instances import Instances

import pandas as pd
import pickle
from tqdm import tqdm

from sklearn.model_selection import cross_val_score
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error, precision_score, roc_auc_score

import seaborn as sns

from utils.metric_by_outputs import calculate_AP
from utils.feature_engineer_2nd_level_model import get_features, calculate_max_IOU_with_gt

# VER = '5_ens_m1022_m1018'

ROOT_FOLDER = './'
ANN_DIR = f'{ROOT_FOLDER}/data'
DATA_DIR = f'{ROOT_FOLDER}/data/annotation_semisupervised_round2/images'

print('Training 2nd level model on the first half, and validate on the second half')

dataDir=Path(DATA_DIR)

register_coco_instances('sartorius_val_1',{}, f'{ANN_DIR}/val_fold0_split_part1.json', dataDir)
register_coco_instances('sartorius_val_2',{}, f'{ANN_DIR}/val_fold0_split_part2.json', dataDir)

metadata = MetadataCatalog.get('sartorius_val_1')
valid_ds1 = DatasetCatalog.get('sartorius_val_1')
valid_ds2 = DatasetCatalog.get('sartorius_val_2')

def get_pkl_folder():
    return f'{ROOT_FOLDER}/data_for_2nd_level_model/'


train_df = pd.DataFrame()

i = 0
for d in tqdm(valid_ds1, total=len(valid_ds1)):
    PKL_FOLDER = get_pkl_folder()
    path = f'{PKL_FOLDER}/{d["image_id"]}.pkl'
    
    im = cv2.imread(d['file_name'])

    with open(path, 'rb') as f:
        outputs = pickle.load(f)
        outputs['instances'] = outputs['instances'].to('cpu')
        
    features = get_features(im, outputs)
    
    features['image_id'] = [d['image_id']]*len(features)
    features['instance_num'] = np.arange(0, len(features))
    features['cell_type'] = [d['annotations'][0]['category_id']]*len(features)
    features['iou'] = calculate_max_IOU_with_gt(d['annotations'], outputs)
    
    train_df = pd.concat([train_df, features], axis=0)

    i+=1
    if i == 10:
        break


valid_df = pd.DataFrame()

i = 0
for d in tqdm(valid_ds2, total=len(valid_ds2)):
    PKL_FOLDER = get_pkl_folder()
    path = f'{PKL_FOLDER}/{d["image_id"]}.pkl'
    
    im = cv2.imread(d['file_name'])

    with open(path, 'rb') as f:
        outputs = pickle.load(f)
        outputs['instances'] = outputs['instances'].to('cpu')
        
    features = get_features(im, outputs)
    features['image_id'] = [d['image_id']]*len(features)
    features['instance_num'] = np.arange(0, len(features))
    features['cell_type'] = [d['annotations'][0]['category_id']]*len(features)
    features['iou'] = calculate_max_IOU_with_gt(d['annotations'], outputs)
    
    valid_df = pd.concat([valid_df, features], axis=0)

    i+=1
    if i == 10:
        break

print('Number of instance on train set:', len(train_df))
print('Number of instance on valid set:', len(valid_df))


excluded = ['image_id', 'iou', 'instance_num']
X_cols = []
for col in train_df.columns:
    if(col not in excluded):
        X_cols.append(col)
y_col = 'iou'

# detect cols with many nulls (> 30%)
null_count = train_df[X_cols].isnull().mean().sort_values(ascending=False)
cols_many_nulls = null_count[null_count > 0.3].index.tolist()
print(len(cols_many_nulls))


# detect cols with many same values (>90%)
max_counts = []

for col in X_cols:
    counts = train_df[col].value_counts()
    if(len(counts) > 0):
        max_count = counts.iloc[0]
    max_counts.append(max_count / len(train_df))

freq = pd.DataFrame({'feature':X_cols, 'max_freq':max_counts})
cols_same_values = freq[freq.max_freq > 0.9].feature.tolist()
print(len(cols_same_values))

X_cols2 = [col for col in X_cols if col not in cols_many_nulls and col not in cols_same_values]


model = CatBoostRegressor(n_estimators=700, max_depth=2, 
                        learning_rate=0.02, reg_lambda=1,
                        loss_function='RMSE',
                         verbose=False, 
                        random_state=67)

X_train = train_df[X_cols2].fillna(-1)
y_train = train_df['iou']

X_valid = valid_df[X_cols2].fillna(-1)
y_valid = valid_df['iou']

model.fit(X_train,y_train)
y_pred_train = model.predict(X_train)
y_pred_valid = model.predict(X_valid)

ft_imp = pd.DataFrame({'features':X_cols2, 'importance':model.feature_importances_})
ft_imp = ft_imp.sort_values('importance', ascending=False)
TOP = 10
print(f'Top {TOP} most important features:')
print(ft_imp['features'].iloc[:TOP])

train_df['pred_iou'] = y_pred_train
valid_df['pred_iou'] = y_pred_valid

train_df['FP'] = train_df['iou'] < 0.5
valid_df['FP'] = valid_df['iou'] < 0.5

train_df['pred_FP'] = train_df['pred_iou'] < 0.3
valid_df['pred_FP'] = valid_df['pred_iou'] < 0.3

train_precision = precision_score(train_df['FP'], train_df['pred_FP'])
valid_precision = precision_score(valid_df['FP'], valid_df['pred_FP'])

print('Train precision:', train_precision)
print('Valid precision:', valid_precision)

all_df = pd.concat([train_df, valid_df], axis=0)
model.fit(all_df[X_cols2].fillna(-1), all_df[y_col])

with open(os.path.join('2nd_level_model/', f'catboost.pkl'), 'wb') as f:
    pickle.dump(model, f)

pd.DataFrame(X_cols2, columns=['features']).to_csv(os.path.join('2nd_level_model/', f'features.csv'), index=False)