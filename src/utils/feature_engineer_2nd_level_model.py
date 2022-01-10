
'''
    Support functions for 2nd-level feature engineering
'''
import pandas as pd
import pycocotools.mask as mask_util
from tqdm import tqdm
from sklearn.neighbors import KDTree
import numpy as np
import cv2

def calculate_max_IOU_with_gt(targ, pred):
    '''Calculate IOU between predicted instances and target instances'''
    pred_masks = pred['instances'].pred_masks >= 0.5
    enc_preds = [mask_util.encode(np.asarray(p, order='F')) for p in pred_masks]
    enc_targs = list(map(lambda x:x['segmentation'], targ))
    ious = mask_util.iou(enc_preds, enc_targs, [0]*len(enc_targs))
    
    return ious.max(axis=1)

def print_log(log):
    for k in log.keys():
        print(k, log[k])
        
def get_overlapping_features(pred):
    '''Compute features representing overlapping characteristics of each instance'''
    pred_masks = pred['instances'].pred_masks >= 0.5
    enc_preds = [mask_util.encode(np.asarray(p, order='F')) for p in pred_masks]
    ious = mask_util.iou(enc_preds, enc_preds, [0]*len(enc_preds))
    return ious.max(axis=1), ious.min(axis=1), ious.mean(axis=1),\
             ious.std(axis=1), (ious > 0).sum(axis=1)

def get_contour_features(pred):
    '''Get some morphology features'''
    masks = (pred['instances'].pred_masks.numpy() >= 0.5).astype('uint8')
    
    data_dict = {
        'centroid_x':[],
        'centroid_y':[],
        'num_contours': [],
        'equi_diameter':[],
        'hull_area':[],
        'solidity':[],
        'is_convex':[],
        'perimeter':[],
        'rotation_ang':[],
        'major_axis_length':[],
        'minor_axis_length':[]
    }
    
    for mask in masks:
        contours, _ = cv2.findContours(mask, 1, 2)
        areas = [cv2.contourArea(cnt) for cnt in contours]
        max_ind = np.argmax(areas)
        
        area = areas[max_ind]
        cnt = contours[max_ind]
        
        M = cv2.moments(cnt)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = float(area)/hull_area if hull_area > 0 else -1
        
        equi_diameter = np.sqrt(4*area/np.pi)
        is_convex = int(cv2.isContourConvex(cnt))
        perimeter = cv2.arcLength(cnt,True)
        
        try:
            ellipse = cv2.fitEllipse(cnt)
            _,(major_axis_length, minor_axis_length), rotation_ang = ellipse
        except:
            (major_axis_length, minor_axis_length), rotation_ang = (-1,-1),-1
        
        data_dict['centroid_x'].append(cx)
        data_dict['centroid_y'].append(cy)
        data_dict['num_contours'].append(len(contours))
        data_dict['equi_diameter'].append(equi_diameter)
        data_dict['solidity'].append(solidity)
        data_dict['hull_area'].append(hull_area)
        data_dict['is_convex'].append(is_convex)
        data_dict['perimeter'].append(perimeter)
        data_dict['rotation_ang'].append(rotation_ang)
        data_dict['major_axis_length'].append(major_axis_length)
        data_dict['minor_axis_length'].append(minor_axis_length)
        
    return pd.DataFrame(data_dict)

def get_pixel_scores_features(outputs):
    '''Get features related to mask scores at pixel level'''
    pred_masks = outputs['instances'].pred_masks
    pred_masks_non_zeros = [mask[mask > 0] for mask in pred_masks]
    min_pscores = [mask.min().item() for mask in pred_masks_non_zeros]
    max_pscores = [mask.max().item() for mask in pred_masks_non_zeros]
    median_pscores = [mask.median().item() for mask in pred_masks_non_zeros]
    mean_pscores = [mask.mean().item() for mask in pred_masks_non_zeros]
                    
    q1_pscores =  [mask.quantile(0.25).item() for mask in pred_masks_non_zeros]
    q3_pscores =  [mask.quantile(0.75).item() for mask in pred_masks_non_zeros]
    std_pscores = [mask.std().item() for mask in pred_masks_non_zeros]
    
    ret = {
        'min_pixel_score':min_pscores,
        'max_pixel_score':max_pscores,
        'median_pixel_score':median_pscores,
        'mean_pixel_score':mean_pscores,
        'q1_pixel_score':q1_pscores,
        'q3_pixel_score':q3_pscores,
        'std_pixel_score':std_pscores
    }
    return pd.DataFrame(ret)

def get_image_pixel_features(im, outputs):
    '''Get features related to pixels on the original images'''
    pred_masks = outputs['instances'].pred_masks
    pred_masks_binary = [mask > 0.5 for mask in pred_masks]
    im_masks = [im[mask,0] for mask in pred_masks_binary]
    
    min_pscores = [mask.min().item() for mask in im_masks]
    max_pscores = [mask.max().item() for mask in im_masks]
    median_pscores = [np.median(mask).item() for mask in im_masks]
    mean_pscores = [mask.mean().item() for mask in im_masks]
                    
    q1_pscores =  [np.quantile(mask, 0.25).item() for mask in im_masks]
    q3_pscores =  [np.quantile(mask, 0.75) for mask in im_masks]
    std_pscores = [mask.std() for mask in im_masks]
    
    ret = {
        'im_min_pixel':min_pscores,
        'im_max_pixel':max_pscores,
        'im_median_pixel':median_pscores,
        'im_mean_pixel':mean_pscores,
        'im_q1_pixel':q1_pscores,
        'im_q3_pixel':q3_pscores,
        'im_std_pixel':std_pscores
    }
    return pd.DataFrame(ret)

def get_kdtree_nb_features(single_features):
    '''Get features related to neighboring relation ship determine by distance'''
    cols = ['centroid_x', 'centroid_y']
    X = single_features[cols]
    tree = KDTree(X)  
    
    ret = dict()
    for r in [25, 50, 75, 100, 150, 200]:
        ind, dist = tree.query_radius(X, r=r, return_distance=True, sort_results=True)  
        ind = [i[1:] for i in ind] # exclude neareast neighbor (itself)
        dist = [d[1:] for d in dist] # exclude neareast neighbor (itself)
        
        ret[f'kdtree_nb_r{r}_count'] = [len(ind) for i in ind]
        
        ret[f'kdtree_nb_r{r}_median_dist'] = [np.median(d) if len(d)>0 else -1 for d in dist]
        ret[f'kdtree_nb_r{r}_mean_dist'] = [d.mean() if len(d)>0 else -1 for d in dist]
        ret[f'kdtree_nb_r{r}_std_dist'] = [np.std(d) if len(d)>0 else -1 for d in dist]
        
        ret[f'kdtree_nb_r{r}_median_area'] = [single_features.loc[i, 'mask_area'].median() if len(i)>0 else -1 for i in ind]
        ret[f'kdtree_nb_r{r}_mean_area'] = [single_features.loc[i, 'mask_area'].mean() if len(i)>0 else -1 for i in ind]
        ret[f'kdtree_nb_r{r}_std_area'] = [single_features.loc[i, 'mask_area'].std() if len(i)>0 else -1 for i in ind]
        
        ret[f'kdtree_nb_r{r}_median_box_score'] = [single_features.loc[i, 'box_score'].median() if len(i)>0 else -1 for i in ind]
        ret[f'kdtree_nb_r{r}_mean_box_score'] = [single_features.loc[i, 'box_score'].mean() if len(i)>0 else -1 for i in ind]
        ret[f'kdtree_nb_r{r}_std_box_score'] = [single_features.loc[i, 'box_score'].std() if len(i)>0 else -1 for i in ind]
        
    for k in [2,3,5,7]:
        dist, ind = tree.query(X, k=k, return_distance=True)  
        ind = [i[1:] for i in ind] # exclude neareast neighbor (itself)
        dist = [d[1:] for d in dist] # exclude neareast neighbor (itself)
        
        ret[f'kdtree_nb_top{k}_median_dist'] = [np.median(d) if len(d)>0 else -1 for d in dist]
        ret[f'kdtree_nb_top{k}_mean_dist'] = [d.mean() if len(d)>0 else -1  for d in dist]
        ret[f'kdtree_nb_top{k}_std_dist'] = [np.std(d) if len(d)>0 else -1 for d in dist]
        
        ret[f'kdtree_nb_top{k}_median_area'] = [single_features.loc[i, 'mask_area'].median() if len(i)>0 else -1 for i in ind]
        ret[f'kdtree_nb_top{k}_mean_area'] = [single_features.loc[i, 'mask_area'].mean() if len(i)>0 else -1 for i in ind]
        ret[f'kdtree_nb_top{k}_std_area'] = [single_features.loc[i, 'mask_area'].std() if len(i)>0 else -1 for i in ind]
        
        ret[f'kdtree_nb_top{k}_median_box_score'] = [single_features.loc[i, 'box_score'].median() if len(i)>0 else -1 for i in ind]
        ret[f'kdtree_nb_top{k}_mean_box_score'] = [single_features.loc[i, 'box_score'].mean() if len(i)>0 else -1 for i in ind]
        ret[f'kdtree_nb_top{k}_std_box_score'] = [single_features.loc[i, 'box_score'].std() if len(i)>0 else -1 for i in ind]
        
    return pd.DataFrame(ret)    
    
def get_features(im, outputs):
    '''Master function for generating features'''
    pred_masks = outputs['instances'].pred_masks
    
    mask_areas = (pred_masks >= 0.5).sum(axis=(1,2))
    pred_boxes = outputs['instances'].pred_boxes.tensor
    widths = pred_boxes[:,2] - pred_boxes[:,0]
    heights = pred_boxes[:,3] - pred_boxes[:,1]
    box_areas = widths * heights
    
    box_scores = outputs['instances'].scores
    instance_count = len(outputs['instances'])
    
    aspect_ratios = widths / heights
    extents = mask_areas / box_areas
    
    neighbor_iou_max, neighbor_iou_min, neighbor_iou_mean, \
        neighbor_iou_std, neighbor_overlap_count = get_overlapping_features(outputs)
    
    contour_features = get_contour_features(outputs)
    pixel_features = get_pixel_scores_features(outputs)
    im_pixel_features = get_image_pixel_features(im, outputs)
    
    ret = pd.DataFrame({
        'box_score':box_scores,
        'mask_area':mask_areas,
        'box_area':box_areas,
        'box_x1':pred_boxes[:,0],
        'box_y1':pred_boxes[:,1],
        'box_x2':pred_boxes[:,2],
        'box_y2':pred_boxes[:,3],
        'width':widths,
        'height':heights,
        'instance_count':instance_count,
        
        'neighbor_iou_max':neighbor_iou_max,
        'neighbor_iou_min':neighbor_iou_min,
        'neighbor_iou_mean':neighbor_iou_mean,
        'neighbor_iou_std':neighbor_iou_std,
        'neighbor_overlap_count':neighbor_overlap_count,
        
        'aspect_ratio':aspect_ratios,
        'extent':extents
    })
    
    ret = pd.concat([ret, contour_features, pixel_features, im_pixel_features], axis=1)
    kdtree_nb_features = get_kdtree_nb_features(ret)
    ret = pd.concat([ret, kdtree_nb_features], axis=1)
    return ret

FEATURES_TO_AGG = ['box_score', 'mask_area', 'width', 'height']

def get_aggregation_features(features):
    '''Generate aggregation features by some other fields'''
    ret = features.copy()
    for agg_scheme in ['image_id', 'cell_type']:
        agg_ft = features.groupby(agg_scheme)[FEATURES_TO_AGG].agg(['mean', 'median'])
        agg_ft.columns = [f"agg_by_{agg_scheme}_"+"_".join(a) for a in agg_ft.columns.to_flat_index()]
        agg_ft = agg_ft.reset_index()
        
        ret = ret.merge(agg_ft, on=agg_scheme)
    return ret