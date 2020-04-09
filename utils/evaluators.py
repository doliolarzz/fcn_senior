"""
    File name: evaluate.py
    Author: Guodong DU, R-corner, WNI
    Date created: 2019-02-25
    Edited: Nutnaree Kleawsirikul
    Date Edited: 2019-11-25
    Python Version: 3.6
"""


import numpy as np
import torch
from global_config import global_config
import numba
from numba import prange

def hex2Rgb(tn_rgb):
    return list(int(tn_rgb[i:i+2], 16) for i in (0, 2, 4))

def fp_fn_image(gt, pred, config=None, threshold=125):
    # categorize
    gt, pred = gt.copy(), pred.copy()
    gt[gt < threshold] = 0
    gt[gt > threshold] = 1
    pred[pred < threshold] = 0
    pred[pred > threshold] = 1

    # evaluate
    fp = (gt == 0) & (pred == 1)
    fn = (gt == 1) & (pred == 0)
    tp = (gt == 1) & (pred == 1)

    # summarize results
    error = np.zeros_like(gt)
    error[fp] = 1
    error[fn] = 2
    error[tp] = 3

    if config['eval']['plot_color'] is True:
        r, c = error.shape
        rgb_error = np.zeros((r, c, 3), 'uint8')
        rgb_error[..., 0] = error
        rgb_error[..., 1] = error
        rgb_error[..., 2] = error

        tn = np.where((rgb_error[:, :, 0] == 0) & (rgb_error[:, :, 1] == 0) & (rgb_error[:, :, 2] == 0))
        fp = np.where((rgb_error[:, :, 0] == 1) & (rgb_error[:, :, 1] == 1) & (rgb_error[:, :, 2] == 1))
        fn = np.where((rgb_error[:, :, 0] == 2) & (rgb_error[:, :, 1] == 2) & (rgb_error[:, :, 2] == 2))
        tp = np.where((rgb_error[:, :, 0] == 3) & (rgb_error[:, :, 1] == 3) & (rgb_error[:, :, 2] == 3))

        if config==None:
            tn_rgb = '#C6EDE7'
            fp_rgb = '#2832C2'
            fn_rgb = '#FF6464'
            tp_rgb = '#FFFFFF'
        else:
            tn_rgb = config["eval"]["color"]["tn"]
            fp_rgb = config["eval"]["color"]["fp"]
            fn_rgb = config["eval"]["color"]["fn"]
            tp_rgb = config["eval"]["color"]["tp"]

        rgb_error[tn] = hex2Rgb(tn_rgb)
        rgb_error[fp] = hex2Rgb(fp_rgb)
        rgb_error[fn] = hex2Rgb(fn_rgb)
        rgb_error[tp] = hex2Rgb(tp_rgb)

        error = rgb_error

    return error


def fp_fn_image_csi(pred, gt, threshold=1):
    # categorize
    gt, pred = gt.copy(), pred.copy()
    gt[gt < threshold] = 0
    gt[gt >= threshold] = 1
    pred[pred < threshold] = 0
    pred[pred >= threshold] = 1

    # evaluate
    fp = np.sum((gt == 0) & (pred == 1))
    fn = np.sum((gt == 1) & (pred == 0))
    tp = np.sum((gt == 1) & (pred == 1))
    tn = np.sum((gt == 0) & (pred == 0))
    
    csi = float(tp + 1e-4) / (fp + fn + tp + 1e-4) * 100

    return csi

def fp_fn_image_csi_muti(pred, gt):
    bucket = global_config['LEVEL_BUCKET']
    side = global_config['LEVEL_SIDE']
    # categorize
    pred_cat = np.searchsorted(bucket, pred, side=side)
    gt_cat = np.searchsorted(bucket, gt, side=side)

    # evaluate
    all_csi = []
    w = []
    for i in range(len(bucket) + 1):
        gt_e = gt_cat == i
        gt_ne = gt_cat != i
        pred_e = pred_cat == i
        pred_ne = pred_cat != i

        fp = np.sum(gt_ne & pred_e)
        fn = np.sum(gt_e & pred_ne)
        tp = np.sum(gt_e & pred_e)
        tn = np.sum(gt_ne & pred_ne)

        all_csi.append(float(tp + 1e-4) / (fp + fn + tp + 1e-4) * 100)
        w.append(np.sum(gt_e)/gt_cat.size)

    csis = np.array(all_csi)
    return csis, np.sum(csis * np.array(w))

def fp_fn_image_csi_muti_seg(pred, gt):
    # evaluate
    pred = np.array(pred, dtype=np.int)
    gt = np.array(gt, dtype=np.int)
    all_csi = []
    for i in range(len(global_config['LEVEL_BUCKET']) + 1):
        
        fp = np.sum((gt != i) & (pred == i))
        fn = np.sum((gt == i) & (pred != i))
        tp = np.sum((gt == i) & (pred == i))
        tn = np.sum((gt != i) & (pred != i))

        all_csi.append(float(tp + 1e-4) / (fp + fn + tp + 1e-4) * 100)

    return np.array(all_csi)


def fp_fn_image_hit(gt, pred, threshold, mask=None):
    # categorize
    gt, pred = gt.copy(), pred.copy()
    if not (mask is None):
        print("Use mask!!!")
        gt = gt[mask]
        pred = pred[mask]

    gt[gt < threshold] = 0
    gt[gt >= threshold] = 1
    pred[pred < threshold] = 0
    pred[pred >= threshold] = 1

    # evaluate
    fp = np.sum((gt == 0) & (pred == 1))
    fn = np.sum((gt == 1) & (pred == 0))
    tp = np.sum((gt == 1) & (pred == 1))
    tn = np.sum((gt == 0) & (pred == 0))

    hit = float(tp + tn) / (fp + fn + tp + tn) * 100

    return hit

def cal_rmse(pred, label):
    return np.sqrt(np.mean(np.square(pred - label)))

def cal_rmse_all(pred, label, thres=1):
    pred = pred.reshape(-1)
    label = label.reshape(-1)

    mask = label>thres

    rmse = cal_rmse(pred, label)
    rmse_rain = cal_rmse(pred[mask], label[mask])
    rmse_non_rain = cal_rmse(pred[~mask], label[~mask])

    return rmse, rmse_rain, rmse_non_rain

def torch_cal_rmse(yhat,y):
    return torch.sqrt(torch.mean((yhat-y)**2))

def torch_cal_rmse_all(pred, label, thres=1):

    assert pred.size() == label.size()
    
    mask = label>thres

    rmse = torch_cal_rmse(pred, label)
    rmse_rain = torch_cal_rmse(pred[mask], label[mask])
    rmse_non_rain = torch_cal_rmse(pred[~mask], label[~mask])

    return rmse, rmse_rain, rmse_non_rain

def torch_csi_muti(pred, gt):
    bucket = [0] + global_config['LEVEL_BUCKET'].tolist() + [1000]

    # evaluate
    all_csi = []
    w = []
    for i in range(len(bucket) - 1):

        gt_e = (bucket[i] <= gt) & (gt < bucket[i+1])
        gt_ne = ~gt_e
        pred_e = (bucket[i] <= pred) & (gt < bucket[i+1])
        pred_ne = ~pred_e

        fp = torch.sum(gt_ne & pred_e).cpu().numpy()
        fn = torch.sum(gt_e & pred_ne).cpu().numpy()
        tp = torch.sum(gt_e & pred_e).cpu().numpy()
        tn = torch.sum(gt_ne & pred_ne).cpu().numpy()

        all_csi.append(float(tp + 1e-4) / (fp + fn + tp + 1e-4) * 100)
        w.append(torch.sum(gt_e)/gt.numel())

    csis = np.array(all_csi)
    return csis, np.sum(csis * np.array(w))