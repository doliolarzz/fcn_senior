import torch
import numpy as np

global_config = {
    'DATA_PATH': '/mnt/raid0/wni_dataset/*/*/*.bin',
    'TEST_PATH': '/media/doliolarzz/Ubuntu_data/test/*.bin',
    'IMG_SIZE': 480,
    'STRIDE': 120,
    'OUT_TARGET_LEN': 18,
    'DATA_WIDTH': 2560,
    'DATA_HEIGHT': 3360,
    'lAT_MIN': 20.005,
    'LAT_MAX': 47.9958,
    'LON_MIN': 118.006,
    'LON_MAX': 149.994,
    'MISSINGS': np.load('../weight_npz/missings.npz')['m'],
    'MERGE_WEIGHT': np.load('../weight_finder/learnt_weight_ones.npz')['w'] + 1e-3,
    'LEVEL_BUCKET': np.array([0.2, 10, 30]),
    'LEVEL_SIDE': 'right',
}