import argparse
import datetime
import os, sys
import os.path as osp
sys.path.insert(0, '../../')
import torch
import yaml
import numpy as np

from models.unet.unet_model import UNet
from utils.generators import DataGenerator
from global_config import global_config
from summary.test import test

save_dir = '/home/warit/fcn_senior/experiments/unet/unet_logs/logs_3_1_03011438'
config = {
    'DEVICE': torch.device('cuda:0'),
    'IN_LEN': 3,
    'OUT_LEN': 1,
    'BATCH_SIZE': 2,
    'SCALE': 0.25,
    'TASK': 'reg',
    'DIM': '2D',
}

data_loader = DataGenerator(data_path=global_config['DATA_PATH'], config=config)
n_classes = 1
if 'seg' in config['TASK']:
    n_classes = 4
model = UNet(n_channels=config['IN_LEN'], n_classes=n_classes)
model = torch.nn.DataParallel(model, device_ids=[0, 2])
model = model.to(config['DEVICE'])

weight_path = save_dir + '/model_8000.pth'
model.load_state_dict(torch.load(weight_path, map_location='cuda'))
test(model, data_loader, config, save_dir, crop=None)