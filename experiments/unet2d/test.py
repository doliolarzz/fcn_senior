import argparse
import datetime
import os, sys
import os.path as osp
sys.path.insert(0, '../../')
import torch
import yaml
import numpy as np

from models.unet.model import UNet2D
from utils.generators import DataGenerator
from global_config import global_config
from summary.test import test

save_dir = '/home/warit/fcn/experiments/unet2d/model_logs/logs_4_1_04031532'
config = {
    'DEVICE': torch.device('cuda:0'),
    'CAL_DEVICE': torch.device('cuda:2'),
    'IN_LEN': 4,
    'OUT_LEN': 1,
    'BATCH_SIZE': 2,
    'SCALE': 0.25,
    'TASK': 'reg',
    'DIM': '2D',
}

data_loader = DataGenerator(data_path=global_config['DATA_PATH'], config=config)
model = UNet2D(in_channels=config['IN_LEN'], out_channels=1, final_sigmoid=False, layer_order='gcr', is_segmentation=False)
model = torch.nn.DataParallel(model, device_ids=[0, 3])
model = model.to(config['DEVICE'])

weight_path = save_dir + '/model_42000.pth'
model.load_state_dict(torch.load(weight_path, map_location='cuda'))
test(model, data_loader, config, save_dir, crop=None)