import argparse
import datetime
import os, sys, glob
import os.path as osp
sys.path.insert(0, '../../')
import torch
import yaml
import numpy as np

from models.unet.unet_model import UNet
from utils.generators import DataGenerator
from global_config import global_config
from summary.test_case import test
from summary.case import case
from tqdm import tqdm

save_dir = './case_result'
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

weight_path = '/home/warit/models/logs_3_1_03011438/model_8000.pth'
model.load_state_dict(torch.load(weight_path, map_location='cuda'))

files = sorted([file for file in glob.glob(global_config['DATA_PATH'])])
for i in tqdm(case):
    file_name = i[0]
    crop = i[1]
    sp = save_dir + '/' + file_name[:-4]
    if not os.path.exists(sp):
        os.makedirs(sp)
    test(model, data_loader, config, sp, files, file_name, crop=crop)