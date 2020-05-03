import argparse
import datetime
import os, sys, glob
import os.path as osp
sys.path.insert(0, '../../')
import torch
import yaml

from models.unet.model import UNet3D
from utils.trainer import Trainer
from utils.generators import DataGenerator
from global_config import global_config
from summary.test_case import test
from summary.case import case
from tqdm import tqdm
from summary.extract import extract

save_dir = './extract'
config = {
    'DEVICE': torch.device('cuda:0'),
    'CAL_DEVICE': torch.device('cuda:1'),
    'IN_LEN': 4,
    'OUT_LEN': 4,
    'BATCH_SIZE': 2,
    'SCALE': 0.2,
    'TASK': 'reg',
    'DIM': '3D',
    'OPTFLOW': False,
}

data_loader = DataGenerator(data_path=global_config['DATA_PATH'], config=config)
n_classes = 1
if 'seg' in config['TASK']:
    n_classes = 4
model = UNet3D(in_channels=1, out_channels=n_classes, final_sigmoid=False, num_levels=3, is_segmentation=False)
model = torch.nn.DataParallel(model, device_ids=[0, 3])
model = model.to(config['DEVICE'])

weight_path = '/home/warit/models/unet3d_4_4/model_last.pth'
model.load_state_dict(torch.load(weight_path, map_location='cuda'))

start_file_name = '20191012_0000.bin'
files = sorted([file for file in glob.glob(global_config['DATA_PATH'])])
extract(model, config, save_dir, files, start_file_name, crop=None)