import argparse
import datetime
import os, sys, glob
import os.path as osp
sys.path.insert(0, '../../')
import torch
import yaml

from models.UNet3DOptGeo.model import UNet3DOptGeo
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
    'IN_LEN': 5,
    'OUT_LEN': 4,
    'BATCH_SIZE': 2,
    'SCALE': 0.2,
    'TASK': 'reg',
    'DIM': '3D',
    'OPTFLOW': True,
}

data_loader = DataGenerator(data_path=global_config['DATA_PATH'], config=config)
n_classes = 1
if 'seg' in config['TASK']:
    n_classes = 4
config['IN_HEIGHT'] = int(config['SCALE'] * global_config['DATA_HEIGHT'])
config['IN_WIDTH'] = int(config['SCALE'] * global_config['DATA_WIDTH'])
model = UNet3DOptGeo(config, use_optFlow=True)
model = torch.nn.DataParallel(model, device_ids=[0, 3])
model = model.to(config['DEVICE'])

weight_path = '/home/warit/fcn/experiments/UNet3DOptGeo/model_logs/logs_4_4_04231853/model_37500.pth'
model.load_state_dict(torch.load(weight_path, map_location='cuda'))

start_file_name = '20191012_0000.bin'
files = sorted([file for file in glob.glob(global_config['DATA_PATH'])])
extract(model, config, save_dir, files, start_file_name, crop=None)