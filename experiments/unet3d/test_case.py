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

save_dir = './case_result'
config = {
    'DEVICE': torch.device('cuda:0'),
    'IN_LEN': 4,
    'OUT_LEN': 4,
    'BATCH_SIZE': 2,
    'SCALE': 0.2,
    'TASK': 'reg',
    'DIM': '3D',
}

data_loader = DataGenerator(data_path=global_config['DATA_PATH'], config=config)
n_classes = 1
if 'seg' in config['TASK']:
    n_classes = 4
model = UNet3D(in_channels=1, out_channels=n_classes, final_sigmoid=False, num_levels=3, is_segmentation=False)
model = torch.nn.DataParallel(model, device_ids=[0, 2])
model = model.to(config['DEVICE'])

weight_path = '/home/warit/models/logs_4_4_03131408/model_last.pth'
model.load_state_dict(torch.load(weight_path, map_location='cuda'))

files = sorted([file for file in glob.glob(global_config['DATA_PATH'])])
for i in tqdm(case):
    file_name = i[0]
    crop = i[1]
    sp = save_dir + '/' + file_name[:-4]
    if not os.path.exists(sp):
        os.makedirs(sp)
    test(model, data_loader, config, sp, files, file_name, crop=crop)