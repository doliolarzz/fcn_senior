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

save_dir = './case_result'
config = {
    'DEVICE': torch.device('cuda:0'),
    'CAL_DEVICE': torch.device('cuda:3'),
    'IN_LEN': 5,
    'OUT_LEN': 4,
    'BATCH_SIZE': 2,
    'SCALE': 0.2,
    'TASK': 'reg',
    'DIM': '3D',
    'OPTFLOW': True,
}

data_loader = DataGenerator(data_path=global_config['DATA_PATH'], config=config)
config['IN_HEIGHT'] = int(config['SCALE'] * global_config['DATA_HEIGHT'])
config['IN_WIDTH'] = int(config['SCALE'] * global_config['DATA_WIDTH'])
model = UNet3DOptGeo(config, use_optFlow=True)
model = torch.nn.DataParallel(model, device_ids=[0, 3])
model = model.to(config['DEVICE'])

weight_path = '/home/warit/fcn/experiments/UNet3DOptGeo/model_logs/logs_4_4_04231853/model_37500.pth'
model.load_state_dict(torch.load(weight_path, map_location='cuda'))

files = sorted([file for file in glob.glob(global_config['DATA_PATH'])])
for i in tqdm(case):
    file_name = i[0]
    crop = i[1]
    sp = save_dir + '/' + file_name[:-4]
    if not os.path.exists(sp):
        os.makedirs(sp)
    test(model, data_loader, config, sp, files, file_name, crop=crop)