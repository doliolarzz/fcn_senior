import argparse
import datetime
import os, sys
import os.path as osp
sys.path.insert(0, '../../')
import torch
import yaml

from models.UNet3DOptGeo.model import UNet3DOptGeo
from utils.trainer import Trainer
from utils.generators import DataGenerator
from global_config import global_config
from summary.test import test

save_dir = '/home/warit/fcn/experiments/UNet3DOptGeo/model_logs/logs_4_4_04231853'
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

weight_path = save_dir + '/model_37500.pth'
model.load_state_dict(torch.load(weight_path, map_location='cuda'))
test(model, data_loader, config, save_dir, crop=None)