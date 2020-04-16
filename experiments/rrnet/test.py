import argparse
import datetime
import os, sys
import os.path as osp
sys.path.insert(0, '../../')
import torch
import yaml
import numpy as np

from models.rrnet.model import RRNet
from utils.generators import DataGenerator
from global_config import global_config
from summary.test import test

torch.cuda.set_enabled_lms(True)

save_dir = '/home/warit/fcn/experiments/rrnet/model_logs/logs_5_1_04110300'
config = {
    'DEVICE': torch.device('cuda:0'),
    'CAL_DEVICE': torch.device('cuda:3'),
    'IN_LEN': 5,
    'OUT_LEN': 1,
    'BATCH_SIZE': 1,
    'SCALE': 0.25,
    'DIM': 'RR',
}

data_loader = DataGenerator(data_path=global_config['DATA_PATH'], config=config)
config['IN_HEIGHT'] = int(config['SCALE'] * global_config['DATA_HEIGHT'])
config['IN_WIDTH'] = int(config['SCALE'] * global_config['DATA_WIDTH'])
model = RRNet(config, 8)
# model = torch.nn.DataParallel(model, device_ids=[0, 3])
model = model.to(config['DEVICE'])

weight_path = save_dir + '/model_39000.pth'
model.load_state_dict(torch.load(weight_path, map_location='cuda'))
test(model, data_loader, config, save_dir, crop=None)