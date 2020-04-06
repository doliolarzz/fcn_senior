import argparse
import datetime
import os, sys
import os.path as osp
sys.path.insert(0, '../../')
import torch
import yaml
import numpy as np

from models.hrnet.config import config as cfg, update_config
from models.hrnet.seg_hrnet import get_seg_model
from utils.generators import DataGenerator
from global_config import global_config
from summary.test import test

save_dir = '/home/warit/fcn/experiments/hrnet/model_logs/logs_4_1_03291123'
config = {
    'DEVICE': torch.device('cuda:0'),
    'CAL_DEVICE': torch.device('cuda:2'),
    'IN_LEN': 4,
    'OUT_LEN': 1,
    'BATCH_SIZE': 2,
    'SCALE': 0.25,
    'TASK': 'reg',
    'DIM': 'HR',
}

data_loader = DataGenerator(data_path=global_config['DATA_PATH'], config=config)
update_config(cfg, { 'cfg': './params3.yaml' })
model = get_seg_model(cfg)
model = torch.nn.DataParallel(model, device_ids=[0, 3])
model = model.to(config['DEVICE'])

weight_path = save_dir + '/model_28500.pth'
model.load_state_dict(torch.load(weight_path, map_location='cuda'))
test(model, data_loader, config, save_dir, crop=None)