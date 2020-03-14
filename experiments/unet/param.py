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
from torchsummary import summary

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

n_classes = 1
if 'seg' in config['TASK']:
    n_classes = 4
model = UNet(n_channels=config['IN_LEN'], n_classes=n_classes)
model = torch.nn.DataParallel(model, device_ids=[0, 2])
model = model.to(config['DEVICE'])

summary(model, input_size=(3, 512, 672))