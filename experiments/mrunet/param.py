import argparse
import datetime
import os, sys, glob
import os.path as osp
sys.path.insert(0, '../../')
import torch
import yaml
import numpy as np

from models.mrunet.model import MRUNet
from utils.generators import DataGenerator
from global_config import global_config
from summary.test_case import test
from summary.case import case
from tqdm import tqdm
from torchsummary import summary

config = {
    'DEVICE': torch.device('cuda:0'),
    'IN_LEN': 3,
    'OUT_LEN': 1,
    'BATCH_SIZE': 2,
    'SCALE': 0.25,
    'TASK': 'reg',
    'DIM': '2D',
}
config['IN_HEIGHT'] = int(config['SCALE'] * global_config['DATA_HEIGHT'])
config['IN_WIDTH'] = int(config['SCALE'] * global_config['DATA_WIDTH'])
model = MRUNet(config)

summary(model, input_size=(config['IN_LEN'], config['IN_HEIGHT'], config['IN_WIDTH']))