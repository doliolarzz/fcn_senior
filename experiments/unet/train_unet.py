import argparse
import datetime
import os, sys
import os.path as osp
sys.path.insert(0, '../../')
import torch
import yaml

from models.unet.unet_model import UNet
from utils.trainer import Trainer
from utils.generators import DataGenerator
from global_config import global_config

here = osp.dirname(osp.abspath(__file__))

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='')
    parser.add_argument('--lr', type=float, default=1.0e-2)
    parser.add_argument('--weight-decay', type=float, default=0.0005)
    parser.add_argument('--momentum', type=float, default=0.99, help='momentum')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--in', type=int, default=5)
    parser.add_argument('--out', type=int, default=1)
    parser.add_argument('--batchsize', type=int, default=4)
    args = vars(parser.parse_args())

    if not os.path.exists('./unet_logs'):
        os.makedirs('./unet_logs')
    save_dir = './unet_logs/logs_' + args['name']
    config = {
        'DEVICE': torch.device(args['device']),
        'IN_LEN': int(args['in']),
        'OUT_LEN': int(args['out']),
        'BATCH_SIZE': int(args['batchsize']),
        'SCALE': 0.25,
        'TASK': 'reg',
    }
    torch.cuda.manual_seed(1337)

    # 1. dataset

    data_loader = DataGenerator(data_path=global_config['DATA_PATH'], config=config)

    # 2. model
    n_classes = 1
    if 'seg' in config['TASK']:
        n_classes = 4
    model = UNet(n_channels=config['IN_LEN'], n_classes=n_classes)
    model = torch.nn.DataParallel(model, device_ids=[0, 2])
    model = model.to(config['DEVICE'])

    # 3. optimizer

    optim = torch.optim.Adam(model.parameters(), lr=args['lr'])


    trainer = Trainer(
        config=config,
        model=model,
        optimizer=optim,
        data_loader=data_loader,
        save_dir=save_dir
    )
    trainer.train()


if __name__ == '__main__':
    main()
