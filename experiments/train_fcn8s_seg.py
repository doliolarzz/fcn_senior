import argparse
import datetime
import os, sys
import os.path as osp
sys.path.insert(0, '../')
import torch
import yaml

from models.fcn8s import FCN8s, FCN8sAtOnce
from models.fcn16s import FCN16s
from models.fcn32s import FCN32s
from utils.trainer import Trainer
from utils.generators import DataGenerator
from global_config import global_config

def get_parameters(model, bias=False):
    import torch.nn as nn
    modules_skipped = (
        nn.ReLU,
        nn.MaxPool2d,
        nn.Dropout2d,
        nn.Sequential,
        FCN32s,
        FCN16s,
        FCN8s,
    )
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            if bias:
                yield m.bias
            else:
                yield m.weight
        elif isinstance(m, nn.ConvTranspose2d):
            # weight is frozen because it is just a bilinear upsampling
            if bias:
                assert m.bias is None
        elif isinstance(m, modules_skipped):
            continue
        else:
            raise ValueError('Unexpected module: %s' % str(m))


here = osp.dirname(osp.abspath(__file__))

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='')
    parser.add_argument('--lr', type=float, default=1.0e-4)
    parser.add_argument('--weight-decay', type=float, default=0.0005)
    parser.add_argument('--momentum', type=float, default=0.99, help='momentum')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--in', type=int, default=5)
    parser.add_argument('--out', type=int, default=1)
    parser.add_argument('--batchsize', type=int, default=4)
    args = vars(parser.parse_args())

    if not os.path.exists('./fcn_logs'):
        os.makedirs('./fcn_logs')
    save_dir = './fcn_logs/logs_' + args['name']
    config = {
        'DEVICE': torch.device(args['device']),
        'IN_LEN': int(args['in']),
        'OUT_LEN': int(args['out']),
        'BATCH_SIZE': int(args['batchsize']),
        'SCALE': 0.25,
    }
    torch.cuda.manual_seed(1337)

    # 1. dataset

    data_loader = DataGenerator(data_path=global_config['DATA_PATH'], config=config)

    # 2. model

    model = FCN8s(n_class=4, n_channel=config['IN_LEN'])
    model = model.to(config['DEVICE'])

    # 3. optimizer

#     optim = torch.optim.SGD(
#         [
#             {'params': get_parameters(model, bias=False)},
#             {'params': get_parameters(model, bias=True),
#              'lr': args['lr'] * 2, 'weight_decay': 0},
#         ],
#         lr=args['lr'],
#         momentum=args['momentum'],
#         weight_decay=args['weight_decay'])
    optim = torch.optim.Adam(get_parameters(model, bias=True), lr=args['lr'])

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
