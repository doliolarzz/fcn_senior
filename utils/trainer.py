import datetime
from distutils.version import LooseVersion
import math
import os
import os.path as osp
import shutil

import numpy as np
import pytz
import skimage.io
import torch
import torchvision
from torch.autograd import Variable
import torch.nn.functional as F
from tqdm import tqdm
from utils.losses import cross_entropy2d
from sklearn.metrics import accuracy_score, f1_score
from utils.evaluators import fp_fn_image_csi_muti, fp_fn_image_csi_muti_seg
from utils.units import dbz_mm
from utils.visualizers import rainfall_shade
from tensorboardX import SummaryWriter
from datetime import datetime
from global_config import global_config

class Trainer(object):

    def __init__(
        self,
        config,
        model,
        optimizer,
        data_loader,
        save_dir,
        max_iterations=3,
        interval_validate=50,
        interval_checkpoint=2000
    ):
        self.config              = config
        self.model               = model
        self.optim               = optimizer
        self.data_loader         = data_loader
        self.save_dir            = save_dir + datetime.now().strftime("_%m%d%H%M")
        self.max_iterations      = max_iterations
        self.interval_validate   = interval_validate
        self.interval_checkpoint = interval_checkpoint

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.epoch = 1
        self.iteration = 0
        self.pbar_i = tqdm(range(1, max_iterations + 1))

        self.mse_loss = torch.nn.MSELoss().to(config['DEVICE'])
        self.mae_loss = torch.nn.L1Loss().to(config['DEVICE'])
        self.cat_loss = cross_entropy2d
        self.cat_weight = torch.tensor([1, 20, 50, 100]).float().to(config['DEVICE'])

        self.train_loss = 0
        self.val_loss = 0
        self.best_val_loss = np.inf
        self.metrics_name = ['csi_0', 'csi_1', 'csi_2', 'csi_3']
        self.train_metrics_value = np.zeros(len(self.metrics_name))
        self.val_metrics_value = np.zeros(len(self.metrics_name))

        self.writer = SummaryWriter(os.path.join(self.save_dir, 'train_logs'))
        
    def denorm(self, value):
        return dbz_mm(value * global_config['NORM_DIV'] + global_config['NORM_MIN'])

    def validate(self):

        self.model.eval()
        n_val_batch = self.data_loader.n_val_batch()
        n_val = 20
        self.val_loss = 0
        self.val_metrics_value[:] = 0
        for ib_val, b_val in enumerate(np.random.choice(n_val_batch, n_val)):

            self.pbar_i.set_description("Validating at batch %d / %d" % (ib_val, n_val))
            val_data, val_label = self.data_loader.get_val(b_val)
            with torch.no_grad():
                output = self.model(val_data)

            if self.config['TASK'] == 'seg':
                loss = self.cat_loss(output, val_label, weight=self.cat_weight)
                self.val_loss += loss.data.item() / len(val_data)
                lbl_pred = output.max(1)[1].detach().cpu().numpy()
                lbl_true = val_label.cpu().numpy()[:, 0]
                self.val_metrics_value += fp_fn_image_csi_muti_seg(lbl_pred, lbl_true)
            elif self.config['TASK'] == 'reg':
                loss = self.mse_loss(output[:, 0], val_label[:, 0]) + self.mae_loss(output[:, 0], val_label[:, 0])
                self.val_loss += loss.data.item() / len(val_data)
                lbl_pred = output.detach().cpu().numpy()
                lbl_true = val_label.cpu().numpy()
                self.val_metrics_value += fp_fn_image_csi_muti(self.denorm(lbl_pred), self.denorm(lbl_true))

        self.train_loss /= self.interval_validate
        self.train_metrics_value /= self.interval_validate
        self.val_loss /= n_val
        self.val_metrics_value /= n_val
        self.writer.add_scalars('loss', {
            'train': self.train_loss,
            'valid': self.val_loss
        }, self.epoch)
        for i in range(len(self.metrics_name)):
            self.writer.add_scalars(self.metrics_name[i], {
                'train': self.train_metrics_value[i],
                'valid': self.val_metrics_value[i]
            }, self.epoch)

        if self.config['TASK'] == 'seg':
            img_pred = cv2.cvtColor(np.array(lbl_pred[0] / 4 * 255, dtype=np.uint8), cv2.COLOR_GRAY2RGB)
            img_true = cv2.cvtColor(np.array(lbl_true[0] / 4 * 255, dtype=np.uint8), cv2.COLOR_GRAY2RGB)
            self.writer.add_image('result/pred',
                img_pred.swapaxes(0,2), 
                self.epoch)
            self.writer.add_image('result/true',
                img_true.swapaxes(0,2),
                self.epoch)

        elif self.config['TASK'] == 'reg':
            self.writer.add_image('result/pred',
                rainfall_shade(self.denorm(lbl_pred[0])).swapaxes(0,2), 
                self.epoch)
            self.writer.add_image('result/true',
                rainfall_shade(self.denorm(lbl_true[0])).swapaxes(0,2), 
                self.epoch)

        if self.val_loss <= self.best_val_loss:
            torch.save(self.model.state_dict(), os.path.join(self.save_dir, 
                'model_{}_best.pth'.format(self.epoch)))
            self.best_val_loss = self.val_loss
            
        self.train_loss = 0
        self.train_metrics_value[:] = 0

    def add_epoch(self):

        self.epoch += 1
        if self.epoch % self.interval_validate == 0:
            self.validate()
        if self.epoch % self.interval_checkpoint == 0:
            torch.save(self.model.state_dict(), os.path.join(self.save_dir, 
                'model_{}.pth'.format(self.epoch)))

    def train_iteration(self):

        self.model.train()

        n_train_batch = self.data_loader.n_train_batch()
        pbar_b = tqdm(range(n_train_batch))
        for b in pbar_b:
            pbar_b.set_description('Training at batch %d / %d' % (b, n_train_batch))
            train_data, train_label = self.data_loader.get_train(b)
            self.optim.zero_grad()
            output = self.model(train_data)

            if self.config['TASK'] == 'seg':
                loss = self.cat_loss(output, train_label, weight=self.cat_weight)
            elif self.config['TASK'] == 'reg':
                loss = self.mse_loss(output[:, 0], train_label[:, 0]) + self.mae_loss(output[:, 0], train_label[:, 0])
            
            loss.backward()
            self.optim.step()
            self.train_loss += loss.data.item() / len(train_data)

            if self.config['TASK'] == 'seg':
                lbl_pred = output.max(1)[1].detach().cpu().numpy()
                lbl_true = train_label.cpu().numpy()[:, 0]
                self.train_metrics_value += fp_fn_image_csi_muti_seg(lbl_pred, lbl_true)
            elif self.config['TASK'] == 'reg':
                lbl_pred = output[:, 0].detach().cpu().numpy()
                lbl_true = train_label[:, 0].cpu().numpy()
                self.train_metrics_value += fp_fn_image_csi_muti(self.denorm(lbl_pred), self.denorm(lbl_true))
            self.add_epoch()

    def train(self):
        for i in tqdm(range(self.max_iterations)):
            self.train_iteration()
        self.pbar_i.close()
        self.writer.close()
        torch.save(self.model.state_dict(), os.path.join(self.save_dir, 
            'model_last.pth'.format(self.epoch)))
        
