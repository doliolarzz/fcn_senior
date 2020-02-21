import os, glob
import torch
import cv2
import numpy as np
from global_config import global_config
import itertools
from utils.units import mm_dbz, get_crop_boundary_idx
np.random.seed(42)

class DataGenerator():

    def __init__(self, data_path, config):

        self.config = config
        self.batch_size = config['BATCH_SIZE']
        self.in_len = config['IN_LEN']
        self.out_len = config['OUT_LEN']
        self.windows_size = config['IN_LEN'] + config['OUT_LEN']
        self.files = sorted([file for file in glob.glob(data_path)])
        self.n_files = len(self.files) - self.windows_size + 1
        self.n_val = int(self.n_files / 5)
        self.n_train = self.n_files - self.n_val
        self.last_data = None
        self.shuffle()

    def get_data(self, indices):
        scale = self.config['SCALE']
        h = int(global_config['DATA_HEIGHT'] * scale)
        w = int(global_config['DATA_WIDTH'] * scale)
        sliced_data = np.zeros((len(indices), self.windows_size, h, w), dtype=np.float32)
        for i, idx in enumerate(indices):
            for j in range(self.windows_size):
                f = np.fromfile(self.files[idx + j], dtype=np.float32) \
                    .reshape((global_config['DATA_HEIGHT'], global_config['DATA_WIDTH']))
                sliced_data[i, j] = \
                    cv2.resize(f, (w, h), interpolation = cv2.INTER_AREA)
                
        return (mm_dbz(sliced_data) - global_config['NORM_MIN']) / global_config['NORM_DIV']

    def get_indices(self, idx):

        if self.last_data is not None:
            for i in self.last_data:
                del i
            torch.cuda.empty_cache()

        self.last_data = []
        # idx = self.train_indices[i * self.batch_size : min((i+1) * self.batch_size, self.train_indices.shape[0])]
        data = self.get_data(idx)
        self.last_data.append(torch.from_numpy(data[:, :self.in_len]).to(self.config['DEVICE']))
        self.last_data.append(torch.from_numpy(data[:, self.in_len:]).to(self.config['DEVICE']))
        cat_data = np.searchsorted(mm_dbz(global_config['LEVEL_BUCKET']), data[:, self.in_len:], side=global_config['LEVEL_SIDE'])
        self.last_data.append(torch.from_numpy(cat_data).to(self.config['DEVICE']))
        return tuple(self.last_data)

    def get_train(self, i):
        idx = self.train_indices[i * self.batch_size : min((i+1) * self.batch_size, self.train_indices.shape[0])]
        return self.get_indices(idx)

    def get_val(self, i):
        idx = self.val_indices[i * self.batch_size : min((i+1) * self.batch_size, self.val_indices.shape[0])]
        return self.get_indices(idx)

    def shuffle(self):
        self.train_indices = np.arange(self.n_train)
        np.random.shuffle(self.train_indices)
        self.val_indices = np.arange(self.n_val) + self.n_train

    def n_train_batch(self):
        return int(np.ceil(self.train_indices.shape[0]/self.batch_size))

    def n_val_batch(self):
        return int(np.ceil(self.val_indices.shape[0]/self.batch_size))