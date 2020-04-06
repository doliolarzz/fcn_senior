"""
    File name: cross_corr.py
    Author: Guodong DU, R-corner, WNI
    Date created: 2018-12-27
    Python Version: 3.6
"""

from numba import jit
import numpy as np
import cv2

from .func import im_resize


@jit
def _func_cal_cc(im0, im1):
    offset = im0.shape[0]
    ys, xs = im1.shape
    corr = np.zeros((ys-offset, xs-offset))
    for j in range(0, ys - offset):
        for i in range(0, xs - offset):
            im1_ji = im1[j: j+offset, i: i+offset]
            corr[j, i] = np.mean((im0 - im0.mean()) * (im1_ji - im1_ji.mean())) / \
                                           (im0.std() * im1_ji.std() + 1e-13)
    return corr


class CrossCorrelation(object):
    def __init__(self, block_dia, block_skip,
                 corr_threshold=.01, area_threshold=.01, int_threshold=1, ):
        self.u = None
        self.v = None
        self.corr = None

        self.dia = block_dia if block_dia % 2 != 0 else block_dia + 1
        self.r = self.dia // 2
        self.skip = block_skip
        self.corr_th = corr_threshold
        self.area_th = area_threshold
        self.int_th = int_threshold

        self.debug = []

    def __call__(self, img, imgp):
        # image padding
        offset = self.r + 10
        ys_r, xs_r = img.shape
        img = cv2.copyMakeBorder(img,
                                 top=offset, bottom=offset, left=offset, right=offset,
                                 borderType=cv2.BORDER_CONSTANT, value=0)
        imgp = cv2.copyMakeBorder(imgp,
                                  top=offset, bottom=offset, left=offset, right=offset,
                                  borderType=cv2.BORDER_CONSTANT, value=0)

        # get block indices
        y = np.arange(offset, img.shape[0] - offset, self.skip)
        x = np.arange(offset, img.shape[1] - offset, self.skip)

        # initialize u and v
        self.u = np.zeros((len(y), len(x)), dtype=np.float64)
        self.v = self.u.copy()
        self.corr = self.u.copy()

        # main loop
        for yj, j in enumerate(y):
            for xi, i in enumerate(x):
                subblock1 = imgp[j-self.r: j+self.r+1, i-self.r: i+self.r+1]  # sub-block as base image
                subblock2 = img[j-offset: j+offset+1, i-offset: i+offset+1]  # sub-block as moving piece

                # exclude block with low intensity
                check = self._select_area(imgp, self.int_th, self.area_th)
                if check is False:
                    continue

                # calculate velocity and corr
                u, v, corr = self._base(subblock1, subblock2)

                # exclude block with low max-corr
                check = self._select_corr(corr, self.corr_th)
                if check is False:
                    continue

                # fill-in
                self.u[yj, xi] = u
                self.v[yj, xi] = v
                self.corr[yj, xi] = corr

        # up-sample
        self.u, self.v = im_resize((xs_r, ys_r), 'up', self.u, self.v)

        # robust
        self.u = self._robust(self.u)
        self.v = self._robust(self.v)

    def _base(self, subblock1, subblock2):
        # calculate corr
        corr = _func_cal_cc(subblock1, subblock2)
        corr_max = np.max(corr)
        corr_max_index = np.argmax(corr)
        idx_y, idx_x = np.unravel_index(corr_max_index, corr.shape)

        # boundary
        if idx_y == 0:
            idx_y = 1
        elif idx_y == corr.shape[0] - 1:
            idx_y = corr.shape[0] - 2
        if idx_x == 0:
            idx_x = 1
        elif idx_x == corr.shape[1] - 1:
            idx_x = corr.shape[0] - 2

        # calculate vel
        udiv1 = corr[idx_y, idx_x+1] - corr[idx_y, idx_x-1]
        udiv2 = 2. * (corr[idx_y, idx_x-1] - 2. * corr[idx_y, idx_x] + corr[idx_y, idx_x+1])
        vdiv1 = corr[idx_y+1, idx_x] - corr[idx_y-1, idx_x]
        vdiv2 = 2. * (corr[idx_y-1, idx_x] - 2. * corr[idx_y, idx_x] + corr[idx_y+1, idx_x])

        r = corr.shape[0] // 2
        u = idx_x - r + udiv1 / udiv2 if udiv2 != 0. else 0.
        v = idx_y - r + vdiv1 / vdiv2 if vdiv2 != 0. else 0.

        self.debug.append(udiv1 / udiv2)

        return u, v, corr_max

    @classmethod
    def _select_area(cls, img, int_threshold, area_threshold):
        cond = img > int_threshold
        cnt = np.sum(cond)
        total_area = img.shape[0] * img.shape[1]
        if cnt / total_area < area_threshold:
            return False
        else:
            return True

    @classmethod
    def _select_corr(cls, corr, corr_threshold):
        # lower = np.quantile()
        if corr > corr_threshold:
            return True
        else:
            return False

    @classmethod
    def _robust(cls, vel):
        upper = np.quantile(vel[vel != 0], .9)
        lower = np.quantile(vel[vel != 0], .1)
        vel[vel < lower] = upper
        vel[vel > upper] = lower
        return vel
