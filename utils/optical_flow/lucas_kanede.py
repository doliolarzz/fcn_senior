"""
    File name: lucas_kanede.py
    Author: Guodong DU, R-corner, WNI
    Date created: 2018-12-19
    Python Version: 3.6
"""


import math

from numba import jit
import numpy as np

from .func import (im_resize,
                   im_affine,
                   boundary,
                   calc_differential)
from .. import smooth


@jit
def lk_calc(rx, ry, rt, x, y, x_off, y_off):
    dx = rx[y - y_off: y + y_off, x - x_off: x + x_off + 1]
    dy = ry[y - y_off: y + y_off, x - x_off: x + x_off + 1]
    dt = rt[y - y_off: y + y_off, x - x_off: x + x_off + 1]

    # could extend by multiplying weights, to improve stability
    s1 = np.sum(dx ** 2)
    s2 = np.sum(dx * dy)
    s3 = np.sum(dy ** 2)
    s4 = np.sum(dt * dx)
    s5 = np.sum(dt * dy)

    delta = s1 * s3 - s2 ** 2

    if delta == 0:
        u_lk = v_lk = 0
    else:
        u_lk = (s2 * s5 - s3 * s4) / delta
        v_lk = (s2 * s4 - s1 * s5) / delta

    return u_lk, v_lk


@jit
def lk_matrix(rx, ry, rt, x, y, x_off, y_off, method, tau=1e-5):
    # XXu=Xb の一般逆行列問題
    dx = rx[y - y_off: y + y_off, x - x_off: x + x_off + 1]
    dy = ry[y - y_off: y + y_off, x - x_off: x + x_off + 1]
    dt = rt[y - y_off: y + y_off, x - x_off: x + x_off + 1]
    A = np.stack((dx.ravel(), dy.ravel()), axis=1)
    by = dt.ravel().reshape(-1, 1)

    # 擬似行列作成：正方行列
    ATA = np.matmul(A.T, A)

    if method == 'SVD':
        pass
    elif method == 'INVMAT':
        if np.min(np.abs(np.linalg.eigvals(
                ATA))) >= tau:  # ignore small motion # ADD BLOCKS DEALING WITH LARGE OF BY EITHER HALVING OR DOWN-SCALE
            nv = - np.matmul(np.linalg.pinv(A), by)
            u_lk, v_lk = nv.ravel()
        else:
            u_lk, v_lk = 0., 0.

    return u_lk, v_lk


class LucasKanade(object):
    def __init__(self, r=10, sor_coef=1.3, robust_mode=2,
                 dx=2.5, dy=2.5, dt=1.,
                 linalg='inv',
                 **kwargs):
        # placeholders
        self.u = None
        self.v = None

        # parameters
        self.r = r
        self.sor = sor_coef
        self.robust_mode = robust_mode
        self.dx = dx
        self.dy = dy
        self.dt = dt
        self.linalg = linalg
        self.kwargs = kwargs

        self.debug = None

    def __call__(self, img, imgp):
        # initialize self.u and self.v
        self.u = np.zeros_like(img, dtype=np.float64)
        self.v = np.zeros_like(img, dtype=np.float64)

        # update self.u and self.v
        self.lk_base(img, imgp)

    def lk_base(self, img, imgp):
        # calculate gradient
        rx, ry, rt = calc_differential(img, imgp,
                                       self.dx, self.dy, self.dt,
                                       self.robust_mode,
                                       **self.kwargs)
        rx = boundary(rx)
        ry = boundary(ry)
        rt = boundary(rt)

        for i in range(self.r, img.shape[0] - self.r):
            for j in range(self.r, img.shape[1] - self.r):
                if self.linalg == 'inv':
                    uij, vij = lk_matrix(rx, ry, rt, j, i, self.r, self.r, 'INVMAT')
                elif self.linalg == 'direct':
                    uij, vij = lk_calc(rx, ry, rt, j, i, self.r, self.r)
                else:
                    raise Exception('Unrecognized linalg solving method')
                self.u[i, j] = uij
                self.v[i, j] = vij

    def smooth_vel(self, mode, **kwargs):
        self.u, self.v = self._smooth(mode, self.u, self.v, **kwargs)

    @classmethod
    def _smooth(cls, mode, *args, **kwargs):
        res = []
        for a in args:
            if mode == 'fill_average':
                a = smooth.fill_average(a)
            elif mode == 'fill_ma':
                a = smooth.fill_moving_average(a, kwargs['window'], kwargs['smooth_n'])
            elif mode == 'fill_svd':
                a = smooth.fill_svd(a, kwargs['k'], kwargs['th'])
            elif mode == 'ma':
                a = smooth.moving_average(a, kwargs['window'], kwargs['smooth_n'])
            elif mode == 'svd':
                a = smooth.svd(a, kwargs['k'], kwargs['keep_ratio'])
            else:
                raise Exception('Unrecognized smoothing mode')
            res.append(a)
        return res



class PyramidLucasKanade(LucasKanade):
    def __init__(self, max_level=5, pyr_scale=.8,
                 r=10, sor_coef=1.3, robust_mode=2,
                 dx=2.5, dy=2.5, dt=1.,
                 linalg='inv',
                 verbose=True,
                 **kwargs):
        super().__init__(r, sor_coef, robust_mode,
                         dx, dy, dt,
                         linalg,
                         **kwargs)

        self.max_lvl = max_level
        self.pyr_scale = pyr_scale

        self.verbose = verbose

    def __call__(self, img, imgp):
        # get the down-sampling scale for each level
        pyr_scales = [self.pyr_scale ** i for i in range(self.max_lvl)][::-1]

        # get the image size corresponding to each scale
        y_size, x_size = img.shape
        y_sizes = [math.ceil(y_size * i) for i in pyr_scales]
        x_sizes = [math.ceil(x_size * i) for i in pyr_scales]

        # main loop
        for lvl in range(self.max_lvl):
            # downscale inputs
            img_lvl, imgp_lvl, = im_resize((x_sizes[lvl], y_sizes[lvl]), 'down', img, imgp)

            # initialize self.u and self.v
            if lvl == 0:
                self.u = np.zeros_like(img_lvl, dtype=np.float64)
                self.v = np.zeros_like(img_lvl, dtype=np.float64)

            # update self.v and self.v
            if self.verbose is True:
                print('Level: %i: image size: (%i, %i)' % (lvl, img_lvl.shape[0], img_lvl.shape[1]))
            self.lk_base(img_lvl, imgp_lvl)

            # upscale u and v
            if lvl < self.max_lvl - 1:
                self.u, self.v = im_resize((x_sizes[lvl+1], y_sizes[lvl+1]), 'up', self.u, self.v)
                # # adjust the velocity based on physical implication
                # self.u *= self.pyr_scale
                # self.v *= self.pyr_scale
