"""
    File name: horn_schunck.py
    Author: Guodong DU, R-corner, WNI
    Date created: 2018-12-19
    Python Version: 3.6
"""

import matplotlib.pyplot as plt
import math
import time

import numpy as np
from numba import jit
from scipy import signal

from .func import (im_resize,
                   im_affine,
                   boundary,
                   calc_differential)
from src import smooth #edited


class HornSchunck(object):

    __slots__ = [
        'u', 'v', 'alpt', 'k',
        'dx', 'dy', 'dt',
        'block_dia', 'block_skip',
        'max_iter', 'early_stop',
        'thread', 'robust_mode',
        'verbose',
        'kwargs'
    ]

    def __init__(self, dx=2.5, dy=2.5, dt=1.,
                 block_dia=65, block_skip=33,
                 max_iter=1200, early_stop=1e-5,
                 thread=2, robust_mode=2,
                 verbose=True,
                 **kwargs):
        """

        :param dx:
        :param dy:
        :param dt:
        :param black_dia:
        :param block_skip:
        :param max_iter:
        :param early_stop:
        :param thread:
        :param robust_mode:
        :param kwargs:
        """
        # placeholders
        self.u = None
        self.v = None

        self.alpt = None
        self.k = None

        # parameters
        self.max_iter = max_iter
        self.early_stop = early_stop
        self.thread = thread
        self.robust_mode = robust_mode
        self.dx = dx
        self.dy = dy
        self.dt = dt
        self.block_dia = block_dia
        self.block_skip = block_skip
        self.verbose = verbose
        self.kwargs = kwargs

    def __call__(self, img, imgp, alpt=None):
        # get alpha
        if alpt is None:
            self._calc_alpha(img)
        else:
            self.alpt = np.zeros_like(img, dtype=np.float64)
            self.alpt.fill(alpt)

        # initialize self.u and self.v
        self.u = np.zeros_like(img, dtype=np.float64)
        self.v = np.zeros_like(img, dtype=np.float64)

        # update self.u and self.v
        self._hs_base(img, imgp, self.alpt)

    def _hs_base(self, img, imgp, alpt):
        # get parameters
        eval_threshold = self.kwargs.pop('early_stop_eval_threshold', .5)

        # calculate gradient
        rx, ry, rt = calc_differential(img, imgp,
                                       self.dx, self.dy, self.dt,
                                       self.robust_mode,
                                       **self.kwargs)
        rx = boundary(rx) # extend boundary
        ry = boundary(ry)
        rt = boundary(rt)

        erra = []   # stoping cond
        start = time.time()
        # max_iter and eval_threshold
        for k in range(self.max_iter):
            cond = img > eval_threshold # stoppoing threshold
            avg_x = np.mean(self.u[cond])
            avg_y = np.mean(self.v[cond])
            erra.append((avg_x ** 2 + avg_y ** 2) ** .5)
            update = abs(erra[k] - erra[k - 1]) / (erra[k - 1] + 1e-13)

            if k % (self.max_iter // 10) == 0:
                if self.verbose is True:
                    print('update of velocity: %.7f at step %i, time elapsed %.3f min' %
                          (update, k, (time.time() - start) / 60))
            if k > 0:
                if update < self.early_stop:
                    if self.verbose is True:
                        print('stop at step', k)
                    break

            # updating velocity
            if self.thread == 1:
                self._hs_thread_1(rx, ry, rt, alpt)
            elif self.thread == 2:
                self._hs_thread_2(rx, ry, rt, alpt, self.dx, self.dy)
            else:
                raise Exception('Unrecognized thread mode')

    def _hs_thread_1(self, rx, ry, rt, alpt):
        """
        First-order scheme
        :param u: np.ndarray, velocity
        :param v: np.ndarray, velocity
        :param rx: np.ndarray, gradient along dim x
        :param ry: np.ndarray, gradient along dim y
        :param rt: np.ndarray, gradient along dim t
        :param alpt: np.ndarray ???, regularization coefficients
        :return:
        """
        # get parameter
        sor_coef = self.kwargs.pop('sor_coef', 1.3) # acceleration coefficient, boost or slow, > 1 boost, <1 slow

        # calculate average u and v
        ua, va = self._uv_avg(self.u, self.v)  # ua and va have size like u[1:-1, 1:-1]

        # extract working domain
        u, v = self.u[1:-1, 1:-1], self.v[1:-1, 1:-1]
        rx, ry, rt = rx[1:-1, 1:-1], ry[1:-1, 1:-1], rt[1:-1, 1:-1]
        alpt = alpt[1:-1, 1:-1]

        # calculate update
        div1 = rx * ua + ry * va + rt
        div2 = alpt ** 2 + rx ** 2 + ry ** 2
        bufx = ua - rx * div1 / (div2 + 1e-13)
        bufy = va - ry * div1 / (div2 + 1e-13)

        # apply update
        u_calc = (1. - sor_coef) * u + sor_coef * bufx
        v_calc = (1. - sor_coef) * v + sor_coef * bufy

        # get back to attribute u and v
        cond = div2 != 0
        if np.sum(~cond) != 0:
            self.u[1:-1, 1:-1] = np.where(cond, u_calc, u)
            self.v[1:-1, 1:-1] = np.where(cond, v_calc, v)
        else:
            self.u[1:-1, 1:-1] = u_calc.copy()
            self.v[1:-1, 1:-1] = v_calc.copy()
        self.u = boundary(self.u)
        self.v = boundary(self.v)

    def _hs_thread_2(self, rx, ry, rt, alpt, dx, dy):
        """
        Second-order scheme?
        αを速度の成分方向ごとに独立に与えるようにする．
        速度の分母にも，格子幅を入れる．
        :param u: np.ndarray, velocity
        :param v: np.ndarray, velocity
        :param rx: np.ndarray, gradient along dim x
        :param ry: np.ndarray, gradient along dim y
        :param rt: np.ndarray, gradient along dim t
        :param alpt: np.ndarray ???, regularization coefficients
        :param dx: float, step size at dim x
        :param dy: float, step size at dim y
        :return: u and v
        """
        # calculate average u and v
        ua, va = self._uv_avg(self.u, self.v)  # ua and va have size like u[1:-1, 1:-1]

        # extract working domain
        rx, ry, rt = rx[1:-1, 1:-1], ry[1:-1, 1:-1], rt[1:-1, 1:-1]
        alpt = alpt[1:-1, 1:-1]

        # ???
        dx2 = 1. / dx ** 2
        dy2 = 1. / dy ** 2

        alpx = alpt.copy()
        alpy = alpt.copy()
        alpx2 = alpx ** 2
        alpy2 = alpy ** 2

        # ???
        ur = - rx * ry * va - \
             rx * rt + \
             alpx2 * (self.u[1:-1, 2:] + self.u[1:-1, :-2]) * dx2 + \
             alpy2 * (self.u[2:, 1:-1] + self.u[:-2, 1:-1]) * dy2
        vr = - rx * ry * ua - \
             ry * rt + \
             alpx2 * (self.v[1:-1, 2:] + self.v[1:-1, :-2]) * dx2 + \
             alpy2 * (self.v[2:, 1:-1] + self.v[:-2, 1:-1]) * dy2

        udiv = rx ** 2 + alpx2 * dx2 * 2 + alpy2 * dy2 * 2
        vdiv = ry ** 2 + alpx2 * dx2 * 2 + alpy2 * dy2 * 2

        u_calc = ur / (udiv + 1e-13)
        v_calc = vr / (vdiv + 1e-13)

        # get back to attribute u and v
        u_cond = udiv != 0
        v_cond = vdiv != 0
        if np.sum(~u_cond) != 0:
            self.u[1:-1, 1:-1] = np.where(u_cond, u_calc, 0.)
        else:
            self.u[1:-1, 1:-1] = u_calc.copy()
        if np.sum(~v_cond) != 0:
            self.v[1:-1, 1:-1] = np.where(v_cond, v_calc, 0.)
        else:
            self.v[1:-1, 1:-1] = v_calc.copy()

    def _calc_alpha(self, img):
        plot_alpha=False
        # get parameters
        model_mode = self.kwargs.get('alpha_model', 'v1')
        smooth = self.kwargs.get('alpha_smooth', 'fill_ma')
        smooth_n = self.kwargs.get('alpha_smooth_n', 30)
        smooth_window = self.kwargs.get('smooth_window', (5, 5))
        smooth_th = self.kwargs.get('alpha_smooth_th', .02)
        base_num = self.kwargs.get('alpha_base_num', None)
        adjust = self.kwargs.get('alpha_adjust_ratio', None)

        # get r
        r = self.block_dia // 2

        # init alpt with considering skipping
        idx_y = np.arange(r, img.shape[0], self.block_skip)
        idx_x = np.arange(r, img.shape[1], self.block_skip)
        self.alpt = np.zeros((len(idx_y), len(idx_x)), dtype=np.float64)
        self.k = self.alpt.copy()

        # get model
        model = self.alpha_model(model_mode)

        # loop over blocks
        for aj, j in enumerate(idx_y):
            for ai, i in enumerate(idx_x):
                # get sub-block
                subblock = img[j-r: j+r+1, i-r:i+r+1]
                # fft
                f = np.fft.fft2(subblock)
                fshift = np.fft.fftshift(f)
                # spectrum = 20 * np.log(np.abs(fshift))
                spectrum = np.abs(np.real(fshift))
                spectrum[np.isnan(spectrum)] = np.nanmean(spectrum)
                # k
                y, x = np.indices(spectrum.shape)
                k = np.maximum(np.abs(y-r), np.abs(x-r))
                # average k
                spectrum_sum = np.sum(spectrum)
                if spectrum_sum == 0:
                    k_avg = 0
                else:
                    k_avg = int(np.sum(spectrum * k) / spectrum_sum)
                self.k[aj, ai] = k_avg
                # calculate std of sub-block
                std = np.std(subblock)
                # calculate self.alpt
                k_max = np.max(list(model.keys()))
                self.alpt[aj, ai] = model.get(k_avg, model.get(k_max))(std)

        # up-scale alpt and k to the shape as that of img
        self.alpt, self.k = im_resize((img.shape[1], img.shape[0]), 'up', self.alpt, self.k)

        if plot_alpha:
            #print("before")
            i = np.asarray(self.alpt, dtype=np.uint8)
            plt.subplot(111)
            plt.imshow(i, cmap='gray')
            plt.title("Before smoothing")
            plt.tight_layout()
            plt.savefig("before_smooth.png")
            plt.show()

        # smooth self.alpt by moving average
        if smooth is not None:
            self.alpt, = self._smooth(smooth, self.alpt,
                                      smooth_n=smooth_n,
                                      window=smooth_window)
        if plot_alpha:
            print("smooth")
            plt.subplot(111)
            i = np.asarray(self.alpt, dtype=np.uint8)
            plt.imshow(i, cmap='gray')
            plt.title("After smoothing")
            plt.tight_layout()
            plt.savefig("after_smooth.png")
            plt.show()
        # smooth self.alpt by adding a base number
        if base_num is not None:
            self.alpt += base_num
        # adjust self.alpt by multiplying adjust ratio
        if adjust is not None:
            self.alpt *= adjust

    def smooth_vel(self, mode, **kwargs):
        self.u, self.v = self._smooth(mode, self.u, self.v, **kwargs)

    @classmethod
    def _smooth(cls, mode, *args, **kwargs):
        res = []
        for a in args:
            if mode == 'fill_average':
                a = smooth.fill_average(a, kwargs['threshold'])
            elif mode == 'fill_ma':
                a = smooth.fill_moving_average(a, kwargs['window'], kwargs['smooth_n'], kwargs['threshold'])
            elif mode == 'fill_svd':
                a = smooth.fill_svd(a, kwargs['k'], kwargs['threshold'])
            elif mode == 'ma':
                window = kwargs['window']
                smooth_n = kwargs['smooth_n']
                window = tuple(window)
                a = smooth.moving_average(a, window, smooth_n)
            elif mode == 'svd':
                a = smooth.svd(a, kwargs['k'], kwargs['keep_ratio'])
            else:
                raise Exception('Unrecognized smoothing mode')
            res.append(a)
        return res

    @classmethod
    def alpha_model(cls, model):
        if model == 'v1':
            return {
                1: lambda sd: 1.03 * sd - .15,
                2: lambda sd: 2.16 * sd - .2,
                3: lambda sd: 3.15 * sd + .42,
                4: lambda sd: 4.02 * sd + .4,
                5: lambda sd: 4.71 * sd + .09,
                6: lambda sd: 5.19 * sd - .42,
                7: lambda sd: 5.39 * sd - .03
            }
        elif model == 'v2':
            return {
                1: lambda sd: 0.287 * sd - 0.917,
                2: lambda sd: 0.287 * sd - 0.917,
                3: lambda sd: 0.287 * sd - 0.917,
                4: lambda sd: 0.494 * sd - 1.057,
                5: lambda sd: 0.678 * sd + 2.109,
                6: lambda sd: 1.256 * sd - 1.547,
                7: lambda sd: 1.703 * sd - 0.330,
                8: lambda sd: 2.242 * sd - 0.803,
                9: lambda sd: 2.508 * sd - 0.524,
                10: lambda sd: 2.908 * sd - 0.563,
                11: lambda sd: 3.307 * sd - 0.563,
                12: lambda sd: 3.706 * sd - 0.597,
                13: lambda sd: 4.105 * sd - 0.598,
                14: lambda sd: 4.505 * sd - 0.651,
                15: lambda sd: 4.901 * sd - 0.579,
                16: lambda sd: 5.304 * sd - 0.698,
            }

    @classmethod
    def _uv_avg(cls, u, v):
        """
        Important for stable solution
        :param u:
        :param v:
        :return:
        """
        v0 = 0.
        v1 = 1. / 6.
        v2 = 1. / 12.
        kernel = np.array([[v2, v1, v2],
                           [v1, v0, v1],
                           [v2, v1, v2]])
        ua = signal.convolve2d(u, kernel, mode='valid', boundary='symm')
        va = signal.convolve2d(v, kernel, mode='valid', boundary='symm')
        return ua, va


class IntVarHornSchunck(HornSchunck):

    __slots__ = [
        'u', 'v', 'alpt', 'k',
        'levels',
        'dx', 'dy', 'dt',
        'block_dia', 'block_skip',
        'max_iter', 'early_stop',
        'thread', 'robust_mode',
        'verbose',
        'kwargs'
    ]

    def __init__(self, levels=(0, 100, 120, 140, 160, 0, 0, 0, 0),
                 dx=2.5, dy=2.5, dt=1.,
                 block_dia=65, block_skip=33,
                 max_iter=1200, early_stop=1e-5,
                 thread=2, robust_mode=2,
                 verbose=True,
                 **kwargs):
        """

        :param levels:
        :param dx:
        :param dy:
        :param dt:
        :param block_dia:
        :param block_skip:
        :param max_iter:
        :param early_stop:
        :param thread:
        :param robust_mode:
        :param verbose:
        :param kwargs:
        """
        super().__init__(dx, dy, dt,
                         block_dia, block_skip,
                         max_iter, early_stop,
                         thread, robust_mode,
                         verbose,
                         **kwargs)
        # placeholders
        self.u = None
        self.v = None
        self.alpt = None

        # parameters
        self.levels = levels

    def __call__(self, img, imgp, alpt=None):
        # get alpha
        if alpt is None:
            self._calc_alpha(img)
        else:
            self.alpt = alpt

        # initialize self.u and self.v
        self.u = np.zeros_like(img, dtype=np.float64)
        self.v = np.zeros_like(img, dtype=np.float64)

        # main loop
        for _, lvl in enumerate(self.levels):

            # preserve prior u and prior v
            up = self.u.copy()
            vp = self.v.copy()

            # affine images based on intensity
            img_affined, imgp_affined = im_affine(lvl, img, imgp)

            # update self.u and self.v
            self._hs_base(img_affined, imgp_affined, self.alpt)

            # merge prior and updated
            self.u = (up + self.u) * .5
            self.v = (vp + self.v) * .5


class PyramidHornSchunck(HornSchunck):

    __slots__ = [
        'u', 'v', 'alpt', 'k',
        'max_lvl', 'pyr_scale',
        'dx', 'dy', 'dt',
        'block_dia', 'block_skip',
        'max_iter', 'early_stop',
        'thread', 'robust_mode',
        'verbose',
        'kwargs'
    ]

    def __init__(self, max_level=5, pyr_scale=.8,
                 dx=2.5, dy=2.5, dt=1.,
                 block_dia=65, block_skip=33,
                 max_iter=1200, early_stop=1e-5,
                 thread=2, robust_mode=2,
                 verbose=True,
                 **kwargs):
        super().__init__(dx, dy, dt,
                         block_dia, block_skip,
                         max_iter, early_stop,
                         thread, robust_mode,
                         verbose,
                         **kwargs)

        self.max_lvl = max_level
        self.pyr_scale = pyr_scale

    def __call__(self, img, imgp, alpt=None):
        # get the down-sampling scale for each level
        pyr_scales = [self.pyr_scale ** i for i in range(self.max_lvl)][::-1]

        # get the image size corresponding to each scale
        y_size, x_size = img.shape
        y_sizes = [math.ceil(y_size * i) for i in pyr_scales]
        x_sizes = [math.ceil(x_size * i) for i in pyr_scales]

        # alpha
        if alpt is None:
            self._calc_alpha(img)
        else:
            self.alpt = np.zeros_like(img, dtype=np.float64)
            self.alpt.fill(alpt)

        # main loop
        for lvl in range(self.max_lvl):
            # downscale inputs
            img_lvl, imgp_lvl, = im_resize((x_sizes[lvl], y_sizes[lvl]), 'down', img, imgp)

            # get alpha
            if alpt is None:
                self._calc_alpha(img_lvl)
            else:
                alpt_base = np.zeros_like(img, dtype=np.float64)
                alpt_base.fill(alpt)
                self.alpt, = im_resize((x_sizes[lvl], y_sizes[lvl]), 'down', alpt_base)

            # initialize self.u and self.v
            if lvl == 0:
                self.u = np.zeros_like(img_lvl, dtype=np.float64)
                self.v = np.zeros_like(img_lvl, dtype=np.float64)

            # update self.v and self.v
            if self.verbose is True:
                print('\nLevel: %i: image size: (%i, %i)' % (lvl, img_lvl.shape[0], img_lvl.shape[1]))
                print('========================================')
            self._hs_base(img_lvl, imgp_lvl, self.alpt)

            # upscale u and v
            if lvl < self.max_lvl - 1:
                self.u, self.v = im_resize((x_sizes[lvl+1], y_sizes[lvl+1]), 'up', self.u, self.v)
                # # adjust the velocity based on physical implication
                # self.u *= self.pyr_scale
                # self.v *= self.pyr_scale
