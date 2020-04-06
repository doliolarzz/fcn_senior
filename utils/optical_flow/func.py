"""
    File name: func.py
    Author: Guodong DU, R-corner, WNI
    Date created: 2018-12-19
    Python Version: 3.6
"""


import numpy as np
from numba import jit
import cv2
import matplotlib.pyplot as plt

def im_resize(dsize, direction, *args):
    res = []
    for img in args:
        if direction == 'down':
            img_d = cv2.resize(img, dsize, interpolation=cv2.INTER_AREA)
        elif direction == 'up':
            img_d = cv2.resize(img, dsize, interpolation=cv2.INTER_CUBIC)
        else:
            raise Exception('Unrecognized sampling direction')
        res.append(img_d)
    return res


def im_affine(threshold, *args):
    res = []
    for img in args:
        img_tmp = img.copy()
        img_tmp[img_tmp < threshold] = 0.
        res.append(img_tmp)
    return res


def calc_differential(img, imgp, dx, dy, dt, robust_mode, **kwargs):
    """ simple difference method, NEED TO UPDATE ???
    :param img: np.ndarray, 初期(レーダ)画像
    :param imgp: np.ndarray, 初期画像の一つ前の画像
    :param dx: np.ndarray, 偏微分におけるdxの値
    :param dy: np.ndarray, 偏微分におけるdyの値
    :param dt: np.ndarray, 偏微分におけるdtの値
    :param robust_mode: int, mode of robust estimation; 0 if not using robust estimation
    :return:
    """
    wx = np.zeros_like(img, dtype=np.float64)
    wy = np.zeros_like(img, dtype=np.float64)
    wx[1:-1, 1:-1] = img[1:-1, 2:] - img[1:-1, :-2]
    wy[1:-1, 1:-1] = img[2:, 1:-1] - img[:-2, 1:-1]
    rx = wx / (dx * 2)
    ry = wy / (dy * 2)

    wt = img - imgp
    rt = wt / dt

    if robust_mode != 0:
        rx, ry, rt = robust(robust_mode, rx, ry, rt,
                            wx, wy, wt,
                            **kwargs)

    return rx, ry, rt


def boundary(img):
    """2次元配列のテーブルで最外周の領域に内周の値を与える
    :param img: input image
    :return:
    """
    # left and right
    img[:, 0] = img[:, 1]
    img[:, -1] = img[:, -2]
    # top and bottom
    img[0, :] = img[1, :]
    img[-1, :] = img[-2, :]
    # corners
    img[0, 0] = img[1, 1]
    img[0, -1] = img[1, -2]
    img[-1, 0] = img[-2, 1]
    img[-1, -1] = img[-2, -2]
    return img


def robust(mode, rx, ry, rt, wx, wy, wt, **kwargs):
    """
    :param mode: int, 1, 2, or 3
    :param rx: np.ndarray, 微分値
    :param ry: np.ndarray, 微分値
    :param rt: np.ndarray, 微分値
    :param wx: np.ndarray, 画像X方向差分値(現在 img(x+1)-img(x-1) )
    :param wy: np.ndarray, 画像Y方向差分値(現在 img(x+1)-img(x-1) )
    :param wt: np.ndarray, 画像時間差分値(現在−過去)
    :param kwargs: dict, optional parameters for implementing each mode
    :return:
    """
    if mode == 1:
        # condition 1
        cond = wt[1:-1, 1:-1] <= 20. * 9.
        wt[1:-1, 1:-1][cond] = (1. - wt[1:-1, 1:-1][cond] / 20. / 9. ** 2) ** 2
        wt[1:-1, 1:-1][~cond] = 0.

        # condition 2
        cond1 = wt[1:-1, 1:-1] > 1.
        cond2 = wt[1:-1, 1:-1] < -1.
        wt[1:-1, 1:-1][cond1] = 1.
        wt[1:-1, 1:-1][cond2] = -1.

        # final
        rx[1:-1, 1:-1] *= wt[1:-1, 1:-1]
        ry[1:-1, 1:-1] *= wt[1:-1, 1:-1]
        rt[1:-1, 1:-1] *= wt[1:-1, 1:-1]

    elif mode == 2:
        threshold = kwargs.get('robust_threshold', 10.)
        cond = wt[1:-1, 1:-1] > threshold
        rx[1:-1, 1:-1][cond] = 0.
        ry[1:-1, 1:-1][cond] = 0.
        rt[1:-1, 1:-1][cond] = 0.

    elif mode == 3:
        # get optional parameters for mode 3
        threshold = kwargs.get('robust_threshold', 10.)
        xy = kwargs.get('process_xy', True)
        x_coef = kwargs.get('cx', 10.)
        y_coef = kwargs.get('cy', 10.)
        t_coef = kwargs.get('ct', 2.)

        # rx and ry
        if xy is True:
            # x
            wx_tmp = wx[:, 1:-1]
            wx_tmp = np.abs(wx_tmp[wx_tmp != 0.])
            x_med = np.median(wx_tmp)
            cond = wx[1:-1, 1:-1] > x_med * x_coef
            rx[1:-1, 1:-1][cond] *= np.cos((np.pi / 2) * (np.abs(wx[1:-1, 1:-1][cond]) / 255.))
            # y
            wy_tmp = wy[1:-1, :]
            wy_tmp = np.abs(wy_tmp[wy_tmp != 0.])
            y_med = np.median(wy_tmp)
            cond = wy[1:-1, 1:-1] > y_med * y_coef
            ry[1:-1, 1:-1][cond] *= np.cos((np.pi / 2) * (np.abs(wy[1:-1, 1:-1][cond]) / 255.))

        # rt
        wt_tmp = np.abs(wt[wt != 0.])
        t_med = np.median(wt_tmp)
        cond1 = wt[1:-1, 1:-1] >= threshold
        cond2 = np.abs(wt[1:-1, 1:-1]) > t_med * t_coef
        cond2 = (~cond1) & cond2
        rt[1:-1, 1:-1][cond1] = 0.
        rt[1:-1, 1:-1][cond2] *= np.cos((np.pi / 2) * (np.abs(wt[1:-1, 1:-1][cond2]) / 255.))

    return rx, ry, rt


def vel_affine(uh, vh, xoff, yoff):
    """ ???
    :param uh:
    :param vh:
    :param xoff:
    :param yoff:
    :return:
    """
    dm = 6
    dy = 2 * yoff + 1
    dx = 2 * xoff + 1

    an = np.zeros(dm, dtype=np.float64)
    w = np.zeros(dm, dtype=np.float64)
    bg = np.zeros(dm, dtype=np.float64)

    cx = np.ones((dy, dx))

    cy = np.ones((dy, dx))
    cx = np.cumsum(cx, axis=1)
    cy = np.cumsum(cy, axis=0)

    xoff2param_iter = {
        10: 1e-3,
        20: 1e-4,
        30: 1e-5,
        40: 1e-6,
        50: 1e-7,
        60: 1e-8
    }  # 10x10:0.001	30x30:0.0001  40x40:0.00001
    param_iter = xoff2param_iter[xoff * 2]

    for i in range(100):
        eng = 0
        w = np.zeros(dm, dtype=np.float64)

        if i < 10:
            ua = an[0]
            va = an[3]
            an[1, 2, 4, 5] = 0.
        else:
            ua = an[0] + an[1] * cx + an[2] * cx
            va = an[3] + an[4] * cy + an[5] * cy
        bufx = uh - ua
        bufy = vh - va

        w[0] = np.sum(bufx)
        w[1] = np.sum(bufx * cx)
        w[2] = np.sum(bufx * cy)
        w[3] = np.sum(bufy)
        w[4] = np.sum(bufy * cx)
        w[5] = np.sum(bufy * cy)

        eng = np.sum(bufx ** 2 + bufy ** 2)

        cond = w != 0
        an[cond] += param_iter * w[cond] / (dy * dx)  # ??? QUESTION, NEED TO CONFIRM

    u_affine = np.sum(an[:3] * [1, dx/2., dy/2.])
    v_affine = np.sum(an[3:] * [1, dx/2., dy/2.])

    return u_affine, v_affine


@jit
def corr(i0, i1):
    return ((i0 - i0.mean())*(i1 - i1.mean())).mean()/(i0.std()*i1.std()+1e-3)

# From https://algorithm.joho.info/programming/python/opencv-template-matching-zncc-py/
# Author : 西住工房
# Rename from template_matching_zncc(src, temp)
# Input: Source and template images
# Out: coordinate
def zncorr(roi, temp):
    #print('zncc')
    mu_t = np.mean(temp)
    mu_r = np.mean(roi)
    # 窓画像 - 窓画像の平均
    roi = roi - mu_r
    # テンプレート画像 - 窓画像の平均
    temp = temp - mu_t

    # ZNCCの計算式
    num = np.sum(roi * temp)
    den = np.sqrt(np.sum(roi ** 2)) * np.sqrt(np.sum(temp ** 2))

    if den == 0:
        return 0
    else:
        return num/den

def cross_correlation_global_motion(im0, im1, step=10, shift=1, method='zncc'):
    h, w = im1.shape
    win_width = win_height = min(h - step, w - step)
    x = np.arange(0, h, shift)
    y = np.arange(0, w, shift)

    w0 = int(w / 2)
    h0 = int(h / 2)

    #target = im0[h0 - int(win_height / 2):h0 - int(win_height / 2) + win_width,
    #         w0 - int(win_width / 2): w0 - int(win_width / 2) + win_width]
    off = int(step/2)
    target = im0[off: h-off, off: w-off]
    win_height = h - step
    win_width = w - step
    res = []
    for r in x:
        row = []
        if r + win_height > h:
            break
        for c in y:
            if c + win_width > w:
                break
            patch = im1[r: r + win_height, c: c + win_width]
            #print(patch.shape, target.shape)
            # print(patch.shape, target.shape)
            # print(r, c)
            if method=='ncc':
                #print("NCC")
                row.append(corr(patch, target))
            else:
                # zncc
                #print("ZNCC")
                row.append(zncorr(patch, target))
        res.append(row)
    res = np.array(res)

    ww = int(res.shape[1] / 2)
    hh = int(res.shape[0] / 2)

    corr_max = np.max(res)
    corr_max_index = np.argmax(res)
    print(corr_max_index)
    idx_y, idx_x = np.unravel_index(corr_max_index, res.shape)
    print("idx_x", idx_x, ww)
    idx_x = idx_x - ww
    print("idx_x", idx_x, ww)
    print("idx_y", idx_y, hh)
    idx_y = idx_y - hh
    print("idx_y", idx_y, hh)
    magnitude = np.sqrt((idx_y - hh) * (idx_y - hh) + (idx_x - ww) * (idx_x - ww))
    return idx_x, idx_y, res

    #if method=='zncc':
    #    return zncc(im1, target)
    #elif method=='ncc':
    #    return ncc(im1, target, win_height, win_width, x, y, h, w)
