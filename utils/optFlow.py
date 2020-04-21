import numpy as np
import cv2

def gaussian_curvature(Z):
    Zy, Zx = np.gradient(Z)
    Zxy, Zxx = np.gradient(Zx)
    Zyy, _ = np.gradient(Zy)
    K = (Zxx * Zyy - (Zxy ** 2)) / (1 + (Zx ** 2) + (Zy ** 2)) ** 2
    return K

def zncorr(roi, temp):
    mu_t = np.mean(temp)
    mu_r = np.mean(roi)
    roi = roi - mu_r
    temp = temp - mu_t

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
            if method=='ncc':
                row.append(corr(patch, target))
            else:
                row.append(zncorr(patch, target))
        res.append(row)
    res = np.array(res)

    ww = int(res.shape[1] / 2)
    hh = int(res.shape[0] / 2)

    corr_max = np.max(res)
    corr_max_index = np.argmax(res)
    idx_y, idx_x = np.unravel_index(corr_max_index, res.shape)
    idx_x = idx_x - ww
    idx_y = idx_y - hh
    magnitude = np.sqrt((idx_y - hh) * (idx_y - hh) + (idx_x - ww) * (idx_x - ww))
    return idx_x, idx_y, res

def gb(i0, i1):
    i0 = cv2.GaussianBlur(i0, (5, 5), 10)
    i1 = cv2.GaussianBlur(i1, (5, 5), 10)

    u_inflow, v_inflow, corr = cross_correlation_global_motion(i0, i1, method='zncc')
    uu = np.ones(i0.shape) * u_inflow
    vv = np.ones(i0.shape) * v_inflow
    z = gaussian_curvature(corr)

    a = np.arange(-5, 5, 0.25)
    xx, yy = np.meshgrid(a, a)
    z2 = cv2.resize(z, xx.shape)
    xx = np.array(xx)
    yy = np.array(yy)
    ww = int(z2.shape[1] / 2)
    hh = int(z2.shape[0] / 2)

    corr_max_index = np.argmax(z2)
    idx_y, idx_x = np.unravel_index(corr_max_index, z2.shape)
    new_u = xx[idx_y, idx_x] - xx[hh, ww]
    new_v = yy[idx_y, idx_x] - yy[hh, ww]

    uu = np.ones(i0.shape) * new_u
    vv = np.ones(i0.shape) * new_v

    return uu, vv

def dual_optFlow(im0, im1):

    u_gb, v_gb = gb(im0, im1)
    delta = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM).calc(im0, im1, None)
    u_lc = delta[..., 0]
    v_lc = delta[..., 1]
    glob_speedup = 1
    loc_speedup = 1
    u = (u_lc * loc_speedup) + (u_gb * glob_speedup)
    v = (v_lc * loc_speedup) + (v_gb * glob_speedup)

    return u, v