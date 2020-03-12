import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('Agg')
from matplotlib.lines import Line2D
import io
from PIL import Image
from global_config import global_config
import copy

def make_video(image_conv, image_label, h, w):
    out = cv2.VideoWriter('compare_1hr.mp4',cv2.VideoWriter_fourcc(*'MP4V'), 10, (w*2+3*3,h+3*2))
    mn = min(image_conv.min(), image_label.min())
    mx = max(image_conv.max(), image_label.max())
    for i in range(image_label.shape[1]):
        left = np.ones((h,3)) * mn
        f = np.concatenate([left,image_conv[-1,i,0],left,image_label[-1,i,0],left], axis=1)
        f = cv2.cvtColor(np.array((f - mn) / (mx - mn)*255,dtype=np.uint8), cv2.COLOR_GRAY2RGB)
        top = np.zeros((3,w*2+3*3,3),dtype=np.uint8)
        f = np.concatenate([top,f,top], axis=0)
        out.write(f)
    out.release()

def gray_shade(im, lvl=100):
    imres = np.zeros_like(im)
    imres[im > 0.2] = 40
    imres[im > 15] = 80
    imres[im > 20] = 120
    imres[im > 30] = 160
    imres[im > 50] = 200
    imres[im > 80] = 255
    return imres


def rainfall_shade(im, mode='RGB'):
    rgb = cv2.cvtColor(im.astype(np.uint8), cv2.COLOR_GRAY2RGB)

    indices = np.where(im <= 0.2)
    rgb[indices[0], indices[1], :] = [0, 0, 0]

    indices = np.where(im > 0.2)
    rgb[indices[0], indices[1], :] = [242, 242, 254]

    indices = np.where(im > 1)
    rgb[indices[0], indices[1], :] = [170, 209, 251]

    indices = np.where(im > 5)
    rgb[indices[0], indices[1], :] = [66, 140, 247]

    indices = np.where(im > 10)
    rgb[indices[0], indices[1], :] = [15, 70, 245]

    indices = np.where(im > 20)
    rgb[indices[0], indices[1], :] = [250, 244, 81]

    indices = np.where(im > 30)
    rgb[indices[0], indices[1], :] = [241, 157, 56]

    indices = np.where(im > 50)
    rgb[indices[0], indices[1], :] = [235, 65, 37]

    indices = np.where(im > 60)
    rgb[indices[0], indices[1], :] = [159, 33, 99]
    if mode=='BGR':
        rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return rgb

labels = ['< 0.2', '0.2 - 1', '1 - 5', '5 - 10', '10 - 20', '20 - 30', '30 - 50', '50 - 60', '> 60']
colors = (np.array([
    [0, 0, 0],
    [242, 242, 254],
    [170, 209, 251],
    [66, 140, 247],
    [15, 70, 245],
    [250, 244, 81],
    [241, 157, 56],
    [235, 65, 37],
    [159, 33, 99]
])/255).tolist()
lines = [Line2D([0], [0], color=colors[i], linewidth=7) for i, l in enumerate(labels)]

def make_img(img_gt, img_pred, label):
    fig, ax = plt.subplots(1, 2, figsize=(8, 4), facecolor='white')
    fig.suptitle(label, fontsize=14)
    ax[0].imshow(img_gt)
    ax[0].set_title('Ground Truth')
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[1].imshow(img_pred)
    ax[1].set_title('Prediction')
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    plt.legend(lines[::-1], labels[::-1], loc='center left', bbox_to_anchor=(1, 0.5))
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    pil_img = copy.deepcopy(Image.open(buf))
    buf.close()
    plt.close()
    return pil_img

def make_gif_color_label(gts, preds, labels, fname='test.gif'):
    img_seq = []
    for i in range(gts.shape[0]):
        c_gt = rainfall_shade(gts[i])
        c_pred = rainfall_shade(preds[i])
        img = make_img(c_gt, c_pred, labels[i])
        img_seq.append(img)
    img = img_seq[0]
    img.save(fname, save_all=True, append_images=img_seq[1:], loop=0, duration=500)

def make_gif(data, fname='test.gif'):
    img_seq = []
    for img in data:
        if len(img.shape) == 2:
            img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        img_seq.append(Image.fromarray(img, 'RGB'))
    img = img_seq[0]
    img.save(fname, save_all=True, append_images=img_seq[1:], loop=0, duration=500)

def make_gif_color(data, fname='test.gif'):
    c_imgs = []
    for i in range(data.shape[0]):
        c_img = rainfall_shade(data[i])
        c_imgs.append(c_img)
    make_gif(c_imgs, fname)