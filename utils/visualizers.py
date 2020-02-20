import cv2
import numpy as np
from PIL import Image

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


def make_gif(data, fname='test.gif'):
    img_seq = [Image.fromarray(img.astype('uint8'), 'RGB') for img in data]
    img = img_seq[0]
    img.save(fname, save_all=True, append_images=img_seq[1:], loop=True)

def make_gif_color(data, fname='test.gif'):
    c_imgs = []
    for i in range(data.shape[0]):
        c_img = rainfall_shade(data[i] / data.max() * 255)
        c_imgs.append(c_img)
    make_gif(c_imgs, fname)