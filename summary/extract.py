import sys
sys.path.insert(0, '../')
import glob
import os
import cv2
import numpy as np
import torch
from utils.units import mm_dbz, dbz_mm, denorm, torch_denorm
from utils.visualizers import make_gif_color, rainfall_shade, make_gif, make_gif_color_label
from utils.evaluators import fp_fn_image_csi, cal_rmse_all, fp_fn_image_csi_muti, torch_cal_rmse_all
from global_config import global_config
from models.unet.model import UNet2D
from tqdm import tqdm

rs_img = torch.nn.Upsample(size=(global_config['DATA_HEIGHT'], global_config['DATA_WIDTH']), mode='bilinear')
thres = (mm_dbz(0.5) - global_config['NORM_MIN']) / global_config['NORM_DIV']

def read_file(file_name, h=None, w=None, resize=False):
    f = np.fromfile(file_name, dtype=np.float32) \
            .reshape((global_config['DATA_HEIGHT'], global_config['DATA_WIDTH']))
    if resize:
        f = cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA)
    return f

def extract(model, config, save_dir, files, file_name, crop=None):

    idx = 0
    try:
        idx = next(i for i, f in enumerate(files)
                   if os.path.basename(f) == file_name)
    except:
        print('not found')
        return

    scale = config['SCALE']
    h = int(global_config['DATA_HEIGHT'] * scale)
    w = int(global_config['DATA_WIDTH'] * scale)
    sliced_input = np.zeros((1, config['IN_LEN'], h, w), dtype=np.float32)
    sliced_label = np.zeros(
        (1, global_config['OUT_TARGET_LEN'], global_config['DATA_HEIGHT'], global_config['DATA_WIDTH']), dtype=np.float32)

    for i, j in enumerate(range(idx - config['IN_LEN'], idx)):
        sliced_input[0, i] = read_file(files[j], h, w, resize=True)

    for i, j in enumerate(range(idx, idx+global_config['OUT_TARGET_LEN'])):
        sliced_label[0, i] = read_file(files[j])

    sliced_input = (mm_dbz(sliced_input) -
                    global_config['NORM_MIN']) / global_config['NORM_DIV']
    sliced_input = torch.from_numpy(sliced_input).to(config['DEVICE'])[:, None]

    save_dir = save_dir + '/extracted'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    model.eval()
    for dt in tqdm(range(6*24)):
        cur_input = sliced_input.clone()
        out_time = int(np.ceil(global_config['OUT_TARGET_LEN']/config['OUT_LEN']))
        outputs = None
        with torch.no_grad():
            for t in range(out_time):
                output = model(cur_input)
                if outputs is None:
                    outputs = output.detach().cpu().numpy()
                else:
                    outputs = np.concatenate(
                        [outputs, output.detach().cpu().numpy()], axis=2)
                if config['OPTFLOW']:
                    cur_input = torch.cat([cur_input[:, :, -1, None], output], axis=2)
                else:
                    cur_input = output
        pred = np.array(outputs)[:,]
        pred = pred[:, 0, :global_config['OUT_TARGET_LEN']]
        pred = denorm(pred)
        pred_resized = np.zeros((pred.shape[0], pred.shape[1], global_config['DATA_HEIGHT'], global_config['DATA_WIDTH']))
        for i in range(pred.shape[0]):
            for j in range(pred.shape[1]):
                pred_resized[i, j] = cv2.resize(pred[i, j], (global_config['DATA_WIDTH'], global_config['DATA_HEIGHT']), interpolation = cv2.INTER_AREA)

        csi = fp_fn_image_csi(pred_resized, sliced_label)
        csi_multi, micro_csi = fp_fn_image_csi_muti(pred_resized, sliced_label)
        sum_rmse = np.zeros((3, ), dtype=np.float32)
        for i in range(global_config['OUT_TARGET_LEN']):
            rmse, rmse_rain, rmse_non_rain = torch_cal_rmse_all(torch.from_numpy(pred_resized[:,i]).to(config['CAL_DEVICE']), torch.from_numpy(sliced_label[:,i]).to(config['CAL_DEVICE']))
            sum_rmse += np.array([rmse.cpu().numpy(), rmse_rain.cpu().numpy(), rmse_non_rain.cpu().numpy()])
        mean_rmse = sum_rmse / global_config['OUT_TARGET_LEN']

        h_small = pred.shape[2]
        w_small = pred.shape[3]
        label_small = np.zeros((sliced_label.shape[0], sliced_label.shape[1], h_small, w_small))
        for i in range(sliced_label.shape[0]):
            for j in range(sliced_label.shape[1]):
                label_small[i, j] = cv2.resize(sliced_label[i, j], (w_small, h_small), interpolation = cv2.INTER_AREA)
        
        time_name = os.path.basename(files[idx+dt])[:-4]
        path = save_dir + '/' + time_name
        if not os.path.exists(path):
            os.makedirs(path)
        if not os.path.exists(path+'/pred'):
            os.makedirs(path+'/pred')
        if not os.path.exists(path+'/label'):
            os.makedirs(path+'/label')
        for i in range(pred_resized.shape[0]):
            for j in range(global_config['OUT_TARGET_LEN']):
                alpha_mask = (label_small[i, j] > 0.2).astype(np.uint8) * 255
                lb = rainfall_shade(label_small[i, j], mode='BGR')
                lb = np.dstack((lb, alpha_mask))

                alpha_mask = (pred[i, j] > 0.2).astype(np.uint8) * 255
                prd = rainfall_shade(pred[i, j], mode='BGR')
                prd = np.dstack((prd, alpha_mask))

                cv2.imwrite(path+'/label/'+str(j)+'.png', lb)
                cv2.imwrite(path+'/pred/'+str(j)+'.png', prd)
                np.savetxt(path+'/metrics.txt', [csi, micro_csi, np.mean(csi_multi)] + list(csi_multi) + list(mean_rmse), delimiter=',', fmt='%.2f')

        sliced_input[:, :-1] = sliced_input[:, 1:]
        next_input = (mm_dbz(read_file(files[idx+dt], h, w, resize=True)) -
                    global_config['NORM_MIN']) / global_config['NORM_DIV']
        sliced_input[: , -1] = torch.from_numpy(next_input).to(config['DEVICE'])

        sliced_label[:, :-1] = sliced_label[:, 1:]
        sliced_label[: , -1] = read_file(files[idx+global_config['OUT_TARGET_LEN']+dt])