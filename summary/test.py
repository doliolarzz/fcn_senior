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
from models.unet.unet_model import UNet
from tqdm import tqdm

rs_img = torch.nn.Upsample(size=(global_config['DATA_HEIGHT'], global_config['DATA_WIDTH']), mode='bilinear')

def test(model, data_loader, config, save_dir, crop=None):

    save_dir = save_dir + '/res'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    result_all = []
    n_test = data_loader.n_test_batch()
    # test_idx = np.arange(n_test)
    # np.random.shuffle(test_idx)
    # for b in tqdm(test_idx[:20]):
    for b in tqdm(range(n_test)):
        data, label = data_loader.get_test(b)
        outputs = None
        with torch.no_grad():
            for t in range(int(global_config['OUT_TARGET_LEN']//config['OUT_LEN'])):
#                 print('input data', data.shape)
                output = model(data)
#                 print('output', output.shape)
                if outputs is None:
                    outputs = output.detach().cpu().numpy()
                else:
                    outputs = np.concatenate([outputs, output.detach().cpu().numpy()], axis=1)
                if config['DIM'] == '3D':
                    data = torch.cat([data[:, :, 1:], output[:, :, None]], dim=2)
                else:
                    data = torch.cat([data[:, 1:], output], dim=1)
        pred = np.array(outputs)[:,]
#         print('pred label shape', pred.shape, label.shape)
        pred = denorm(pred)
        pred_resized = np.zeros((pred.shape[0], pred.shape[1], global_config['DATA_HEIGHT'], global_config['DATA_WIDTH']))
        for i in range(pred.shape[0]):
            for j in range(pred.shape[1]):
                pred_resized[i, j] = cv2.resize(pred[i, j], (global_config['DATA_WIDTH'], global_config['DATA_HEIGHT']), interpolation = cv2.INTER_AREA)
        # don't need to denorm test
        csi = fp_fn_image_csi(pred_resized, label)
        csi_multi = fp_fn_image_csi_muti(pred_resized, label)
        # rmse, rmse_rain, rmse_non_rain = cal_rmse_all(pred_resized, label)
        sum_rmse = np.zeros(3)
        for i in range(global_config['OUT_TARGET_LEN']):
            rmse, rmse_rain, rmse_non_rain = torch_cal_rmse_all(pred_resized[:,i].to(), label[:,i].to())
            sum_rmse += np.array(rmse, rmse_rain, rmse_non_rain)
        mean_rmse = sum_rmse / global_config['OUT_TARGET_LEN']
        result_all.append([csi] + list(csi_multi) + list(mean_rmse))
        
        h_small = pred.shape[2]
        w_small = pred.shape[3]
        label_small = np.zeros((label.shape[0], label.shape[1], h_small, w_small))
        for i in range(label.shape[0]):
            for j in range(label.shape[1]):
                label_small[i, j] = cv2.resize(label[i, j], (w_small, h_small), interpolation = cv2.INTER_AREA)
        
        path = save_dir + '/imgs'
        if not os.path.exists(path):
            os.makedirs(path)
        for i in range(pred_resized.shape[0]):
            #Save pred gif
#             make_gif(pred[i] / 80 * 255, path + '/pred_{}_{}.gif'.format(b, i))
            #Save colored pred gif
            make_gif_color(pred[i], path + '/pred_colored_{}_{}.gif'.format(b, i))
            #Save gt gif
#             make_gif(label_small[i] / 80 * 255, path + '/gt_{}_{}.gif'.format(b, i))
            #Save colored gt gif
            make_gif_color(label_small[i], path + '/gt_colored_{}_{}.gif'.format(b, i))

    result_all = np.array(result_all)
    result_all_mean = np.mean(result_all, axis=0)
    np.savetxt(save_dir + '/result.txt', result_all_mean, delimiter=',')
    # np.savez(save_dir + '/result.npz', r = result_all_mean)