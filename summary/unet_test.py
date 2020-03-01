import sys
sys.path.insert(0, '../')
import glob
import os
import cv2
import numpy as np
import torch
from utils.units import mm_dbz, dbz_mm, denorm
from utils.visualizers import make_gif_color, rainfall_shade, make_gif
from utils.evaluators import fp_fn_image_csi, cal_rmse_all, fp_fn_image_csi_muti_reg
from global_config import global_config
from models.unet.unet_model import UNet

def test(model, weight_path, data_loader, config, save_dir, crop=None):

    model.load_state_dict(torch.load(weight_path, map_location='cuda'))

    result_all = []
    n_test = data_loader.n_test_batch()
    for b in range(n_test):
        data, label = data_loader.get_test(b)
        outputs = []
        with torch.no_grad():
            for t in range(int(global_config['OUT_TARGET_LEN']//config['OUT_LEN'])):
                print('input data', data.shape)
                output = model(data)
                print('output', output.shape)
                outputs.append(output.detach().cpu().numpy())
                data = torch.cat([data[:, 1:], output], dim=1)
        pred = np.array(outputs)
        print('pred label shape', pred.shape, label.shape)
        pred = denorm(pred)
        pred = cv2.resize(pred, (global_config['DATA_WIDTH'], global_config['DATA_HEIGHT']), interpolation = cv2.INTER_AREA)
        # don't need to denorm test
        csi = fp_fn_image_csi(pred, label)
        csi_multi = fp_fn_image_csi_muti_reg(pred, label)
        rmse, rmse_rain, rmse_non_rain = cal_rmse_all(pred, label)
        result_all.append([rmse, rmse_rain, rmse_non_rain, csi, csi_multi])

        # path = save_dir + '/imgs'
        # if not os.path.exists(path):
        #     os.makedirs(path)
        # for i in range(pred.shape[0]):
        #     #Save pred gif
        #     make_gif(pred[i] / 80 * 255, path + '/pred_{}_{}.gif')
        #     #Save colored pred gif
        #     make_gif_color(pred[i], path + '/pred_colored_{}_{}.gif')
        #     #Save gt gif
        #     make_gif(label[i] / 80 * 255, path + 'gt_{}_{}.gif')
        #     #Save colored gt gif
        #     make_gif_color(label[i], path + '/gt_colored_{}_{}.gif')

    result_all = np.array(result_all)
    result_all_mean = np.mean(result_all, axis=0)
    np.savez(save_dir + '/result.npz', r = result_all_mean)