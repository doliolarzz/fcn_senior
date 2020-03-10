import sys
sys.path.insert(0, '../')
import glob
import os
import cv2
import numpy as np
import torch
from utils.units import mm_dbz, dbz_mm, denorm, torch_denorm
from utils.visualizers import make_gif_color, rainfall_shade, make_gif
from utils.evaluators import fp_fn_image_csi, cal_rmse_all, fp_fn_image_csi_muti
from global_config import global_config
from models.unet.unet_model import UNet
from tqdm import tqdm

rs_img = torch.nn.Upsample(size=(global_config['DATA_HEIGHT'], global_config['DATA_WIDTH']), mode='bilinear')

def test(model, weight_path, data_loader, config, save_dir, crop=None):

    save_dir = save_dir + '/res'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model.load_state_dict(torch.load(weight_path, map_location='cuda'))

    result_all = []
    n_test = data_loader.n_test_batch()
    test_idx = np.arange(n_test)
    # np.random.shuffle(test_idx)
    for b in tqdm(test_idx):
        data, label = data_loader.get_test(b)
        outputs = None
        with torch.no_grad():
            for t in range(int(global_config['OUT_TARGET_LEN']//config['OUT_LEN'])):
#                 print('input data', data.shape)
                output = model(data)
#                 print('output', output.shape)
                if outputs is None:
                    outputs = output
                else:
                    outputs = torch.cat([outputs, output], axis=1)
                data = torch.cat([data[:, 1:], output], dim=1)

        rmse, rmse_rain, rmse_non_rain = cal_rmse_all(rs_img(torch_denorm(outputs)), label)
        
        pred = outputs.detach().cpu().numpy()
#         print('pred label shape', pred.shape, label.shape)
        pred = denorm(pred)
        pred_resized = np.zeros((pred.shape[0], pred.shape[1], global_config['DATA_HEIGHT'], global_config['DATA_WIDTH']))
        for i in range(pred.shape[0]):
            for j in range(pred.shape[1]):
                pred_resized[i, j] = cv2.resize(pred[i, j], (global_config['DATA_WIDTH'], global_config['DATA_HEIGHT']), interpolation = cv2.INTER_AREA)
        # don't need to denorm test
        csi = fp_fn_image_csi(pred_resized, label)
        csi_multi = fp_fn_image_csi_muti(pred_resized, label)
        
        result_all.append([rmse, rmse_rain, rmse_non_rain, csi, csi_multi])

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
            make_gif(pred[i] / 80 * 255, path + '/pred_{}_{}.gif'.format(b, i))
            #Save colored pred gif
            make_gif_color(pred[i], path + '/pred_colored_{}_{}.gif'.format(b, i))
            #Save gt gif
            make_gif(label_small[i] / 80 * 255, path + '/gt_{}_{}.gif'.format(b, i))
            #Save colored gt gif
            make_gif_color(label_small[i], path + '/gt_colored_{}_{}.gif'.format(b, i))

    result_all = np.array(result_all)
    result_all_mean = np.mean(result_all, axis=0)
    np.savez(save_dir + '/result.npz', r = result_all_mean)