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

def test(model, data_loader, config, save_dir, crop=None):

    save_dir = save_dir + '/res'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    model.eval()
    result_all = []
    n_test = data_loader.n_test_batch()
    test_idx = np.arange(0, n_test - 3 * global_config['OUT_TARGET_LEN'], 3*global_config['OUT_TARGET_LEN'])
    # np.random.seed(42)
    # np.random.shuffle(test_idx)
    for b in tqdm(test_idx):
    # for b in tqdm(range(n_test)):
        data, label = data_loader.get_test(b)
        outputs = None
        if config['DIM'] == 'RR':
            out_time = int(np.ceil(global_config['OUT_TARGET_LEN']/(config['IN_LEN'] - 1)))
        else:
            out_time = int(np.ceil(global_config['OUT_TARGET_LEN']/config['OUT_LEN']))
        with torch.no_grad():
            for t in range(out_time):
#                 print('input data', data.shape)
                if config['DIM'] == 'RR':
                    output = model(data[:, -1, None])
                else:
                    output = model(data)
#                 print('output', output.shape)
                if outputs is None:
                    outputs = output.detach().cpu().numpy()
                else:
                    outputs = np.concatenate([outputs, output.detach().cpu().numpy()], axis=1)
                if config['DIM'] == '3D':
                    data = output #torch.cat([data[:, :, 1:], output[:, :, None]], dim=2)
                if config['DIM'] == 'RR':
                    data = output
                else:
                    data = torch.cat([data[:, 1:], output], dim=1)
        pred = np.array(outputs)[:,]
        if config['DIM'] == '3D':
            pred = pred[:, 0]
            pred = pred[:, :global_config['OUT_TARGET_LEN']]
        elif config['DIM'] == '2D':
            pred = pred[:, :, 0]
            pred = pred[:, :global_config['OUT_TARGET_LEN']]
            label = label[:, :, 0]
        elif config['DIM'] == 'RR':
            pred = pred[:, :global_config['OUT_TARGET_LEN']]
        # print('pred label shape', pred.shape, label.shape)
        pred = denorm(pred)
        pred_resized = np.zeros((pred.shape[0], pred.shape[1], global_config['DATA_HEIGHT'], global_config['DATA_WIDTH']))
        for i in range(pred.shape[0]):
            for j in range(pred.shape[1]):
                pred_resized[i, j] = cv2.resize(pred[i, j], (global_config['DATA_WIDTH'], global_config['DATA_HEIGHT']), interpolation = cv2.INTER_AREA)
        # don't need to denorm test
        csi = fp_fn_image_csi(pred_resized, label)
        csi_multi, macro_csi = fp_fn_image_csi_muti(pred_resized, label)
        # rmse, rmse_rain, rmse_non_rain = cal_rmse_all(pred_resized, label)
        sum_rmse = np.zeros((3, ), dtype=np.float32)
        for i in range(global_config['OUT_TARGET_LEN']):
            rmse, rmse_rain, rmse_non_rain = torch_cal_rmse_all(torch.from_numpy(pred_resized[:,i]).to(config['CAL_DEVICE']), torch.from_numpy(label[:,i]).to(config['CAL_DEVICE']))
            sum_rmse += np.array([rmse.cpu().numpy(), rmse_rain.cpu().numpy(), rmse_non_rain.cpu().numpy()])
        mean_rmse = sum_rmse / global_config['OUT_TARGET_LEN']
        result_all.append([csi, macro_csi] + list(csi_multi) + list(mean_rmse))

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
            # Save pred gif
            #             make_gif(pred[i] / 80 * 255, path + '/pred_{}_{}.gif'.format(b, i))
            # Save colored pred gif
            # make_gif_color(pred[i], path + '/pred_colored.gif')
            # Save gt gif
    #             make_gif(label_small[i] / 80 * 255, path + '/gt_{}_{}.gif'.format(b, i))
            # Save colored gt gif
            # make_gif_color(label_small[i], path + '/gt_colored.gif')

            labels = ['' for i in range(global_config['OUT_TARGET_LEN'])]
            make_gif_color_label(label_small[i], pred[i], labels, fname=path + '/{}.gif'.format(b))

    result_all = np.array(result_all)
    result_all_mean = np.mean(result_all, axis=0)
    result_all_mean = np.around(result_all_mean, decimals=3)
    np.savetxt(save_dir + '/result.txt', result_all_mean, delimiter=',', fmt='%.3f')
    # np.savez(save_dir + '/result.npz', r = result_all_mean)