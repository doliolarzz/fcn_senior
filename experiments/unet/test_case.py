import sys
sys.path.insert(0, '../')
import glob
import os
import cv2
import numpy as np
import torch
from utils.units import mm_dbz, dbz_mm, denorm, torch_denorm
from utils.visualizers import make_gif_color, rainfall_shade, make_gif
from utils.evaluators import fp_fn_image_csi, cal_rmse_all, fp_fn_image_csi_muti, torch_cal_rmse_all
from utils.units import dbz_mm, get_crop_boundary_idx
from global_config import global_config
from tqdm import tqdm
from summary.case import case

rs_img = torch.nn.Upsample(size=(global_config['DATA_HEIGHT'], global_config['DATA_WIDTH']), mode='bilinear')

def test(model, weight_path, data_loader, config, save_dir, files, file_name, crop=None):

    h1, h2, w1, w2 = 0, global_config['DATA_HEIGHT'] - 1, 0, global_config['DATA_WIDTH'] - 1
    if crop is not None:
        h1, h2, w1, w2 = get_crop_boundary_idx(crop)

    idx = 0
    try:
        idx = next(i for i,f in enumerate(files) if os.path.basename(f) == file_name)
    except:
        print('not found')
        return
    
    scale = config['SCALE']
    h = int(global_config['DATA_HEIGHT'] * scale)
    w = int(global_config['DATA_WIDTH'] * scale)
    sliced_input = np.zeros((1, config['IN_LEN'], h, w), dtype=np.float32)
    sliced_label = np.zeros((1, global_config['OUT_TARGET_LEN'], global_config['DATA_HEIGHT'], global_config['DATA_WIDTH']), dtype=np.float32)
    
    for i, j in enumerate(range(idx - config['IN_LEN'], idx)):
        f = np.fromfile(files[j], dtype=np.float32) \
            .reshape((global_config['DATA_HEIGHT'], global_config['DATA_WIDTH']))
        sliced_input[0, i] = \
            cv2.resize(f, (w, h), interpolation = cv2.INTER_AREA)
            
    for i, j in enumerate(range(idx, idx+global_config['OUT_TARGET_LEN'])):
        sliced_label[0, i] = np.fromfile(files[j], dtype=np.float32) \
            .reshape((global_config['DATA_HEIGHT'], global_config['DATA_WIDTH']))
            
    sliced_input = (mm_dbz(sliced_input) - global_config['NORM_MIN']) / global_config['NORM_DIV']
    sliced_input = sliced_input.to(config['DEVICE'])

    if config['DIM'] == '3D':
        sliced_input = sliced_input[:, None, :]
    
    outputs = None
    with torch.no_grad():
        for t in range(int(global_config['OUT_TARGET_LEN']//config['OUT_LEN'])):
            # print('input data', data.shape)
            output = model(sliced_input)
            # print('output', output.shape)
            if outputs is None:
                outputs = output
            else:
                outputs = torch.cat([outputs, output], axis=1)

            if config['DIM'] == '3D':
                sliced_input = torch.cat([sliced_input[:, :, 1:], output], dim=2)
            else:
                sliced_input = torch.cat([sliced_input[:, 1:], output], dim=1)
    
    outputs_resized = rs_img(torch_denorm(outputs))
    print(outputs_resized.size(), sliced_label.size())
    rmse, rmse_rain, rmse_non_rain = torch_cal_rmse_all(outputs_resized, sliced_label.to(config['DEVICE']))
    pred = outputs_resized.detach().cpu().numpy()

    print('pred label shape', pred.shape, sliced_label.shape)
    csi = fp_fn_image_csi(pred, sliced_label)
    csi_multi = fp_fn_image_csi_muti(pred, sliced_label)
    
    result_all = [rmse, rmse_rain, rmse_non_rain, csi] + csi_multi

    h_small = outputs.shape[2]
    w_small = outputs.shape[3]
    outputs_small = outputs.detach().cpu().numpy
    label_small = np.zeros((sliced_label.shape[0], sliced_label.shape[1], h_small, w_small))
    for i in range(sliced_label.shape[0]):
        for j in range(sliced_label.shape[1]):
            label_small[i, j] = cv2.resize(sliced_label[i, j], (w_small, h_small), interpolation = cv2.INTER_AREA)

    path = save_dir + '/imgs'
    if not os.path.exists(path):
        os.makedirs(path)
    for i in range(outputs_small.shape[0]):
        #Save pred gif
        make_gif(outputs_small[i] / 80 * 255, path + '/pred_{}_{}.gif'.format(b, i))
        #Save colored pred gif
        make_gif_color(outputs_small[i], path + '/pred_colored_{}_{}.gif'.format(b, i))
        #Save gt gif
        make_gif(label_small[i] / 80 * 255, path + '/gt_{}_{}.gif'.format(b, i))
        #Save colored gt gif
        make_gif_color(label_small[i], path + '/gt_colored_{}_{}.gif'.format(b, i))

    result_all = np.array(result_all)
    result_all_mean = np.mean(result_all, axis=0)
    np.savetxt(save_dir + '/result.txt', result_all_mean, delimiter=',')

if __name__ == "__main__":
    
    model.load_state_dict(torch.load(weight_path, map_location='cuda'))
