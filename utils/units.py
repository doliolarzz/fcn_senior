import numpy as np
from global_config import global_config
import torch 

height, width = global_config['DATA_HEIGHT'], global_config['DATA_WIDTH']
lat_min, lat_max, lon_min, lon_max = global_config['LAT_MIN'], global_config['LAT_MAX'], global_config['LON_MIN'], global_config['LON_MAX']
def get_crop_boundary_idx(crop):
    crop_lat1, crop_lat2, crop_lon1, crop_lon2 = crop
    if (crop_lat1 < lat_min) | (crop_lat2 > lat_max) | (crop_lon1 < lon_min) | (crop_lon2> lon_max) :
        print("Crop boundary is out of bound.")
        return
    
    lat_min_idx = round((crop_lat1 - lat_min)/(lat_max - lat_min)*height) # min row idx, num rows = height
    lat_max_idx = round((crop_lat2 - lat_min)/(lat_max - lat_min)*height) # max row idx, num rows = height
    lon_min_idx = round((crop_lon1 - lon_min)/(lon_max - lon_min)*width)  # min col idx, num cols = height
    lon_max_idx = round((crop_lon2 - lon_min)/(lon_max - lon_min)*width)  # max col idx, num cols = height    
    return height - lat_max_idx, height - lat_min_idx, lon_min_idx, lon_max_idx


c_back = np.power(200, 5/8)
def dbz_mm(value):
    value = np.clip(value, global_config['NORM_MIN'], global_config['NORM_MAX'])
    return np.power(10, value/16) / c_back - 1

c_f = 10*np.log10(200)
def mm_dbz(value):
    value = np.clip(value, global_config['MM_MIN'], global_config['MM_MAX'])
    return c_f + 16*np.log10(value + 1)

def denorm(value):
    return dbz_mm(value * global_config['NORM_DIV'] + global_config['NORM_MIN'])

def torch_dbz_mm(value):
    value = torch.clamp(value, global_config['NORM_MIN'], global_config['NORM_MAX'])
    return torch.pow(10, value/16) / c_back - 1

def torch_denorm(value):
    return torch_dbz_mm(value * global_config['NORM_DIV'] + global_config['NORM_MIN'])