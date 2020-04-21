import torch
import torch.nn as nn
import numpy as np
from global_config import global_config
import cv2
from models.unet.model import UNet3D
from utils.optFlow import dual_optFlow

class UNet3DOptGeo(nn.Module):
    
    def __init__(self, config, use_optFlow=False, use_geo=False, geo_size=4):
        
        super(UNet3DOptGeo, self).__init__()

        assert input.shape[2] - 1 == self.config['OUT_LEN']

        self.h, self.w = config['IN_HEIGHT'], config['IN_WIDTH']
        self.config = config
        self.use_optFlow = use_optFlow
        self.use_geo = use_geo
        self.geo = None
        if use_geo:
            self.geo = nn.Parameter(data=torch.randn(1, geo_size, 1, self.h, self.w), requires_grad=True)

        in_channels = 1
        out_channels = 1
        if use_optFlow:
            in_channels += 2
        if use_geo:
            in_channels += geo_size

        self.backbone = UNet3D(
            in_channels=in_channels, 
            out_channels=out_channels,
            final_sigmoid=False, 
            layer_order='gcr', 
            is_segmentation=False, 
            num_levels=4, 
            pool_kernel_size=(1, 2, 2)
        )

    def get_optFlow(self, input):
        input = input.detach().cpu().numpy()
        opt = np.zeros((input.shape[0], 2, input.shape[2] - 1, self.h, self.w), dtype=np.float32)
        for b in range(input.shape[0]):
            for i in range(input.shape[2] - 1):
                u, v = dual_optFlow((input[b, 0, i] * 255).astype(np.uint8), (input[b, 0, i+1] * 255).astype(np.uint8))
                opt[b, 0, i] = u
                opt[b, 1, i] = v
        #Norm
        opt /= 10
        return torch.autograd.Variable(data=torch.from_numpy(opt).float(), requires_grad=False).cuda()

    #input: b,1,t,h,w
    def forward(self, input, optFlow=None):
        
        if self.use_optFlow or self.use_geo:
            cat_x = [input]
            if self.use_geo:
                geo_emb = self.geo.expand(input.shape[0], -1, input.shape[2], -1, -1)
                cat_x.append(geo_emb)
            if self.use_optFlow:
                optFlow = get_optFlow(input)
                cat_x.append(optFlow)
            input = torch.cat(cat_x, 1)
        
        output = self.backbone(input)

        return output

