import torch
import torch.nn as nn
import numpy as np
from models.mrunet3D.unet3d_backbone import UNet3D
from global_config import global_config
import cv2

class MRUNet(nn.Module):
    
    def __init__(self, config, state_weight=0.7, geo_size=4):
        
        super(MRUNet, self).__init__()

        self.h, self.w = config['IN_HEIGHT'], config['IN_WIDTH']
        self.config = config
        self.backbone = UNet3D(in_channels=3, out_channels=1, final_sigmoid=False, num_levels=4, is_segmentation=False)
        self.geo = nn.Parameter(data=torch.randn(1, geo_size, 1, self.h, self.w), requires_grad=True)
        self.state_weight = state_weight
        out_size = self.backbone.get_out_size()
        self.outConv = nn.Conv3d(out_size, 1, kernel_size=1)

    def get_optFlow(self, input):
        input = input.detach().cpu().numpy()
        opt = np.zeros((input.shape[0], 1, 2*self.config['IN_LEN']-2, self.h, self.w), dtype=np.float32)
        for b in range(input.shape[0]):
            for i in range(input.shape[2] - 1):
                delta = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM) \
                    .calc((input[b, 0, i]*255).astype(np.uint8), (input[b, 0, i+1]*255).astype(np.uint8), None)
                opt[b, 0, i*2] = delta[...,0]
                opt[b, 0, i*2+1] = delta[...,1]
        return torch.autograd.Variable(data=torch.from_numpy(opt).float(), requires_grad=False).cuda()

    def get_next_state(self, next_input, prev_state):

        optFlow = self.get_optFlow(next_input)
        next_state = self.state_weight * optFlow + (1 - self.state_weight) * prev_state
        return next_state
    
    def forward(self, input):

        outputs = []
        cur_input = input
        cur_state = self.get_optFlow(cur_input)
        for i in range(self.config['OUT_LEN']):
            
            cur_input = cur_input[:, :, 1:].cuda()
            cur_state = cur_state.cuda()
            x = torch.cat([cur_input, cur_state, self.geo.expand(cur_input.shape[0], -1, cur_input.shape[2], -1, -1)], 1)

            self.backbone = self.backbone
            x = self.backbone(x)

            self.outConv = self.outConv
            x = self.outConv(x)

            outputs.append(x)
            cur_input = torch.cat([cur_input[:, 1:], x], 1)
            cur_state = self.get_next_state(cur_input, cur_state)

        return torch.cat(outputs, 1)


