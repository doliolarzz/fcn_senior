import torch
import torch.nn as nn
import numpy as np
from models.mrunet.unet_backbone import UNet
from global_config import global_config
import cv2

class MRUNet(nn.Module):
    
    def __init__(self, config, state_weight=0.7, geo_size=4):
        
        super(MRUNet, self).__init__()

        self.h, self.w = config['IN_HEIGHT'], config['IN_WIDTH']
        self.config = config
        self.backbone = UNet(n_channels=config['IN_LEN']*3 - 2) #.to(self.config['DEVICE'])
        self.geo = nn.Parameter(data=torch.randn(1, geo_size, self.h, self.w), requires_grad=True) #.to(self.config['DEVICE'])
        self.state_weight = state_weight
        self.outConv = nn.Conv2d(68, 1, kernel_size=1)

    def get_optFlow(self, input, device):
        input = input.detach().cpu().numpy()
        opt = np.zeros((input.shape[0], 2*self.config['IN_LEN']-2, self.h, self.w), dtype=np.float32)
        for b in range(input.shape[0]):
            for i in range(input.shape[1] - 1):
                delta = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM) \
                    .calc((input[b, i]*255).astype(np.uint8), (input[b, i+1]*255).astype(np.uint8), None)
                opt[b, i*2] = delta[...,0]
                opt[b, i*2+1] = delta[...,1]
        return torch.autograd.Variable(data=torch.from_numpy(opt).float(), requires_grad=False).cuda(device) #.to(self.config['DEVICE'])

    def get_next_state(self, next_input, prev_state, device):

        optFlow = self.get_optFlow(next_input, device)
        return self.state_weight * optFlow + (1 - self.state_weight) * prev_state
        
    def forward(self, input):

        outputs = []
        cur_input = input
        cur_state = self.get_optFlow(cur_input, self.config['DEVICE_ALL'][0])
        assert self.config['OUT_LEN'] == len(self.config['DEVICE_ALL'])
        for i in range(self.config['OUT_LEN']):
            
            dev = self.config['DEVICE_ALL'][i]
            cur_state = cur_state.cuda(dev)
            x = torch.cat([cur_input, cur_state], 1).cuda(dev)
            self.backbone = self.backbone.cuda(dev)
            x = self.backbone(x).cuda(dev)
            x = torch.cat([x, self.geo.expand(x.shape[0], -1, -1, -1)], 1).cuda(dev)
            self.outConv = self.outConv.cuda(dev)
            x = self.outConv(x).cuda(dev)
            outputs.append(x)
            cur_input = torch.cat([cur_input[:, 1:], x], 1)
            cur_state = self.get_next_state(cur_input, cur_state, dev).cuda(dev)

        return torch.stack(outputs, 0).cuda(1)


