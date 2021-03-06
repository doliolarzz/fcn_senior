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
        self.backbone = UNet(n_channels=config['IN_LEN']*3 + 2)#.to(self.config['DEVICE'])
        self.geo = nn.Parameter(data=torch.randn(1, geo_size, self.h, self.w), requires_grad=True)#.to(self.config['DEVICE'])
        self.state_weight = state_weight
        self.outConv = nn.Conv2d(64, 1, kernel_size=1)

    def get_optFlow(self, input):
        input = input.detach().cpu().numpy()
        opt = np.zeros((input.shape[0], 2*self.config['IN_LEN']-2, self.h, self.w), dtype=np.float32)
        for b in range(input.shape[0]):
            for i in range(input.shape[1] - 1):
                delta = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM) \
                    .calc((input[b, i]*255).astype(np.uint8), (input[b, i+1]*255).astype(np.uint8), None)
                opt[b, i*2] = delta[...,0]
                opt[b, i*2+1] = delta[...,1]
        return torch.autograd.Variable(data=torch.from_numpy(opt).float(), requires_grad=False).cuda()#.to(self.config['DEVICE'])

    def get_next_state(self, next_input, prev_state):

        optFlow = self.get_optFlow(next_input)
        next_state = self.state_weight * optFlow + (1 - self.state_weight) * prev_state
        return next_state

    
    def forward(self, input):

        outputs = []
        cur_input = input
        cur_state = self.get_optFlow(cur_input)
        for i in range(self.config['OUT_LEN']):

            cur_input = cur_input.cuda()#.to(self.config['DEVICE'])
            cur_state = cur_state.cuda()#.to(self.config['DEVICE'])
            x = torch.cat([cur_input, cur_state, self.geo.expand(cur_input.shape[0], -1, -1, -1)], 1)

            self.backbone = self.backbone
            x = self.backbone(x)

            self.outConv = self.outConv
            x = self.outConv(x)

            outputs.append(x)
            cur_input = torch.cat([cur_input[:, 1:], x], 1)
            cur_state = self.get_next_state(cur_input, cur_state)

        # for i in range(len(outputs)):
        #     print(outputs[i].device, outputs[i].size())

        return torch.cat(outputs, 1)#outputs#torch.cat([o.cuda(dev) for o in outputs], 1)#.cuda(dev)


