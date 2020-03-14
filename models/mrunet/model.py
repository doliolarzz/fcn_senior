import torch
import torch.nn as nn
from models.mrunet.unet_backbone import UNet
from global_config import global_config
import cv2

class MRUNet(nn.Module):
    
    def __init__(self, config, state_weight=0.7, geo_size=4):
        
        self.h, self.w = config['IN_HEIGHT'], config['IN_WIDTH']

        self.config = config
        self.backbone = UNet(n_channels=config['IN_LEN']*3 - 2)
        self.geo = nn.Parameter(data=torch.randn(geo_size, self.h, self.w), requires_grad=True)
        self.state_weight = state_weight
        self.outConv = nn.Conv2d(64, 1, kernel_size=1)

    def get_optFlow(self, input):
        opt = np.zeros((input.shape[0], 2*self.config['IN_LEN']-2, self.h, self.w))
        for b in range(input.shape[0]):
            for i in range(input.shape[1] - 1):
                delta = cv2.optflow.createOptFlow_DIS().calc(input[b, i], input[b, i+1], None)
                opt[b, i*2] = delta[...,0]
                opt[b, i*2+1] = delta[...,1]
        return nn.Variable(data=torch.from_numpy(opt).float(), requires_grad=False)

    def get_next_state(self, next_input, prev_state):

        optFlow = self.get_optFlow(next_input)
        return self.state_weight * optFlow + (1 - self.state_weight) * prev_state
        
    def forward(self, input):

        outputs = []
        cur_input = input
        cur_state = self.get_optFlow(cur_input)

        for i in range(global_config['OUT_TARGET_LEN']):

            x = torch.cat([cur_input, cur_state], 1)
            x = self.backbone(input)
            x = torch.cat([x, self.geo], 1)
            x = self.outConv(x)

            outputs.append(x)
            cur_input = torch.cat([cur_input[:, 1:], x], 1)
            cur_state = self.get_next_state(cur_input, cur_state)

        return torch.stack(outputs, 0)


