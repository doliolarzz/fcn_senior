import torch
import torch.nn as nn
from models.mrunet.unet_backbone import UNet
from global_config import global_config
import optflow

class MRUNet(nn.Module):
    
    def __init__(self, config, img_size, state_weight=0.7, geo_size=4):
        
        h, w = config['IN_HEIGHT'], config['IN_WIDTH']

        self.config = config
        self.backbone = UNet(n_channels=config['IN_LEN']*2-1)
        self.geo = torch.nn.Parameter(data=torch.randn(geo_size, h, w), requires_grad=True)
        self.state_weight = state_weight
        self.outConv = nn.Conv2d(64, 1, kernel_size=1)

    def get_optFlow(self, input):
        # u, v = optflow.opt_horn(mat,mat2)
        opt = None
        return nn.Variable(data=torch.from_numpy(opt).float(),requires_grad=False)

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


