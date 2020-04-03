import torch
import torch.nn as nn
import numpy as np
from models.unet.model import UNet2D
from global_config import global_config
import cv2

class RRNet(nn.Module):
    
    def __init__(self, config, hidden_size, training=True, use_optFlow=False, optFlow_weight=0.7, use_geo=False, geo_size=4):
        
        super(RRNet, self).__init__()

        self.h, self.w = config['IN_HEIGHT'], config['IN_WIDTH']
        self.config = config
        self.hidden_size = hidden_size
        self.training = training
        self.use_optFlow = use_optFlow
        self.optFlow_weight = optFlow_weight
        self.use_geo = use_geo
        self.geo = None
        if use_geo:
            self.geo = nn.Parameter(data=torch.randn(1, geo_size, self.h, self.w), requires_grad=True)

        in_channels = 2
        if use_optFlow:
            in_channels+=1
        if use_geo:
            in_channels+=geo_size

        self.backbone = UNet2D(
            in_channels=in_channels, 
            out_channels=hidden_size*4, 
            final_sigmoid=False, 
            layer_order='gcr', 
            is_segmentation=False
        )
        self.Wci = nn.Parameter(torch.zeros(1, hidden_size, self.h, self.w))
        self.Wcf = nn.Parameter(torch.zeros(1, hidden_size, self.h, self.w))
        self.Wco = nn.Parameter(torch.zeros(1, hidden_size, self.h, self.w))

    def get_optFlow(self, prev_input, next_input):
        prev_input = prev_input.detach().cpu().numpy()
        next_input = next_input.detach().cpu().numpy()
        opt = np.zeros((prev_input.shape[0], 1, self.h, self.w), dtype=np.float32)
        for b in range(input.shape[0]):
            delta = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM) \
                .calc((input[b, i]*255).astype(np.uint8), (input[b, i+1]*255).astype(np.uint8), None)
            opt[b, 0] = delta[...,0]
            opt[b, 0] = delta[...,1]
        return torch.autograd.Variable(data=torch.from_numpy(opt).float(), requires_grad=False)

    def get_next_optFlow(self, prev_input, next_input, prev_optFlow):

        optFlow = self.get_optFlow(prev_input, next_input)
        next_optFlow = self.state_weight * optFlow + (1 - self.state_weight) * prev_optFlow
        return next_optFlow

    #input: b,t,1,h,w
    #optFlow: b,1,h,w
    def forward(self, input, optFlow):

        c = torch.zeros((input.shape[0], self.hidden_size, self.h, self.w), dtype=torch.float)
        h = torch.zeros((input.shape[0], self.hidden_size, self.h, self.w), dtype=torch.float)
        
        outputs = []
        h = inputs[:, 0]
        geo_emb = self.geo.expand(input.shape[0], -1, -1, -1)]
        for t in range(self.config['OUT_LEN']):
            if self.training:
                x = inputs[:, t]
            else:
                x = h

            conv_input = [x, h]
            if self.use_optFlow:
                conv_input.append(optFlow)
            if self.use_geo:
                conv_input.append(geo_emb)

            cat_x = torch.cat(conv_input, dim=1)
            conv_x = self.backbone(cat_x)

            i, f, tmp_c, o = torch.chunk(conv_x, 4, dim=1)

            i = torch.sigmoid(i+self.Wci*c)
            f = torch.sigmoid(f+self.Wcf*c)
            c = f*c + i*torch.tanh(tmp_c)
            o = torch.sigmoid(o+self.Wco*c)
            h_next = o*torch.tanh(c)

            optFlow = get_next_optFlow(h, h_next, optFlow)
            h = h_next
            outputs.append(h)

        return torch.stack(outputs, dim=1), (h, c)

