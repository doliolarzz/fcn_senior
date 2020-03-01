import torch
import torch.nn.functional as F
from distutils.version import LooseVersion
from torch.autograd import Variable

# def cross_entropy2d(input, target, weight=None, size_average=True):
#     # input: (n, c, h, w), target: (n, h, w)
#     n, c, h, w = input.size()
#     # log_p: (n, c, h, w)
#     if LooseVersion(torch.__version__) < LooseVersion('0.3'):
#         # ==0.2.X
#         log_p = F.log_softmax(input)
#     else:
#         # >=0.3
#         log_p = F.log_softmax(input, dim=1)
#     # log_p: (n*h*w, c)
#     log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous()
#     log_p = log_p[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
#     log_p = log_p.view(-1, c)
#     # target: (n*h*w,)
#     mask = target >= 0
#     target = target[mask]
#     loss = F.nll_loss(log_p, target, weight=weight, reduction='sum')
#     if size_average:
#         loss /= mask.data.sum()
#     return loss
def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    # number of channels
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.contiguous().view(C, -1)
    
class WeightedCrossEntropyLoss(torch.nn.Module):

    def __init__(self, ignore_index=-1):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.ignore_index = ignore_index

    def forward(self, input, target):
        weight = self._class_weights(input)
        return F.cross_entropy(input, target, weight=weight, ignore_index=self.ignore_index)

    @staticmethod
    def _class_weights(input):
        # normalize the input first
        input = F.softmax(input, dim=1)
        flattened = flatten(input)
        nominator = (1. - flattened).sum(-1)
        denominator = flattened.sum(-1)
        class_weights = Variable(nominator / denominator, requires_grad=False)
        return class_weights