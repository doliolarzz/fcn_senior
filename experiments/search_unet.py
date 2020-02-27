import os
from tqdm import tqdm

device = 'cuda:1'
in_len = [2]
out_len = [1]
batch_size = 4

for il in in_len:
    for ol in out_len:
        cmdStr = 'python3 train_unet.py --name={} --device={} --in={} --out={} --batchsize={}'.format(
            '_'.join([str(il), str(ol)]),
            device,
            il,
            ol,
            batch_size,
        )
        os.system(cmdStr)
