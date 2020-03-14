import os
from tqdm import tqdm

device = 'cuda:0'
in_len = [4]
out_len = [4]
batch_size = 2

for il in in_len:
    for ol in out_len:
        cmdStr = 'python3 train.py --name={} --device={} --in={} --out={} --batchsize={}'.format(
            '_'.join([str(il), str(ol)]),
            device,
            il,
            ol,
            batch_size,
        )
        os.system(cmdStr)
