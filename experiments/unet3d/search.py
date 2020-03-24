import os
from tqdm import tqdm

device = 'cuda:0'
in_len = [6]
batch_size = 3

for il in in_len:
    cmdStr = 'python3 train.py --name={} --device={} --in={} --out={} --batchsize={}'.format(
        '_'.join([str(il), str(il)]),
        device,
        il,
        il,
        batch_size,
    )
    os.system(cmdStr)
