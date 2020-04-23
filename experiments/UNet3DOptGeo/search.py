import os
from tqdm import tqdm

device = 'cuda:0'
in_len = [4]
batch_size = 2

for il in in_len:
    cmdStr = 'python3 train.py --name={} --device={} --in={} --out={} --batchsize={}'.format(
        '_'.join([str(il), str(il)]),
        device,
        il+1,
        il,
        batch_size,
    )
    os.system(cmdStr)
