from nitorch import io
import torch
import os

root = '.'
root += '/lsfm/slide_center_oct'
device='cuda'
factor = 1

for i in range(8, 24):
    fname = root + f'/vessels.{i:02d}.distance.nii.gz'
    if not os.path.exists(fname):
        continue
    print(i)
    f = io.map(fname)
    d = f.fdata(device=device)
    d = d.mul_(factor).neg_().exp_().neg_().add_(1)
    io.savef(d, root + f'/vessels.{i:02d}.distance.squeeze.nii.gz', like=f, dtype='float32')
