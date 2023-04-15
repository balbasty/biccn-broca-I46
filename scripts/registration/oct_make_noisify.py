from nitorch import io
import torch


for i in range(8, 24):

    fname = f'oct.{i:02d}.masked.nii.gz'
    f = io.map(fname)
    d = f.fdata(device='cuda')
    d = d.clamp_min_(60).sub_(60)
    d += torch.rand_like(d)
    io.savef(d, f'oct.{i:02d}.noisify.nii.gz', like=f)


