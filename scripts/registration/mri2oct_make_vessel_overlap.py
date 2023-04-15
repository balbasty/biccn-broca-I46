from glob import glob
from nitorch import io
import torch
import os


root = '.'
affdirs = list(filter(os.path.isdir, sorted(glob(root + '/mri2oct/affine*'))))

foct = io.map(root + '/oct/vessels.manual.nii.gz')
oct = foct.data(dtype=torch.uint8)
oct *= 2

for affdir in affdirs:
    print(affdir)
    fmri = io.map(affdir + '/vessels.manual.crop.reslice.12um.mgz')
    mri = fmri.data(dtype=torch.uint8)
    mri = (mri > 0).to(torch.uint8).add_(oct)
    io.save(mri, affdir + '/vessels.manual.combined.12um.nii.gz', like=foct, dtype='uint8')

