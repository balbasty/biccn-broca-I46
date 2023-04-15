from nitorch import io
import torch


root = '.'
marker = 'calb'

root += '/warp2oct'
neun = root + f'/{marker}.nii.gz'
infrasupra = root + '/infrasupra.nii.gz'

neun = io.map(neun)
infrasupra = io.map(infrasupra)

dat = neun.fdata().squeeze()
mask = infrasupra.data().squeeze() >= 2

norm = (dat*mask).sum([0, 1]).clamp_min_(1e-3).reciprocal_()
norm *= mask.sum([0, 1])
dat *= norm

io.savef(dat, root + f'/{marker}.norm.nii.gz', like=neun)


