from nitorch import io
import torch


root = '.'
root += '/mri'

mri_recon = root + '/mri.mgz'
mri_raw = root + '/raw/mri.raw.mgz'
lta_raw2samseg = root + '/transforms/raw2samseg.lta'

affine_recon = io.map(mri_recon).affine
affine_raw = io.map(mri_raw).affine

raw2samseg = io.transforms.loadf(lta_raw2samseg).squeeze().to(affine_recon)
recon2raw = affine_raw @ affine_recon.inverse()
recon2samseg = raw2samseg @ recon2raw

io.transforms.savef(recon2raw, root + '/transforms/recon2raw.lta', type=1)
io.transforms.savef(recon2samseg, root + '/transforms/recon2samseg.lta', type=1)

