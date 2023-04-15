import torch
from nitorch import io
from nitorch.spatial import resize, voxel_size
from nitorch.core.py import fileparts
from glob import glob

root = '.'
odir = f'{root}/lsfm/slide_center_oct/'
device = 'cuda'


for i in range(8, 24):

    
    foct = io.map(f'{root}/oct/slices/unstack/oct.{i:02d}.nii.gz')
    centeroct = foct.affine[:3, :3] @ torch.as_tensor(foct.shape).to(foct.affine).sub(1).div(2)[:, None] + foct.affine[:3, -1:]

    fname = f'{root}/lsfm/slide/neun.{i:02d}.slidecrop.nii.gz'
    print(fname)
    f = io.map(fname)
    dat = f.fdata(device=device)
        
    new_shape = list(dat.shape)
    new_shape[2] = 50
        
    dat, aff1 = resize(dat, dim=3, shape=new_shape, anchor='e', affine=f.affine)

    vx1 = voxel_size(aff1)
    vx_new = vx1.clone()
    vx_new[2] = 0.396 / 50
    aff_new = aff1.clone() 
    aff_new[:3, :3] *= (vx_new / vx1)[None, :]
    aff_new[:3, -1:] = centeroct[:3, :] - aff_new[:3, :3] @ torch.as_tensor(dat.shape).to(aff_new).sub(1).div(2)[:, None]
        
    _, base, ext = fileparts(fname)
    base = base.replace('.slidecrop', '')
    base = base.replace('.moved', '')
    io.savef(dat, f'{odir}/{base}{ext}', affine=aff_new, like=f)
    print(' ->', f'{base}{ext}')

    io.transforms.savef(aff_new @ aff1.inverse(), f'{odir}/{base}.to_center.lta', type=1)


