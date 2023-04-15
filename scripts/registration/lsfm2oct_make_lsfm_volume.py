from nitorch import io
from nitorch.spatial import grid_pull, affine_matvec, identity_grid, exp, voxel_size, affine_resize
from nitorch.core.linalg import expm, logm, matvec
import torch
import os

root = '.'

def pullflow(v, g):
    return grid_pull(v.movedim(-1, 0), g).movedim(0, -1)

oct = root + '/oct/oct.nii'
marker = 'calb'
extra = '.float' if marker == 'vessels' else ''

foct = io.map(oct)
oct_vx = voxel_size(foct.affine)
out_vx = torch.as_tensor([0.0099, 0.0099, 0.00792])
print(oct_vx, out_vx)

out_affine, out_shape = affine_resize(foct.affine.float(), foct.shape, factor=oct_vx/out_vx, anchor='e')
print(out_shape, voxel_size(out_affine))
out = torch.zeros(out_shape, dtype=torch.uint8 if marker == 'vessels' else torch.float32)

for i in range(8, 24):   
    print(i)
 
    lsfm = root + f'/lsfm/slide_center_oct/{marker}.{i:02d}{extra}.nii.gz'
    if not os.path.exists(lsfm):
        continue

    affine = root + f'/oct2lsfm/nonlin/{i:02d}/affine.lta'
    svf = root + f'/oct2lsfm/nonlin/{i:02d}/svf.nii.gz'
    
    lsfm = io.map(lsfm)
    lsfm_affine = lsfm.affine.float()
    affine = io.transforms.loadf(affine).squeeze().float()
    affine = expm(logm(affine) * 0.5).float()
    svf = io.map(svf)
    svf_affine = svf.affine.float()

    grid = identity_grid(out_shape)
    grid = affine_matvec(svf_affine.inverse() @ affine.inverse() @ out_affine, grid)
    grid += pullflow(exp(svf.fdata().neg_(), displacement=True), grid)
    grid = affine_matvec(lsfm_affine.inverse() @ affine.inverse() @ svf_affine, grid)

    if marker == 'vessels':
        out += grid_pull(lsfm.fdata().squeeze(), grid, bound='dct2', extrapolate=2).round_().to(out)
    else:
        out += grid_pull(lsfm.fdata().squeeze(), grid, bound='dct2', extrapolate=2)

if marker == 'vessels':
    io.save(out.clamp_max_(1), root + f'/warp2oct/{marker}.nii.gz', affine=out_affine, dtype='uint8')
else:
    io.savef(out, root + f'/warp2oct/{marker}.nii.gz', affine=out_affine)

