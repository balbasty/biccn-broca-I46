import torch
from nitorch import io
from nitorch.spatial import diff1d
from nitorch.spatial import affine_sub

root = '.'
device = 'cuda'

for i in range(9, 24):
    print(i)

    f = io.map('{root}/lsfm/init_oct/neun_638/down_9.9um/neun.{i:02d}.nii.gz')
    dat0 = f.fdata(device=device)

    # we are only looking for the "slide" gradient so we clip large 
    # values to not be sensitive to within tissue gradients.
    dat = dat0.clamp(0, 350)

    # Mask of plane locations that have "zero background"
    # due to 45 degree reslicing. We cannot compute good 
    # gradients there so we discard these locations.
    mask = (dat == 0).any(-1, keepdim=True)

    # compute 1d gradient along the slice direction
    dat = diff1d(dat, dim=2, side='f', bound='dct2').square()
    dat.masked_fill_(mask, float('inf'))

    # compute the minimum gradient in the entire plane
    dat = dat.reshape([-1, dat.shape[2]]).median(dim=0).values
    print(dat.tolist())

    # the planes of maximum "minimum gradient" in each 
    # half of the FOV are the slides limits.
    z0 = dat[:len(dat)//2].argmax() + 1
    z1 = len(dat)//2 + dat[len(dat)//2:].argmax()
    print(z0.item(), z1.item(), len(dat))

    mask = torch.zeros(f.shape, device=device, dtype=bool)
    mask[:, :, z0:z1+1] = True
    io.save(mask, f'{root}/lsfm/slide/neun.{i:02d}.slidemask.nii.gz', dtype='uint8', affine=f.affine)

    dat = dat0[:, :, z0:z1+1]
    new_affine, _ = affine_sub(f.affine, f.shape, (slice(None), slice(None), slice(z0, z1+1)))
    io.savef(dat, f'{root}/lsfm/slide/neun.{i:02d}.slidecrop.nii.gz', like=f, affine=new_affine)

