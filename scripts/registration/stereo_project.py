import sys
sys.path = ['/autofs/cluster/freesurfer/python/fsmodule'] + sys.path

import nitorch as ni
import freesurfer as fs  # Needs python 3.6 or 3.8
from sklearn.metrics import pairwise_distances_argmin_min
import glob
import torch
import os


root = '.'
odir = root + '/warp2surf'
sdir = root + '/mri/samseg/surf'

marker = 'neun'
niter = 3

sites = f'{odir}/{marker}/sites.nii.gz'
sites_layers = f'{odir}/{marker}/sites_layers.nii.gz'
sites_counts = f'{odir}/{marker}/sites_counts.nii.gz'
sites_volumes = f'{odir}/{marker}/sites_volumes.nii.gz'
surf = list(sorted(glob.glob(f'{sdir}/gwdist*')))
surf = surf[1:-1]  # discard white and pial surfaces

os.makedirs(f'{odir}/{marker}/surf', exist_ok=True)

surfaces = []
for s in surf:
    s1 = fs.Surface.read(s)
    s1 = s1.transform(s1.surf2vox())
    surfaces.append(s1)

coord = ni.io.map(sites)
aff = coord.affine
coord = coord.fdata()[0, 0, 0]
# coord = ni.spatial.affine_matvec(aff, coord)
counts = ni.io.loadf(sites_counts)[0, 0, 0]
volumes = ni.io.loadf(sites_volumes)[0, 0, 0]
density = counts / volumes
layers = ni.io.load(sites_layers)[0, 0, 0]

min_dist = torch.full([len(coord)], float('inf'))
best_vertex = torch.full([len(coord), 2], -1, dtype=torch.long)

for i, s in enumerate(surfaces):
    v, d = pairwise_distances_argmin_min(coord.numpy(), s.vertices)
    v = torch.as_tensor(v)
    d = torch.as_tensor(d)
    mask = d < min_dist
    min_dist[mask] = d[mask]
    best_vertex[mask, 1] = v[mask]
    best_vertex[mask, 0] = i

for i, s in enumerate(surfaces):
    mask = best_vertex[:, 0] == i
    vertices = best_vertex[mask, 1]
    overlay = torch.zeros([len(s.vertices)])
    count = torch.zeros([len(s.vertices)])
    overlay.scatter_add_(0, vertices, density[mask])
    count.scatter_add_(0, vertices, torch.as_tensor(1.).expand([len(vertices)]))
    overlay = fs.Overlay.ensure(overlay.numpy())
    overlay.write(f'{odir}/{marker}/surf/surf{i+1:02d}_density_nosmooth.nii.gz')
    count = fs.Overlay.ensure(count.numpy())
    count.write(f'{odir}/{marker}/surf/surf{i+1:02d}_count_nosmooth.nii.gz')
    overlay = s.smooth_overlay(overlay, niter)
    count = s.smooth_overlay(count, niter)
    overlay.data[count.data < 1e-12] = 0
    overlay.data[count.data >= 1e-12] /= count.data[count.data >= 1e-12]
    overlay.write(f'{odir}/{marker}/surf/surf{i+1:02d}_density_smooth{niter}.nii.gz')
    for j, layer in enumerate(["Layer V", "Layer III"]):
        layer = layer.replace(' ', '')
        submask = mask & (layers == j)
        vertices = best_vertex[submask, 1]
        overlay = torch.zeros([len(s.vertices)])
        count = torch.zeros([len(s.vertices)])
        overlay.scatter_add_(0, vertices, density[submask])
        count.scatter_add_(0, vertices, torch.as_tensor(1.).expand([len(vertices)]))
        overlay = fs.Overlay.ensure(overlay.numpy())
        overlay.write(f'{odir}/{marker}/surf/surf{i+1:02d}_{layer}_density_nosmooth.nii.gz')
        count = fs.Overlay.ensure(count.numpy())
        count.write(f'{odir}/{marker}/surf/surf{i+1:02d}_{layer}_count_nosmooth.nii.gz')
        overlay = s.smooth_overlay(overlay, niter)
        count = s.smooth_overlay(count, niter)
        overlay.data[count.data < 1e-12] = 0
        overlay.data[count.data >= 1e-12] /= count.data[count.data >= 1e-12]
        overlay.write(f'{odir}/{marker}/surf/surf{i+1:02d}_{layer}_density_smooth{niter}.nii.gz')




