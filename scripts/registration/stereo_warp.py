import os
import nitorch as ni
from nitorch.mesh import mbf
import glob
import torch
import json
import os

root = '.'
stain = 'calb'
odir = root + f'/warp2surf/{stain}'
os.makedirs(odir, exist_ok=True)

mri = root + '/mri/mri.mgz'
mri_samseg_600 = root + '/mri/samseg/mri.600.mgz'
lsfm_stain = root + f'/lsfm/init_oct/{stain}_*/raw_3.3um/{stain}.*.nii.gz'
lsfm_stain = list(sorted(glob.glob(lsfm_stain)))

stereo = root + f'/lsfm/stereo/{stain}/{stain}.*.ster.xml'
stereo = list(sorted(glob.glob(stereo)))

mri2samseg = root + '/mri/transforms/recon2samseg.lta'
stain2neun = list(sorted(glob.glob(root + f'/lsfm/align_channels/{stain}2neun/*/affine.lta'))) if stain != 'neun' else []
neun2lsfm = list(sorted(glob.glob(root + f'/lsfm/slide_center_oct/neun.*.to_center.lta')))
oct2lsfm = list(sorted(glob.glob(root + '/oct2lsfm/nonlin/*/affine.lta')))
oct2lsfm_diffeo = list(sorted(glob.glob(root + '/oct2lsfm/nonlin/*/svf.nii.gz')))
mri2oct = root + '/mri2oct/affine/affine.lta'


# For each spim slice, we want to build the transformation field that maps
# into MRI (flash20_600_orig) coordinates. We can then interpolate this field
# at the location of the stereo coordinates; the result will be stero
# coordinates in the space of the MRI.
# However, we do not want to generate this field at a super high resolution,
# so we'll generate a reasonable intermediate space.

# the resulting transformation is:
# inv(A_600) * A_orig * inv(A_ref) * inv(S_mri) * A_diffeo_mri * exp(diffeo_mri)
# o inv(A_diffeo_mri) * inv(S_spim) * A_spim * exp(diffeo_spim) o inv(A_spim)
# * A_inter * x_inter o inv(A_inter) * A_spim * stereo
#
# `*` indicates matrix multiplication
# `o` indicates trilinear interpolation
#
# We will precompute
# `inv(A_600) * A_orig * inv(A_ref) * inv(S_mri) * A_diffeo_mri * exp(diffeo_mri)`
# since it does not depend on the SPIM space.

def get_orientation(fname):
    f = ni.io.map(fname)
    return f.affine, f.shape

def pull_displacement(v, g):
    return ni.spatial.grid_pull(v.movedim(-1, 0), g).movedim(0, -1)

def transform_markers(m, g):
    m = ni.spatial.grid_pull(g.movedim(-1, 0), m[None, None])
    m = m.movedim(0, -1)[0, 0]
    return m

def read_stereo(fname):
    sections, contours, markers = mbf.read_contours_xml(fname)

    layers = [k for k in contours.keys() if k not in ('LeftBottom', 'TopRight')]

    rois = {}
    for k in layers:
        for s in sections.keys():
            for p in contours[k][s]:
                rois.setdefault(s, {})
                rois[s].setdefault(k, [])
                rois[s][k].append(torch.as_tensor(p['coord']))

    sites = []
    sites_layers = []
    sites_volumes = []
    for k in contours['LeftBottom'].keys():
        for l, r in zip(contours['LeftBottom'][k], contours['TopRight'][k]):
            l = torch.as_tensor(l['coord'])
            r = torch.as_tensor(r['coord'])
            s = (r.sum(0) + l[1]) / 4
            s[-1] = sections[k]['top'] - sections[k]['thickness'] / 2
            sites.append(s)
            sites_volumes.append((r[0, 0] - r[1, 0]).abs() *
                                 (r[1, 1] - r[2, 1]).abs() *
                                 sections[k]['thickness'] * (1e-9))  # (mm^3)
            layer_found = False
            for layer in rois[k]:
                for p in rois[k][layer]:
                    if mbf.is_inside(s[:2], p[:, :2]):
                        sites_layers.append(layers.index(layer))
                        layer_found = True
                        break
            if not layer_found:
                sites_layers.append(-1)
    sites = torch.stack(sites)
    sites_layers = torch.as_tensor(sites_layers)
    sites_volumes = torch.stack(sites_volumes)

    cells = []
    cells_sites = []
    sites_counts = torch.zeros(len(sites), dtype=torch.long)
    for m in markers:
        p = torch.as_tensor(m['point'])
        s = (sites - p).square().sum(-1).min(0).indices
        cells.append(p)
        cells_sites.append(s)
        sites_counts[s] += 1
    cells = torch.stack(cells)
    cells_sites = torch.stack(cells_sites)

    return layers, sites, sites_layers, sites_counts, sites_volumes, cells, cells_sites

# read all affines
A_600, shape_600 = get_orientation(mri_samseg_600)
A_600 = A_600.float()
S_mri2samseg = ni.io.transforms.loadf(mri2samseg).squeeze().float()
S_mri2oct = ni.io.transforms.loadf(mri2oct).squeeze().float()

# pre-compute MRI component
A2oct = A_600.inverse() @ S_mri2samseg @ S_mri2oct

#A_600, shape_600 = get_orientation(mri)
#A_600 = A_600.float()
#A2oct = A_600.inverse() @ S_mri2oct

# loop over sections
vx_inter = 10e-3  # 10um intermediate space

all_layers = []
all_types = []
all_sites = []
all_sites_volumes = []
all_sites_counts = []
all_sites_layers = []
all_markers = []
all_markers_types = []
all_markers_sites = []
for i in range(len(lsfm_stain)):

    print('lsfm:', lsfm_stain[i])
    print('stereo:', stereo[i])
    print('affine:', oct2lsfm[i])
    print('diffeo:', oct2lsfm_diffeo[i])

    A_lsfm3, shape_lsfm3 = get_orientation(lsfm_stain[i])
    A_lsfm3 = A_lsfm3.float()

    S_oct2lsfm = ni.io.transforms.loadf(oct2lsfm[i]).squeeze().float()    
    S_oct2lsfm = ni.core.linalg.expm(ni.core.linalg.logm(S_oct2lsfm) * 0.5).float()
    
    S_stain2lsfm = ni.io.transforms.loadf(neun2lsfm[i]).squeeze().float()
    if stain2neun:
        S_stain2neun = ni.io.transforms.loadf(stain2neun[i]).squeeze().float()
        S_stain2lsfm = S_stain2neun @ S_stain2lsfm

    A_lsfm9, shape_lsfm9 = ni.spatial.affine_resize(A_lsfm3, shape_lsfm3, 1/3)
    A_lsfm90 = A_lsfm9.float()
    A_lsfm9 = S_stain2lsfm @ A_lsfm90
    
    A_oct2lsfm_diffeo, shape_spim_diffeo = get_orientation(oct2lsfm_diffeo[i])
    A_oct2lsfm_diffeo = A_oct2lsfm_diffeo.float()

    # spim transformation
    y = ni.io.loadf(oct2lsfm_diffeo[i])
    y = ni.spatial.exp(y, displacement=True)

    # sampling grid
    z = ni.spatial.identity_grid(shape_lsfm9)
    z = ni.spatial.affine_matvec(A_oct2lsfm_diffeo.inverse() @ S_oct2lsfm @ A_lsfm9, z)

    # compose
    z += pull_displacement(y, z)
    z = ni.spatial.affine_matvec(A2oct @ S_oct2lsfm @ A_oct2lsfm_diffeo, z)
    J = ni.spatial.grid_jacobian(ni.spatial.affine_matvec(A_600, z),
                                 type='disp', add_identity=False, bound='dct2')
    J = ni.core.linalg.matvec(A_lsfm90[:3, :3].inverse().T, J)
    J = J.det()
    # for some reason all my determinants are negative, which cannot be right FIXME
    J = J.neg_().clamp_min_(0)

    # transform markers
    layers, sites, sites_layers, sites_counts, sites_volumes, \
    markers, markers_sites = read_stereo(stereo[i])
    sites[:, 1:].neg_()
    markers[:, 1:].neg_()
    # layers:           (K,) list[str]                 | layer (ROI) names
    # sites:            (N, 3) tensor[float]           | site center-of-mass
    # sites_layers:     (N,) tensor[int] \in range(K)  | layer of each site
    # sites_counts:     (N,) tensor[int]               | Number of cells in each site
    # sites_volumes:    (N,) tensor[float]             | Volume of each site
    # markers:          (M, 3) tensor[float]           | cell coordinates
    # markers_sites:    (M,) tensor[int] \in range(N)  | site of each cell

    scl = torch.eye(4)
    scl.diagonal(0, -1, -2)[:3] /= 3.3
    A_stereo = A_lsfm90.inverse() @ A_lsfm3 @ scl
    sites = ni.spatial.affine_matvec(A_stereo, sites)
    markers = ni.spatial.affine_matvec(A_stereo, markers)

    J = ni.spatial.grid_pull(J, sites[None, None])[0, 0]
    sites_volumes *= J
    sites = transform_markers(sites, z)
    markers = transform_markers(markers, z)

    all_layers += [layers]
    all_sites += [sites]
    all_sites_volumes += [sites_volumes]
    all_sites_counts += [sites_counts]
    all_sites_layers += [sites_layers]
    all_markers += [markers]
    all_markers_sites += [markers_sites]


# remap layers/types/sites

# make lookup table
new_all_layers = list(set(ni.core.py.flatten(all_layers)))
all_layers_indices = [[new_all_layers.index(l) for l in layers]
                      for layers in all_layers]
all_layers = new_all_layers

# remap layers
all_sites_layers = [ni.core.utils.merge_labels(x, l)
                    for x, l in zip(all_sites_layers, all_layers_indices)]
all_sites_layers = torch.cat(all_sites_layers)

# remap sites
cumulative_nsites = ni.core.py.cumsum([len(x) for x in all_sites], exclusive=True)
all_markers_sites = [x + n for (x, n) in zip(all_markers_sites, cumulative_nsites)]
all_markers_sites = torch.cat(all_markers_sites)

# stack remaining stuff
all_markers = torch.cat(all_markers)
all_sites = torch.cat(all_sites)
all_sites_volumes = torch.cat(all_sites_volumes)
all_sites_counts = torch.cat(all_sites_counts)

ni.io.savef(all_sites[None, None, None], f'{odir}/sites.nii.gz', affine=A_600)
ni.io.savef(all_markers[None, None, None], f'{odir}/markers.nii.gz', affine=A_600)
ni.io.save(all_sites_layers[None, None, None], f'{odir}/sites_layers.nii.gz')
ni.io.save(all_markers_sites[None, None, None], f'{odir}/markers_sites.nii.gz')
ni.io.save(all_sites_counts[None, None, None], f'{odir}/sites_counts.nii.gz')
ni.io.savef(all_sites_volumes[None, None, None], f'{odir}/sites_volumes.nii.gz')
with open(f'{odir}/list_layers.json', 'w') as f:
    json.dump(all_layers, f)


def layer_name_to_int(layer):
    if layer == 'Layer VI':
        return 6
    elif layer == 'Layer V':
        return 5
    elif layer == 'Layer IV':
        return 4
    elif layer == 'Layer III':
        return 3
    elif layer == 'Layer II':
        return 2
    elif layer == 'Layer I':
        return 1
    else:
        return 0

sites_json = ni.spatial.affine_matvec(A_600, all_sites)
sites_json = [{'legacy_stat': 1,
               'statistics': dict(volume=v.item(), count=c.item(),
                                  density=c.item()/v.item(),
                                  layer=layer_name_to_int(all_layers[l.item()])),
               'coordinates': dict(x=p[0].item(), y=p[1].item(), z=p[2].item())}
              for p, v, c, l in zip(sites_json, all_sites_volumes,
                                    all_sites_counts, all_sites_layers)]
sites_json = {
    'vox2ras': 'scanner_ras',
    'data_type': 'fs_pointset',
    'points': sites_json,
}
with open(f'{odir}/sites.json', 'w') as f:
    json.dump(sites_json, f)
