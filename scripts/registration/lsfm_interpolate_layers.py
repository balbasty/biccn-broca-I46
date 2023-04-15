from nitorch import io
from nitorch.spatial import euclidean_distance_transform as edt
import torch

root = '.'
root += '/lsfm/layers'

for i in (8, 9, 10, 11, 12, 17, 19):
    fname = f'{root}/extracted_9.9um/layers.{i:02d}.nii.gz'
    f = io.map(fname)
    d = f.data(device='cuda')
    planemask = d.reshape([-1, d.shape[2]]).any(0)
    d[:, :, planemask] += (d[:, :, planemask] == 0) * 192
    labels = d.unique().sort()[0][1:]
    mindist = torch.full_like(d, float('inf'), dtype=torch.float32)
    new_d = torch.empty_like(d)
    print(labels)
    for label in labels:
        dist = edt(d != label, dim=3)
        mask = dist < mindist
        new_d.masked_fill_(mask, label)
        mindist[mask] = dist[mask]
    new_d.masked_fill_(new_d == 192, 0)
    io.save(new_d, f'{root}/interpolated_9.9um/layers.{i:02d}.nii.gz', like=f)
