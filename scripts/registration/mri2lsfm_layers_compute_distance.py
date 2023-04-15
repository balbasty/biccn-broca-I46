import numpy as np
import nitorch as ni
import json
from metrics.labels import min_distance_boundary, hausdorff_boundary


LABELS_INFRA_MRI = [2]
LABELS_SUPRA_MRI = [3]
LABELS_INFRA_SPIM = [4, 5]
LABELS_SUPRA_SPIM = [1, 2, 3]
LABEL_INFRA = 1
LABEL_SUPRA = 2


ROOT = '.'
MRI_SEG = ROOT + '/warp2lsfm/slide_center_oct/infrasupra.smooth.crop.{:02d}.nii.gz'
LSFM_SEG = ROOT + '/lsfm/slide_center_oct/layers.{:02d}.nii.gz'
LSFM = ROOT + '/lsfm/slide_center_oct/neun.{:02d}.nii.gz'
RESULTS = ROOT + '/warp2lsfm/slide_center_oct/infrasupra_distance_results.json'
LSFM_THRESHOLD = 200


def relabel(x, in_labels, out_labels):
    y = np.zeros_like(x)
    for il, ol in zip(in_labels, out_labels):
         y[np.isin(x, il)] = ol
    return y


def find_sections(x):
    x = x > 0
    x = np.any(x.reshape([-1, x.shape[-1]]), axis=0)
    idx = np.nonzero(x)[0]
    return idx


results = dict(hausdorff={}, hausdorff95={}, hausdorff75={}, hausdorff50={}, meandist={})


for i in [8, 9, 10, 11, 12, 17, 19]:
    print(i)
    for key in results:
        results[key][i] = {}

    fname_mri_seg = MRI_SEG.format(i)
    fname_lsfm_seg = LSFM_SEG.format(i)
    fname_lsfm = LSFM.format(i)
    mri = ni.io.load(fname_mri_seg, dtype='int16', numpy=True).squeeze()
    lsfm = ni.io.load(fname_lsfm_seg, dtype='int16', numpy=True).squeeze()
    mri = relabel(mri, [LABELS_INFRA_MRI, LABELS_SUPRA_MRI], [LABEL_INFRA, LABEL_SUPRA])
    lsfm = relabel(lsfm, [LABELS_INFRA_SPIM, LABELS_SUPRA_SPIM], [LABEL_INFRA, LABEL_SUPRA])
    mask = mri > 0
    mask = np.bitwise_and(mask, ni.io.loadf(fname_lsfm, dtype='float32', numpy=True).squeeze() > LSFM_THRESHOLD)
    mri[~mask] = 0
    lsfm[~mask] = 0

    indices = find_sections(lsfm)
    print(indices, lsfm.shape)
    for idx in indices:
        print('  ', idx)
        mri1 = mri[..., idx]
        lsfm1 = lsfm[..., idx]
        
        h = hausdorff_boundary(mri1, lsfm1, labels=[LABEL_INFRA, LABEL_SUPRA], symmetric=False)
        d = min_distance_boundary(mri1, lsfm1, labels=[LABEL_INFRA, LABEL_SUPRA])
        h95 = min_distance_boundary(mri1, lsfm1, labels=[LABEL_INFRA, LABEL_SUPRA], reduction=lambda x: np.quantile(x, 0.95))
        h75 = min_distance_boundary(mri1, lsfm1, labels=[LABEL_INFRA, LABEL_SUPRA], reduction=lambda x: np.quantile(x, 0.75))
        h50 = min_distance_boundary(mri1, lsfm1, labels=[LABEL_INFRA, LABEL_SUPRA], reduction=lambda x: np.quantile(x, 0.50))
        print('      hausdorff:     ', h)
        print('      hausdorff (95):', h95)
        print('      hausdorff (75):', h75)
        print('      hausdorff (50):', h50)
        print('      mean:          ', d)
        results['meandist'][i][int(idx)] = d
        results['hausdorff'][i][int(idx)] = h
        results['hausdorff95'][i][int(idx)] = h95
        results['hausdorff75'][i][int(idx)] = h75
        results['hausdorff50'][i][int(idx)] = h50


with open(RESULTS, 'w') as outfile:
    json.dump(results, outfile)

