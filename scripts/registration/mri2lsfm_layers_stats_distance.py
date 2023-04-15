import json
import statistics

root = '.'
root += 'warp2lsfm/slide_center_oct'
fname = f'{root}/infrasupra_distance_results.json'

with open(fname, 'rt') as f:
    results = json.load(f)


vx = 9.9  # 9.9 um

for metric, metric_results in results.items():
    val = []
    for slide, slide_results in metric_results.items():
        for plane, plane_results in slide_results.items():
            if plane_results:
                val += [plane_results]
    print(metric)
    print(' - mean: ', statistics.mean(val) * vx)
    print(' - std:  ', statistics.stdev(val) * vx)

