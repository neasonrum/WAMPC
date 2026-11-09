import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parent
bundle = json.loads((ROOT / 'results.json').read_text(encoding='utf-8-sig'))
pairs = bundle['gpu0']['pairs']
cases = [p for p in pairs if p['seed'] == 901]
names = ['Accident', 'Hazard at side lane', 'Parking / pedestrian', 'Yield to emergency vehicle']
fig, axes = plt.subplots(2, 4, figsize=(16, 9), constrained_layout=True)
colors = {'100': '#2166ac', '50': '#e98128'}
for col, (case, name) in enumerate(zip(cases, names)):
    for row, field in enumerate(['spatial', 'temporal_positions']):
        ax = axes[row, col]
        for steps in ['100', '50']:
            points = np.asarray(case['outputs'][steps][field])
            points = np.vstack([np.zeros((1, 2)), points])
            ax.plot(points[:, 1], points[:, 0], marker='o' if steps == '100' else 'x',
                    linewidth=2 if steps == '100' else 1.5, linestyle='-' if steps == '100' else '--',
                    markersize=5, color=colors[steps], label=f'{steps} steps')
        ax.scatter([0], [0], marker='^', s=70, color='#333333', zorder=5)
        ax.grid(alpha=.22)
        ax.set_aspect('equal', adjustable='datalim')
        ax.set_xlabel('Lateral right (m)')
        ax.set_ylabel('Forward (m)')
        ax.legend(loc='best', framealpha=.85, fontsize=9)
        if row == 0:
            ax.set_title(f'{name}\n10 spatial waypoints', fontsize=12)
            mean = case['differences']['spatial_mean_m'] * 100
            ax.text(.04, .97, f'Mean difference: {mean:.2f} cm', transform=ax.transAxes, va='top', fontsize=9,
                    bbox=dict(facecolor='white', alpha=.85, edgecolor='none'))
        else:
            a = case['outputs']['100']['target_speed_mps']
            b = case['outputs']['50']['target_speed_mps']
            ax.set_title(f'6 temporal positions: 0.5 to 3.0 s\nTarget speed: {a:.3f} vs {b:.3f} m/s', fontsize=10)
            mean = case['differences']['temporal_mean_m'] * 100
            ax.text(.04, .97, f'Mean difference: {mean:.2f} cm', transform=ax.transAxes, va='top', fontsize=9,
                    bbox=dict(facecolor='white', alpha=.85, edgecolor='none'))
fig.suptitle('V3: 100 vs 50 denoising steps | identical observations and Gaussian noise (seed 901)', fontsize=15)
fig.savefig(ROOT / 'trajectory_overlay.png', dpi=170)
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
for ax, metric, label in [(axes[0], 'spatial', 'Spatial waypoint index'),
                          (axes[1], 'temporal_positions', 'Future time (s)')]:
    for case, name in zip(cases, names):
        d = np.linalg.norm(np.array(case['outputs']['100'][metric]) - case['outputs']['50'][metric], axis=-1) * 100
        x = np.arange(1, len(d)+1) if metric == 'spatial' else .5 * np.arange(1, len(d)+1)
        ax.plot(x, d, marker='o', markersize=4, label=name)
    ax.set_xlabel(label)
    ax.set_ylabel('100-step / 50-step position difference (cm)')
    ax.set_ylim(bottom=0)
    ax.grid(alpha=.25)
    ax.legend(fontsize=8)
fig.suptitle('Magnified differences for the four illustrated cases (seed 901)')
fig.savefig(ROOT / 'trajectory_differences.png', dpi=170)
plt.close(fig)

cross = []
for a, b in zip(pairs, bundle['gpu1']['pairs']):
    assert a['sample_id'] == b['sample_id'] and a['seed'] == b['seed']
    for steps in ('100', '50'):
        assert a['initial_noise_sha256'] == b['initial_noise_sha256']
        for field in ('spatial', 'temporal_positions'):
            cross.append(float(np.max(np.abs(np.asarray(a['outputs'][steps][field]) - b['outputs'][steps][field]))))

means = {steps: np.mean([bundle[g]['report']['timing_ms'][steps]['wall']['mean'] for g in ['gpu0','gpu1']])
         for steps in ['100','50']}
summary = {
    'mean_ms_across_gpus': {k:float(v) for k,v in means.items()},
    'latency_reduction_percent': float(100 * (1 - means['50']/means['100'])),
    'speedup': float(means['100']/means['50']),
    'max_cross_gpu_coordinate_difference_m': max(cross),
    'per_sample_seed901': [{
        'sample': names[i], 'spatial_mean_cm': c['differences']['spatial_mean_m']*100,
        'temporal_mean_cm': c['differences']['temporal_mean_m']*100,
        'temporal_endpoint_cm': c['differences']['temporal_end_m']*100,
        'speed100_mps':c['outputs']['100']['target_speed_mps'],
        'speed50_mps':c['outputs']['50']['target_speed_mps'],
    } for i,c in enumerate(cases)],
    'difference_summary': bundle['gpu0']['report']['difference_summary'],
}
(ROOT / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')

lines = ['# V3 100-step versus 50-step inference comparison', '',
         'LSF allocation 13053006; two A100-SXM4-80GB GPUs tested sequentially and independently.',
         'Actual policy.forward, batch 1, BF16 autocast, FP32 parameters, eager execution, one CPU thread.',
         'Four real held-out observations; cached history; identical initial Gaussian noise in every pair.',
         'Forty interleaved timed forwards per schedule per GPU, after eight warmups per schedule.',
         'Trajectory comparisons: four observations x three seeds = 12 unique pairs, repeated on both GPUs.',
         'No CARLA simulation or closed-loop scoring. Sensor input transfer, preprocessing and disk are excluded.', '',
         '| GPU | 100 steps mean / P95 (ms) | 50 steps mean / P95 (ms) |', '|---|---:|---:|']
for g in ['gpu0','gpu1']:
    t=bundle[g]['report']['timing_ms']
    lines.append(f"| {g} | {t['100']['wall']['mean']:.2f} / {t['100']['wall']['p95']:.2f} | {t['50']['wall']['mean']:.2f} / {t['50']['wall']['p95']:.2f} |")
lines += ['', f"Across GPUs: {summary['latency_reduction_percent']:.2f}% less latency; {summary['speedup']:.3f}x speedup.",
          f"Maximum coordinate difference between GPU repeats: {max(cross):.8f} m.", '',
          '## Differences between 100-step and 50-step predictions', '',
          'These are prediction-to-prediction differences, not errors against ground truth.', '',
          '| Quantity | Mean over 12 pairs | Worst pair |', '|---|---:|---:|']
for k,title,scale,unit in [('spatial_mean_m','Spatial mean waypoint distance',100,'cm'),
                         ('spatial_max_m','Spatial maximum waypoint distance',100,'cm'),
                         ('temporal_mean_m','Temporal mean position distance',100,'cm'),
                         ('temporal_end_m','Temporal 3-second endpoint distance',100,'cm'),
                         ('target_speed_abs_mps','Absolute target-speed difference',1,'m/s')]:
    d=summary['difference_summary'][k]
    lines.append(f"| {title} | {d['mean']*scale:.4f} {unit} | {d['maximum']*scale:.4f} {unit} |")
lines += ['', '## State after testing', '',
          'Production configuration remains 100 steps. No training or model architecture was changed.',
          'Timed BEV world-model training remains paused at update 149300, during epoch 22.',
          'latest.pt was checked to contain populated optimizer state. STOP remains in place; no automatic resume.', '',
          '## Additional profiling', '',
          'An instrumented 100-step forward was recorded after the uninstrumented timings.',
          'Its timings have profiler overhead and must not replace the wall-clock benchmark above.',
          'See gpu0/profile_operators.json and gpu0/profile_100.json for attribution.', '',
          '![Trajectory overlay](trajectory_overlay.png)', '', '![Magnified differences](trajectory_differences.png)', '']
(ROOT / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
print(json.dumps(summary, indent=2))
