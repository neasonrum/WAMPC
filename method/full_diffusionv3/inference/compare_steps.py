"""Read-only model comparison: actual V3 forward with 100 and 50 DDIM steps."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import statistics
import subprocess
import time

import numpy as np
import torch

from method.irwm_bev_t.launch import guard_context, memory_snapshot, write_json, ROOT

CACHE = Path('/fs/scratch/projects/common_ai/nix3sgh_cache/WMMPC')
RUN = CACHE / 'irwm_bev_t_dt05_h3_b32_train_20261106_v1'
INPUTS = RUN / 'inference_latency_20261106/real_inputs.pt'
CHECKPOINT = CACHE / 'results/full_diffusionv3_dual_query64_automot_pid_train_20261104_v1/epoch_10.pt'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def describe(values):
    return dict(count=len(values), mean=float(np.mean(values)),
                p50=float(np.median(values)), p95=float(np.quantile(values, .95)),
                minimum=float(min(values)), maximum=float(max(values)))


def log(event, **kwargs):
    print(json.dumps(dict(event=event, utc=datetime.now(timezone.utc).isoformat(), **kwargs)), flush=True)


@torch.no_grad()
def main(args):
    guard_context('13053006')
    out = args.output.resolve()
    assert out.is_relative_to(CACHE.resolve())
    out.mkdir(parents=True, exist_ok=False)
    stopped = json.loads((RUN / 'stopped.json').read_text())
    assert (RUN / 'STOP').exists() and stopped['reason'] == 'requested_stop'
    assert json.loads((RUN / 'node0.status.json').read_text())['state'] == 'completed'
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    if processes:
        raise RuntimeError('Expected both allocated GPUs idle for isolated benchmark: ' + processes)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.cuda.set_device(args.device)
    device = torch.device('cuda', args.device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.manual_seed(901)
    from method.full_diffusion.garage import FrozenGaragePerception
    from method.full_diffusion2.data import frame_pose
    from method.full_diffusionv3.inference.policy import FullDiffusionV3Network
    training_checkpoint = torch.load(RUN / 'latest.pt', map_location='cpu', weights_only=False)
    checkpoint_keys = list(training_checkpoint)
    optimizer_key = next((k for k in checkpoint_keys if 'optim' in k.lower()), None)
    assert optimizer_key is not None
    resume_audit = dict(checkpoint_keys=checkpoint_keys, optimizer_key=optimizer_key,
                        optimizer_state_entries=len(training_checkpoint[optimizer_key]['state']),
                        stopped=stopped)
    del training_checkpoint
    perception = FrozenGaragePerception().to(device).eval()
    policy = FullDiffusionV3Network(perception.net, CHECKPOINT, device).eval()
    original_steps = policy.run_config['diffusion']['inference_steps']
    assert original_steps == 100
    items = torch.load(INPUTS, map_location='cpu', weights_only=False)
    samples = []
    for item in items:
        poses = item['poses']
        matrices = [frame_pose({'theta': float(p[2]), 'pos_global': p[:2]}) for p in poses]
        samples.append(dict(batch={k: v.to(device) for k, v in item['batch'].items()},
                            raw=item['raw_bev_history'].to(device), matrices=matrices,
                            poses=poses, sample_id=item['sample_id']))
    report = dict(host=socket.gethostname(), gpu=args.device, device=torch.cuda.get_device_name(device),
                  cpu_affinity=sorted(os.sched_getaffinity(0)), cpu_threads=torch.get_num_threads(),
                  torch_version=torch.__version__, batch_size=1, precision='BF16 autocast, FP32 parameters',
                  compile=False, checkpoint=str(CHECKPOINT), checkpoint_sha256=sha(CHECKPOINT),
                  inputs=str(INPUTS), input_sha256=sha(INPUTS), original_steps=original_steps,
                  training_resume_audit=resume_audit, sample_ids=[s['sample_id'] for s in samples],
                  exclusions='CARLA simulation/rendering, disk, sensor preprocessing and input H2D',
                  timing_protocol='isolated GPU; synchronized wall clock; AB/BA interleaved; 40 forwards per step count',
                  source_sha256={str(p.relative_to(ROOT)): sha(p) for p in [
                      ROOT/'method/full_diffusionv3/model.py', ROOT/'method/full_diffusionv3/diffusion.py',
                      ROOT/'method/full_diffusionv3/inference/policy.py']})
    write_json(out / 'metadata.json', report)
    log('models_loaded', gpu=args.device, memory=memory_snapshot())

    def setup(s, steps, seed):
        policy.run_config['diffusion']['inference_steps'] = steps
        policy.generator.manual_seed(seed)
        policy.history.entries.clear()
        policy.history.entries.append((-.5, s['raw'][:, 0], s['matrices'][0]))
        policy.history.entries.append((-.25, s['raw'][:, 1], s['matrices'][1]))
        policy.set_observation(s['poses'][-1], 0.)

    def forward(s):
        b = s['batch']
        return policy(rgb=b['rgb'], lidar_bev=b['lidar'], target_point=b['target_point'],
                      ego_vel=b['speed'], command=b['command'])

    # Warm both schedules and all four real sensor observations before timing.
    for steps in (100, 50):
        for i in range(8):
            s = samples[i % len(samples)]
            setup(s, steps, 901)
            forward(s)
    torch.cuda.synchronize(device)
    timings = {100: [], 50: []}
    event_times = {100: [], 50: []}
    start = torch.cuda.Event(enable_timing=True)
    finish = torch.cuda.Event(enable_timing=True)
    for i in range(40):
        s = samples[i % len(samples)]
        for steps in ((100, 50) if i % 2 == 0 else (50, 100)):
            setup(s, steps, 901 + i % 3)
            torch.cuda.synchronize(device)
            begin = time.perf_counter()
            start.record()
            output = forward(s)
            finish.record()
            finish.synchronize()
            timings[steps].append(1000 * (time.perf_counter() - begin))
            event_times[steps].append(start.elapsed_time(finish))
        if (i + 1) % 10 == 0:
            log('timing_progress', gpu=args.device, pairs=i+1,
                means_ms={str(k): statistics.mean(v) for k, v in timings.items()})
    report['timing_ms'] = {str(k): dict(wall=describe(v), cuda_stream=describe(event_times[k])) for k, v in timings.items()}
    write_json(out / 'timing.json', report['timing_ms'])

    # Separate capture passes avoid instrumentation overhead in reported latency.
    pairs = []
    for idx, s in enumerate(samples):
        for seed in (901, 902, 903):
            case = dict(sample_index=idx, sample_id=s['sample_id'], seed=seed, outputs={})
            first_noises = {}
            for steps in (100, 50):
                first_noise = []
                calls = []
                def capture(module, arguments):
                    calls.append(1)
                    if not first_noise:
                        first_noise.append(arguments[0].detach().clone())
                hook = policy.head.register_forward_pre_hook(capture)
                try:
                    setup(s, steps, seed)
                    output = forward(s)
                    torch.cuda.synchronize(device)
                finally:
                    hook.remove()
                assert len(calls) == steps
                first_noises[steps] = first_noise[0].cpu()
                spatial = output[2][0].float().cpu().numpy()
                delta = policy.last_temporal_delta[0].float().cpu().numpy()
                temporal = np.cumsum(delta, axis=0)
                case['outputs'][str(steps)] = dict(spatial=spatial.tolist(), temporal_delta=delta.tolist(),
                                                temporal_positions=temporal.tolist(),
                                                target_speed_mps=policy.last_target_speed,
                                                head_calls=len(calls))
            assert torch.equal(first_noises[100], first_noises[50])
            case['initial_noise_identical'] = True
            case['initial_noise_sha256'] = hashlib.sha256(first_noises[100].numpy().tobytes()).hexdigest()
            a, b = case['outputs']['100'], case['outputs']['50']
            distance_s = np.linalg.norm(np.array(a['spatial']) - b['spatial'], axis=-1)
            distance_t = np.linalg.norm(np.array(a['temporal_positions']) - b['temporal_positions'], axis=-1)
            case['differences'] = dict(spatial_mean_m=float(distance_s.mean()), spatial_end_m=float(distance_s[-1]),
                                       spatial_max_m=float(distance_s.max()), temporal_mean_m=float(distance_t.mean()),
                                       temporal_end_m=float(distance_t[-1]), temporal_max_m=float(distance_t.max()),
                                       temporal_distance_by_horizon_m=distance_t.tolist(),
                                       target_speed_abs_mps=abs(a['target_speed_mps'] - b['target_speed_mps']))
            pairs.append(case)
        log('comparison_progress', gpu=args.device, samples=idx+1)
    write_json(out / 'pairs.json', pairs)
    report['difference_summary'] = {key: describe([p['differences'][key] for p in pairs])
                                    for key in pairs[0]['differences'] if key != 'temporal_distance_by_horizon_m'}

    # Optional one-forward attribution, collected only after the uninstrumented timings.
    if args.profile:
        try:
            setup(samples[0], 100, 901)
            torch.cuda.synchronize(device)
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                    torch.profiler.ProfilerActivity.CUDA]) as prof:
                forward(samples[0])
                torch.cuda.synchronize(device)
            prof.export_chrome_trace(str(out / 'profile_100.json'))
            rows = [dict(name=e.key, calls=e.count, cpu_total_us=e.cpu_time_total,
                         cpu_self_us=e.self_cpu_time_total,
                         device_total_us=e.device_time_total, device_self_us=e.self_device_time_total)
                    for e in prof.key_averages()]
            write_json(out / 'profile_operators.json', rows)
            (out / 'profile_cpu.txt').write_text(prof.key_averages().table(sort_by='self_cpu_time_total', row_limit=20))
            (out / 'profile_cuda.txt').write_text(prof.key_averages().table(sort_by='self_cuda_time_total', row_limit=20))
            report['profile'] = 'complete; instrumented timings are not used in latency comparison'
        except Exception as exc:
            report['profile_error'] = repr(exc)
    policy.run_config['diffusion']['inference_steps'] = original_steps
    report['peak_gpu_allocated_gib'] = torch.cuda.max_memory_allocated(device) / 2**30
    report['memory'] = memory_snapshot()
    report['completed_utc'] = datetime.now(timezone.utc).isoformat()
    report['completed'] = True
    write_json(out / 'report.json', report)
    log('completed', gpu=args.device, timing_ms=report['timing_ms'], differences=report['difference_summary'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, choices=(0, 1), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--profile', action='store_true')
    main(parser.parse_args())
