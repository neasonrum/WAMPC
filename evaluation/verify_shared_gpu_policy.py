"""Replay a real original-agent input and verify isolated policy RNG streams."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.bench2drive import check_job
from evaluation.shared_gpu_inference import load_policy, pack, unpack


def difference(reference, actual, label):
    import torch
    rows = []
    if isinstance(reference, torch.Tensor):
        assert reference.shape == actual.shape, label
        left = reference.detach().cpu().float(); right = actual.detach().cpu().float()
        error = float((left-right).abs().max()) if left.numel() else 0.0
        rows.append({'field': label, 'max_absolute_difference': error,
                     'allclose_1e_5': bool(torch.allclose(left, right, atol=1e-5, rtol=1e-5))})
    elif isinstance(reference, (tuple, list)):
        assert len(reference) == len(actual)
        for i, (left, right) in enumerate(zip(reference, actual)):
            rows.extend(difference(left, right, f'{label}[{i}]'))
    elif isinstance(reference, dict):
        assert reference.keys() == actual.keys()
        for key in reference:
            rows.extend(difference(reference[key], actual[key], f'{label}.{key}'))
    else:
        assert reference == actual, label
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--job-id', required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', type=int, choices=(0, 1), default=0)
    args = parser.parse_args()
    check_job(args.job_id)
    assert args.job_id in Path('/proc/self/cgroup').read_text()
    for value in (args.checkpoint, args.capture, args.output):
        value.resolve().relative_to(ROOT.resolve())
    uuids = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip().splitlines()
    uuid = uuids[args.gpu]
    occupied = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True)
    assert uuid not in occupied
    cpus = sorted(os.sched_getaffinity(0)); half = 4
    os.sched_setaffinity(0, cpus[args.gpu*half:(args.gpu+1)*half])
    os.environ['CUDA_VISIBLE_DEVICES'] = uuid
    os.environ['IS_BENCH2DRIVE'] = 'True'
    os.environ.pop('SAVE_PATH', None)
    lock_path = ROOT / '.runtime/carla_gpu_locks' / f'{uuid}.lock'
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        import torch
        torch.set_num_threads(1)
        capture = torch.load(args.capture, map_location='cpu', weights_only=False)
        inputs = unpack(capture['inputs'])
        transported = unpack(pickle.loads(pickle.dumps(capture['inputs'], protocol=5)))
        transport = difference(inputs, transported, 'transport')
        assert all(row['max_absolute_difference'] == 0 for row in transport)
        policy_a = load_policy(args.checkpoint)
        policy_b = load_policy(args.checkpoint)
        with torch.inference_mode():
            gpu_inputs = {k: v.to('cuda:0') if isinstance(v, torch.Tensor) else v for k, v in transported.items()}
            first_a = unpack(pack(policy_a.forward(**gpu_inputs)))
            first_speed = policy_a.last_target_speed
            first_b = unpack(pack(policy_b.forward(**gpu_inputs)))
            second_a = unpack(pack(policy_a.forward(**gpu_inputs)))
            second_b = unpack(pack(policy_b.forward(**gpu_inputs)))
        replay = difference(unpack(capture['output']), first_a, 'original_gpu_replay')
        isolation_first = difference(first_a, first_b, 'session_first_call')
        isolation_second = difference(second_a, second_b, 'session_second_call')
        checks = replay + isolation_first + isolation_second
        result = {'success': all(row['allclose_1e_5'] for row in checks),
                  'transport_lossless': True, 'comparisons': checks,
                  'rng_advanced': not torch.equal(first_a[2], second_a[2]),
                  'target_speed_difference': abs(first_speed-capture['last_target_speed']),
                  'inference_steps': 100, 'sampler': 'Original VPFullDiffusion',
                  'gpu_uuid': uuid, 'unix': time.time()}
        args.output.write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result), flush=True)
        assert result['success'] and result['rng_advanced'] and result['target_speed_difference'] < 1e-5
