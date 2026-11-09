"""Read-only replay diagnosis of the original captured policy call."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.bench2drive import check_job
from evaluation.shared_gpu_inference import load_policy, pack, unpack
from evaluation.verify_shared_gpu_policy import difference

p = argparse.ArgumentParser()
p.add_argument('--job-id', required=True)
p.add_argument('--capture', type=Path, required=True)
p.add_argument('--checkpoint', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
check_job(a.job_id)
assert a.job_id in Path('/proc/self/cgroup').read_text()
for v in (a.capture, a.checkpoint, a.output):
    v.resolve().relative_to(ROOT.resolve())
uuids = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip().splitlines()
occupied = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True)
assert uuids[0] not in occupied
os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:4])
os.environ.update(CUDA_VISIBLE_DEVICES=uuids[0], IS_BENCH2DRIVE='True', OMP_NUM_THREADS='1')
os.environ.pop('SAVE_PATH', None)
with (ROOT/'.runtime/carla_gpu_locks'/f'{uuids[0]}.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    import torch
    torch.set_num_threads(1)
    c = torch.load(a.capture, map_location='cpu', weights_only=False)
    inputs = unpack(c['inputs'])
    reference = unpack(c['output'])
    net = load_policy(a.checkpoint)
    initial = net.generator.get_state()
    report = {'unix': time.time(), 'input_layout': {k: {'shape': list(v.shape), 'stride': list(v.stride())} for k, v in inputs.items() if isinstance(v, torch.Tensor)}, 'cases': {}}
    first = None
    with torch.inference_mode():
        for name in ('captured_contiguous', 'captured_repeat', 'rgb_channels_last'):
            net.generator.set_state(initial)
            data = {k: v.to('cuda:0') if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            if name == 'rgb_channels_last':
                data['rgb'] = data['rgb'].contiguous(memory_format=torch.channels_last)
            out = unpack(pack(net.forward(**data)))
            comparisons = difference(reference, out, name)
            report['cases'][name] = {'comparisons': comparisons, 'speed_difference': abs(c['last_target_speed']-net.last_target_speed)}
            if first is None:
                first = out
            else:
                report['cases'][name]['versus_first'] = difference(first, out, 'repeat')
    a.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)
