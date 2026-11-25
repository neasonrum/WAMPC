"""Verify transport and alternating independent histories/RNG against direct GPU calls."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import pickle
import time

import numpy as np
import torch
from torch.utils.data import default_collate

from evaluation.pointquery_parallel.broker import load_policy, forward
from evaluation.pointquery_parallel.codec import pack, unpack
from evaluation.pointquery_parallel.common import OUTPUT, RUN, ARTIFACTS, save
from method.full_diffusion_pointquery.data import PointDataset


def compare(left, right, name, differences):
    if isinstance(left, torch.Tensor):
        if not isinstance(right, torch.Tensor) or left.shape != right.shape or left.dtype != right.dtype:
            raise AssertionError(f'Tensor shape/dtype changed: {name}')
        error = float((left.float().cpu() - right.float().cpu()).abs().max()) if left.numel() else 0.
        differences[name] = max(differences.get(name, 0.), error)
        torch.testing.assert_close(left.cpu(), right.cpu(), atol=1e-5, rtol=1e-5)
    elif isinstance(left, (tuple, list)):
        assert type(left) is type(right) and len(left) == len(right), name
        for index, (a, b) in enumerate(zip(left, right)):
            compare(a, b, f'{name}/{index}', differences)
    elif left is None:
        assert right is None, name
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    else:
        assert left == right, name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', required=True)
    parser.add_argument('--gpu', type=int, choices=(0, 1), required=True)
    args = parser.parse_args()
    if f'/job.{args.job}.' not in Path('/proc/self/cgroup').read_text():
        raise RuntimeError('Preflight must run in the supplied allocation')
    drained = ARTIFACTS / f'drained_{args.job}_{args.gpu}.json'
    assert drained.exists(), 'Wait for completed-route handoff first'
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    os.environ['IS_BENCH2DRIVE'] = 'True'
    os.environ.pop('SAVE_PATH', None)
    torch.cuda.reset_peak_memory_stats()
    cfg = json.loads((OUTPUT / 'config.json').read_text())
    dataset = PointDataset(cfg)
    examples = []
    for index in (0, 1, 2):
        batch = default_collate([dataset[index]])
        rid, slot, frame = map(int, dataset.samples[index])
        examples.append(({'rgb': batch['rgb'], 'lidar_bev': batch['lidar'],
                          'target_point': batch['target_point'], 'ego_vel': batch['speed'],
                          'command': batch['command'], 'target_point_next': None},
                         dataset.metadata(rid)['poses'][slot], frame * .25))
    for tensor in (torch.randn(1, 3, 16, 32), torch.randn(1, 3, 16, 32).to(memory_format=torch.channels_last),
                   torch.randn(2, 4, dtype=torch.bfloat16).transpose(0, 1)):
        restored = unpack(pickle.loads(pickle.dumps(pack(tensor), protocol=5)), 'cuda:0')
        torch.testing.assert_close(tensor, restored.cpu(), atol=0, rtol=0)
        assert tensor.stride() == restored.stride()
    checkpoint = RUN / 'selected_best.pt'
    reference = [load_policy(checkpoint) for _ in range(2)]
    remote = [load_policy(checkpoint) for _ in range(2)]
    differences, steps = {}, []
    for index, (inputs, pose, timestamp) in enumerate(examples):
        for session in (0, 1):
            altered = dict(inputs)
            if session:
                altered['ego_vel'] = inputs['ego_vel'] + .4
            packet = pickle.loads(pickle.dumps({'inputs': pack(altered), 'pose': pose.copy(),
                                                'timestamp': timestamp}, protocol=5))
            with torch.inference_mode():
                direct = {k: v.to('cuda:0') if isinstance(v, torch.Tensor) else v for k, v in altered.items()}
                restored = unpack(packet['inputs'], 'cuda:0')
                for key, value in direct.items():
                    if isinstance(value, torch.Tensor):
                        torch.testing.assert_close(value, restored[key], atol=0, rtol=0)
                        assert value.stride() == restored[key].stride(), key
                reference[session].set_observation(pose, timestamp)
                expected = reference[session].forward(**direct)
                actual = forward(remote[session], packet)
                compare(expected, unpack(actual['output']), f'session{session}/output', differences)
                compare(reference[session].last_temporal_delta, unpack(actual['last_temporal_delta']),
                        f'session{session}/temporal_delta', differences)
                if expected[6] is not None:
                    compare(reference[session].convert_features_to_bb_metric(expected[6]), actual['decoded_boxes'],
                            f'session{session}/decoded_boxes', differences)
                speed_difference = abs(reference[session].last_target_speed - actual['last_target_speed'])
                assert speed_difference <= 1e-5, speed_difference
                assert torch.equal(reference[session].generator.get_state(), remote[session].generator.get_state())
                assert len(reference[session].history.entries) == actual['history_entries'] == index + 1
                steps.append({'session': session, 'observation': index, 'speed_difference_mps': speed_difference,
                              'history_entries': actual['history_entries']})
    report = {'passed': True, 'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              'inference_steps': 25, 'real_observations_per_session': 3, 'sessions': 2,
              'tensor_values_and_strides_exact': True, 'RNG_state_exact_each_forward': True,
              'GPU_detection_decoding_exact': True,
              'independent_pose_and_history': True, 'max_absolute_differences': differences,
              'steps': steps, 'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30, 'unix': time.time()}
    save(ARTIFACTS / 'preflight.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()

