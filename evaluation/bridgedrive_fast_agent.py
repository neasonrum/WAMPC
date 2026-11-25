"""Telemetry for the 20-step PDM-Lite BridgeDrive closed-loop agent."""

import json
import time

from evaluation.full_diffusion2_fast_agent import FullDiffusionTiming
from method.bridgedrive.carla_agent import BridgeDriveAgent
from method.bridgedrive.model import METHOD, VPBridge


def get_entry_point():
    return 'BridgeDriveFastAgent'


class BridgeDriveFastAgent(FullDiffusionTiming, BridgeDriveAgent):
    method = METHOD
    sampler_type = VPBridge
    inference_steps = 20
    log_prefix = 'BRIDGEDRIVE'

    def sampler_and_steps(self, network):
        return network.head.bridge, network.run_config['bridge']['inference_steps']

    def run_step(self, input_data, timestamp, sensors=None):
        started = time.perf_counter()
        control = BridgeDriveAgent.run_step(self, input_data, timestamp, sensors)
        if self.step < 3 or self.step % 20 == 0:
            record = {'unix': time.time(), 'route': self.route_name, 'step': self.step,
                      'sim_seconds': float(timestamp), 'run_step_seconds': time.perf_counter() - started,
                      'network_seconds': self._network_seconds, 'inference_steps': self.inference_steps,
                      'observed_speed_mps': float(input_data['speed'][1]['speed']),
                      'model_target_speed_mps': self.nets[0].last_target_speed,
                      'control': {'steer': float(control.steer), 'throttle': float(control.throttle),
                                  'brake': float(control.brake)},
                      'history_entries': 0, 'path_xy': self._last_path}
            with (self.directory / 'telemetry.jsonl').open('a') as handle:
                handle.write(json.dumps(record) + '\n')
            print(self.log_prefix + '_TICK ' + json.dumps({k: v for k, v in record.items() if k != 'path_xy'}), flush=True)
        return control
