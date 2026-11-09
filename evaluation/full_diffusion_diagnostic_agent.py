"""Existing Full Diffusion agent with route metrics and passive tick telemetry."""
import json
import os
from pathlib import Path
import time

from method.full_diffusion.inference.carla_agent import FullDiffusionGeoAgent


def get_entry_point():
    return 'FullDiffusionDiagnosticAgent'


class FullDiffusionDiagnosticAgent(FullDiffusionGeoAgent):
    def setup(self, path_to_conf_file, route_index=None, traffic_manager=None):
        parts = path_to_conf_file.split('+')
        # The Bench2Drive evaluator passes its save_name after '+', not as an
        # argument. Forward it so Garage's metric and scenario logging is enabled.
        if route_index is None and len(parts) > 1:
            route_index = parts[-1]
        self._diagnostic_route = route_index
        self._telemetry = Path(os.environ['WMMPC_CARLA_LOG_DIR']) / 'telemetry.jsonl'
        self._last_path = None
        self._network_seconds = None
        super().setup(path_to_conf_file, route_index, traffic_manager)
        # Garage calls net.forward(...) directly, bypassing nn.Module hooks.
        forward = self.nets[0].forward

        def diagnostic_forward(*args, **kwargs):
            started = time.perf_counter()
            output = forward(*args, **kwargs)
            self._network_seconds = time.perf_counter() - started
            if self.step < 3 or self.step % 20 == 0:
                self._last_path = output[2][0].detach().float().cpu().tolist()
            return output

        self.nets[0].forward = diagnostic_forward
        print(f'FULL_DIFFUSION_READY route={route_index} checkpoint={parts[0]}', flush=True)

    def run_step(self, input_data, timestamp, sensors=None):
        started = time.perf_counter()
        control = super().run_step(input_data, timestamp, sensors)
        if self.step < 3 or self.step % 20 == 0:
            record = {'unix': time.time(), 'route': self._diagnostic_route, 'step': self.step,
                      'sim_seconds': float(timestamp),
                      'observed_speed_mps': float(input_data['speed'][1]['speed']),
                      'model_target_speed_mps': self.nets[0].last_target_speed,
                      'control': {'steer': float(control.steer), 'throttle': float(control.throttle),
                                  'brake': float(control.brake)},
                      'run_step_seconds': time.perf_counter() - started,
                      'network_seconds': self._network_seconds, 'path_xy': self._last_path,
                      'filtered_state': self.state_log[-1].tolist() if self.state_log else None}
            with self._telemetry.open('a') as handle:
                handle.write(json.dumps(record) + '\n')
            print('FULL_DIFFUSION_TICK ' + json.dumps({k: v for k, v in record.items()
                                                    if k not in ('path_xy', 'filtered_state')}), flush=True)
        return control
