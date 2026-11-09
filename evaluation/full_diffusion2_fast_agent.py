"""The trained FD2 agent with passive timing; original sampler and control."""
import json
import os
from pathlib import Path
import resource
import time

from method.full_diffusion2.inference.carla_agent import FullDiffusion2GeoAgent
from method.full_diffusion.diffusion import VPFullDiffusion


def get_entry_point():
    return 'FullDiffusion2FastAgent'


class FullDiffusionTiming:
    method = 'full_diffusion2'
    sampler_type = VPFullDiffusion
    log_prefix = 'FULL_DIFFUSION2'

    def event(self, event, **values):
        with (self.directory/'route_events.jsonl').open('a') as handle:
            handle.write(json.dumps({'event': event, 'route': self.route_name,
                                     'unix': time.time(), **values})+'\n')

    def setup(self, path_to_conf_file, route_index=None, traffic_manager=None):
        parts = path_to_conf_file.split('+')
        self.route_name = route_index or (parts[-1] if len(parts) > 1 else None)
        self.directory = Path(os.environ['WMMPC_CARLA_LOG_DIR'])
        self._network_seconds = None
        self._last_path = None
        self.event('agent_setup_started')
        super().setup(path_to_conf_file, self.route_name, traffic_manager)
        if self.nets[0].run_config['diffusion']['inference_steps'] != 100:
            raise ValueError('Exactly 100 diffusion steps are required')
        if self.nets[0].run_config.get('method') != self.method:
            raise ValueError('The checkpoint method differs from the evaluation agent')
        if type(self.nets[0].diffusion) is not self.sampler_type:
            raise TypeError(f'The original {self.sampler_type.__name__} sampler is required')
        forward = self.nets[0].forward

        def timed_forward(*args, **kwargs):
            start = time.perf_counter()
            result = forward(*args, **kwargs)
            self._network_seconds = time.perf_counter()-start
            if self.step < 3 or self.step % 20 == 0:
                self._last_path = result[2][0].detach().float().cpu().tolist()
            return result

        self.nets[0].forward = timed_forward
        self.event('agent_setup_ready', inference_steps=100, sampler=self.sampler_type.__name__,
                   sampler_replaced=False, method=self.method)
        print(f'{self.log_prefix}_READY route={self.route_name}', flush=True)

    def extra_telemetry(self):
        return {}

    def run_step(self, input_data, timestamp, sensors=None):
        started = time.perf_counter()
        cpu_start = resource.getrusage(resource.RUSAGE_SELF)
        control = super().run_step(input_data, timestamp, sensors)
        if self.step < 3 or self.step % 20 == 0:
            cpu_end = resource.getrusage(resource.RUSAGE_SELF)
            history = self.nets[0].history
            record = {'unix': time.time(), 'route': self.route_name, 'step': self.step,
                      'sim_seconds': float(timestamp), 'run_step_seconds': time.perf_counter()-started,
                      'network_seconds': self._network_seconds,
                      'process_cpu_seconds': cpu_end.ru_utime+cpu_end.ru_stime-cpu_start.ru_utime-cpu_start.ru_stime,
                      'observed_speed_mps': float(input_data['speed'][1]['speed']),
                      'model_target_speed_mps': self.nets[0].last_target_speed,
                      'control': {'steer': float(control.steer), 'throttle': float(control.throttle),
                                  'brake': float(control.brake)},
                      'history_entries': len(history.entries), 'path_xy': self._last_path,
                      'filtered_state': self.state_log[-1].tolist() if self.state_log else None}
            record.update(self.extra_telemetry())
            with (self.directory/'telemetry.jsonl').open('a') as handle:
                handle.write(json.dumps(record)+'\n')
            print(self.log_prefix+'_TICK '+json.dumps({k:v for k,v in record.items()
                  if k not in ('path_xy', 'filtered_state')}), flush=True)
        return control

    def destroy(self, results=None):
        super().destroy(results)
        self.event('agent_destroy_finished')


class FullDiffusion2FastAgent(FullDiffusionTiming, FullDiffusion2GeoAgent):
    pass
