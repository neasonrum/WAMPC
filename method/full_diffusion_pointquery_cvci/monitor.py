"""Only periodic NVML samples tolerate failure; cgroup RAM/CPU reads stay live."""
import subprocess
import time

QUERY = ('nvidia-smi', '--query-gpu=index,uuid,memory.used,utilization.gpu', '--format=csv,noheader')
STATE = {'last_success_unix': None, 'stdout': '', 'last_error': None, 'fallbacks': 0}


def install():
    import evaluation.carla_eval_resources as module
    if getattr(module, '_pointquery_cvci_monitor', False):
        return
    original_run, original_resources = subprocess.run, module.resources

    def bounded_run(command, *args, **kwargs):
        if not isinstance(command, (list, tuple)) or tuple(map(str, command)) != QUERY:
            return original_run(command, *args, **kwargs)
        kwargs['timeout'] = min(float(kwargs.get('timeout', 15)), 3.)
        try:
            result = original_run(command, *args, **kwargs)
            if result.returncode or not result.stdout.strip():
                raise RuntimeError(f'NVML monitor return code {result.returncode}')
            STATE.update(last_success_unix=time.time(), stdout=result.stdout, last_error=None)
            return result
        except (subprocess.TimeoutExpired, OSError, RuntimeError) as error:
            STATE['last_error'] = repr(error)
            STATE['fallbacks'] += 1
            timestamp = STATE['last_success_unix']
            fresh_enough = timestamp is not None and time.time() - timestamp <= 60
            return subprocess.CompletedProcess(command, 0, stdout=STATE['stdout'] if fresh_enough else '', stderr=repr(error))

    def live_resources():
        result = original_resources()
        result.update(gpu_monitor_fallback=STATE['last_error'] is not None,
                      gpu_monitor_error=STATE['last_error'], gpu_monitor_fallback_count=STATE['fallbacks'],
                      gpu_monitor_last_success_unix=STATE['last_success_unix'])
        return result

    subprocess.run, module.resources = bounded_run, live_resources
    module._pointquery_cvci_monitor = True
