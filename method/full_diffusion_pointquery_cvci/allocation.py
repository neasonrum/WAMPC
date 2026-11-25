"""Use only the four already assigned, idle A100 allocations."""
import time
from .utils import inspect_allocation
from method.full_diffusionv3_ego12.allocation import query_allocation


def inspect(job, config, require_idle=True):
    verified = inspect_allocation(config, job, idle=require_idle)
    scheduler = query_allocation(job, config['jobs'][job]['host'])
    return {**verified, **scheduler, 'gpu_uuids': config['jobs'][job]['gpu_uuids'],
            'checked_unix': time.time()}
