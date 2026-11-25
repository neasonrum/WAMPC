"""Load only the existing tensor-only timed-world implementation."""
import importlib
from pathlib import Path
import sys
import types


def module(suffix):
    name = 'method.full_diffusionv8._timed_source'
    if name not in sys.modules:
        package = types.ModuleType(name)
        package.__path__ = [str(Path(__file__).resolve().parents[1] / 'irwm_bev_t')]
        package.__package__ = name
        sys.modules[name] = package
    return importlib.import_module(name + '.' + suffix)
