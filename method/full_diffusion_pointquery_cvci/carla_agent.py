"""Garage sensor-agent entry point for the V8 ego12 planner."""
from unittest.mock import patch

import method.full_diffusionv3_ego12.inference.carla_agent as base
from .policy import V8Network


def get_entry_point():
    return 'V8Agent'


class V8Agent(base.Ego12Agent):
    def setup(self, path_to_conf_file, route_index=None, traffic_manager=None):
        with patch.object(base, 'Ego12Network', V8Network):
            super().setup(path_to_conf_file, route_index, traffic_manager)
