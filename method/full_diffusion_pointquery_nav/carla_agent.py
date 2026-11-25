"""Reuse the established sensor setup and controller, replacing only the planner."""
from unittest.mock import patch

import method.full_diffusionv3_ego12.inference.carla_agent as base
from .policy import PointQueryNetwork


def get_entry_point():
    return "PointQueryAgent"


class PointQueryAgent(base.Ego12Agent):
    def setup(self, path_to_conf_file, route_index=None, traffic_manager=None):
        with patch.object(base, "Ego12Network", PointQueryNetwork):
            super().setup(path_to_conf_file, route_index, traffic_manager)

