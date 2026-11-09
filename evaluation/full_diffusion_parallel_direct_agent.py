"""Original GPU agent; capture one policy call for RPC equivalence verification."""
import os
from pathlib import Path

import torch

from evaluation.full_diffusion_fast_agent import FullDiffusionFastAgent
from evaluation.shared_gpu_inference import pack


def get_entry_point():
    return 'FullDiffusionParallelDirectAgent'


class FullDiffusionParallelDirectAgent(FullDiffusionFastAgent):
    def setup(self, *args, **kwargs):
        super().setup(*args, **kwargs)
        forward = self.nets[0].forward
        captured = False

        def traced_forward(*forward_args, **inputs):
            nonlocal captured
            saved = pack(inputs) if not captured else None
            output = forward(*forward_args, **inputs)
            if not captured:
                torch.save({'inputs': saved, 'output': pack(output),
                            'last_target_speed': self.nets[0].last_target_speed},
                           Path(os.environ['WMMPC_CARLA_LOG_DIR']) / 'first_policy_call.pt')
                captured = True
            return output

        self.nets[0].forward = traced_forward
