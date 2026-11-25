"""Exact simulation-tick histories; no nearest-frame substitution."""
from collections import OrderedDict

import numpy as np
import torch

from method.full_diffusion2.data import frame_pose
from .rolling import BatchedRollingPlanStore
from .clock import PlanningClock


class TickBEVHistory:
    def __init__(self, capacity=32):
        if capacity < 11:
            raise ValueError('BEV history must cover ten previous simulation ticks')
        self.capacity = capacity
        self.entries = OrderedDict()
        self.pose = self.timestamp = self.tick = None
        self.last_ticks = [None, None]

    def set_observation(self, pose, timestamp):
        tick = PlanningClock.tick(float(timestamp))
        if self.entries and tick <= next(reversed(self.entries)):
            self.entries.clear()
        self.pose = frame_pose({'theta': float(pose[2]), 'pos_global': pose[:2]})
        if not np.isfinite(self.pose).all():
            raise ValueError('A finite observed ego pose is required')
        self.tick, self.timestamp = tick, tick * PlanningClock.tick_seconds

    def get_inputs(self, current):
        if self.pose is None or current.shape[0] != 1:
            raise RuntimeError('Tick BEV history requires an observation and batch1')
        features, transforms, valid, selected_ticks = [], [], [], []
        for lag in (10, 5):
            target = self.tick - lag
            selected = self.entries.get(target)
            exists = selected is not None
            features.append(selected[0] if exists else current)
            transforms.append(np.linalg.solve(self.pose, selected[1]) if exists else np.eye(3))
            valid.append(exists)
            selected_ticks.append(target if exists else None)
        self.entries[self.tick] = (current.detach().clone(), self.pose.copy())
        while len(self.entries) > self.capacity:
            self.entries.popitem(last=False)
        self.last_ticks = selected_ticks
        return (torch.stack(features, dim=1),
                torch.as_tensor(np.stack(transforms), device=current.device, dtype=torch.float32)[None],
                torch.tensor([valid], device=current.device, dtype=torch.bool))


class LaggedPlanHistory:
    """Read exactly k-lag_ticks, using its timestamp and observed ego pose."""
    def __init__(self, capacity=32, *, lag_ticks=5):
        if not isinstance(lag_ticks, int) or not 0 < lag_ticks < capacity:
            raise ValueError('Plan history capacity must exceed the positive tick lag')
        self.lag_ticks = lag_ticks
        self.capacity = capacity
        self.entries = {}
        self.last_selected_ticks = []
        self._aligner = BatchedRollingPlanStore(max_gap=lag_ticks * PlanningClock.tick_seconds)

    def reset(self, scene=None):
        if scene is None:
            self.entries.clear()
        else:
            self.entries.pop(str(scene), None)
        self.last_selected_ticks = []
        self._aligner.reset()

    @staticmethod
    def _ticks(scenes, timestamps):
        stamps = timestamps.detach().cpu().tolist() if torch.is_tensor(timestamps) else list(timestamps)
        if len(stamps) != len(scenes) or len(set(map(str, scenes))) != len(scenes):
            raise ValueError('Each batch item requires its own scene and timestamp')
        return [PlanningClock.tick(float(stamp)) for stamp in stamps]

    def prior(self, scenes, poses, timestamps):
        ticks = self._ticks(scenes, timestamps)
        selected, selected_ticks = {}, []
        for scene, tick in zip(scenes, ticks):
            scene = str(scene)
            target = tick - self.lag_ticks
            previous = self.entries.get(scene, {}).get(target)
            if previous is not None:
                selected[scene] = previous
            selected_ticks.append(target if previous is not None else None)
        # The alignment store uses the selected plan timestamp and both observed poses.
        # A missing tick never falls back to another plan.
        self._aligner.entries = selected
        self.last_selected_ticks = selected_ticks
        return self._aligner.prior(scenes, poses, [tick * .05 for tick in ticks])

    def update(self, scenes, poses, timestamps, delta):
        ticks = self._ticks(scenes, timestamps)
        if not bool(torch.isfinite(poses).all()):
            raise ValueError('Only finite poses can enter rolling state')
        self._aligner.entries = {}
        self._aligner.update(scenes, poses, [tick * .05 for tick in ticks], delta)
        for scene, tick in zip(scenes, ticks):
            scene = str(scene)
            history = self.entries.setdefault(scene, OrderedDict())
            if history and tick <= next(reversed(history)):
                history.clear()
            history[tick] = self._aligner.entries[scene]
            while len(history) > self.capacity:
                history.popitem(last=False)
