import math
from dataclasses import dataclass
import torch
from .geometry import FUTURE_TIMES, cumulative_valid

@dataclass
class Plan:
    delta: torch.Tensor
    pose: torch.Tensor
    timestamp: float



def advance_plan(delta, source_pose, current_pose, elapsed):
    """Interpolate only inside the old 3s plan; never invent a missing tail."""
    if delta.shape != (6, 2) or source_pose.shape != (3, 3) or current_pose.shape != (3, 3):
        raise ValueError('Expected a six-interval plan and two SE(2) transforms')
    if not 0 <= elapsed < 3 or not bool(torch.isfinite(delta).all()):
        return torch.zeros_like(delta), torch.zeros(6, device=delta.device, dtype=torch.bool)
    with torch.autocast(device_type=delta.device.type, enabled=False):
        times = torch.tensor(FUTURE_TIMES, device=delta.device, dtype=torch.float64)
        query = times + elapsed
        valid = query <= 3.0 + 1e-8
        old = torch.cat((delta.new_zeros(1, 2), delta.float().cumsum(0)), 0).double()
        location = (query / .5).clamp(0, 6)
        left = location.floor().long().clamp(max=5)
        right = left + 1
        weight = location - left
        xy = old[left] * (1 - weight[:, None]) + old[right] * weight[:, None]
        transform = torch.linalg.solve(current_pose.double(), source_pose.double())
        xy = (torch.cat((xy, torch.ones_like(xy[:, :1])), -1) @ transform.T)[:, :2]
        increments = torch.cat((xy[:1], xy[1:] - xy[:-1]), 0)
        increments = torch.where(valid[:, None], increments, 0.)
    return increments.float(), cumulative_valid(valid)



class RollingPlanStore:
    """One state per scene/route, safe for batch reordering and route boundaries."""
    def __init__(self, max_gap=0.5):
        self.max_gap = float(max_gap)
        self.entries = {}

    def reset(self, scene=None):
        if scene is None:
            self.entries.clear()
        else:
            self.entries.pop(str(scene), None)

    def prior(self, scenes, poses, timestamps):
        b, device = len(scenes), poses.device
        deltas = torch.zeros(b, 6, 2, device=device)
        valid = torch.zeros(b, 6, dtype=torch.bool, device=device)
        ages = torch.zeros(b, device=device)
        for i, scene in enumerate(scenes):
            old = self.entries.get(str(scene))
            if old is None:
                continue
            elapsed = float(timestamps[i]) - old.timestamp
            if elapsed <= 0 or elapsed > self.max_gap + 1e-8:
                self.reset(scene)
                continue
            deltas[i], valid[i] = advance_plan(old.delta.to(device), old.pose.to(device), poses[i], elapsed)
            ages[i] = elapsed
        return deltas, valid, ages

    def update(self, scenes, poses, timestamps, delta):
        if delta.shape != (len(scenes), 6, 2) or not bool(torch.isfinite(delta).all()):
            raise ValueError('Only finite decoded metric temporal predictions can enter rolling state')
        for i, scene in enumerate(scenes):
            self.entries[str(scene)] = Plan(delta[i].detach().clone(), poses[i].detach().double().clone(),
                                             float(timestamps[i]))

    def state_dict(self):
        return {key: {'delta': value.delta.cpu(), 'pose': value.pose.cpu(), 'timestamp': value.timestamp}
                for key, value in self.entries.items()}

    def load_state_dict(self, state):
        self.entries = {key: Plan(**value) for key, value in state.items()}


class BatchedRollingPlanStore(RollingPlanStore):
    def prior(self, scenes, poses, timestamps):
        stamps = timestamps.detach().cpu().tolist() if torch.is_tensor(timestamps) else list(timestamps)
        b, device = len(scenes), poses.device
        delta = torch.zeros(b, 6, 2, device=device)
        valid = torch.zeros(b, 6, dtype=torch.bool, device=device)
        ages = torch.zeros(b, device=device)
        selected, old, elapsed = [], [], []
        for i, scene in enumerate(scenes):
            previous = self.entries.get(str(scene))
            if previous is None:
                continue
            age = float(stamps[i]) - previous.timestamp
            if age <= 0 or age > self.max_gap + 1e-8:
                self.reset(scene)
            else:
                selected.append(i); old.append(previous); elapsed.append(age)
        if not selected:
            return delta, valid, ages
        index = torch.tensor(selected, device=device)
        with torch.autocast(device_type=device.type, enabled=False):
            previous_delta = torch.stack([p.delta.to(device) for p in old]).double()
            previous_pose = torch.stack([p.pose.to(device) for p in old]).double()
            age = torch.tensor(elapsed, device=device, dtype=torch.float64)
            query = torch.tensor(FUTURE_TIMES, device=device, dtype=torch.float64)[None] + age[:, None]
            available = query <= 3. + 1e-8
            positions = torch.cat((previous_delta.new_zeros(len(old), 1, 2), previous_delta.cumsum(1)), 1)
            location = (query / .5).clamp(0, 6)
            left = location.floor().long().clamp(max=5)
            right, weight = left + 1, location - left
            points = (positions.gather(1, left[..., None].expand(-1, -1, 2)) * (1 - weight[..., None])
                      + positions.gather(1, right[..., None].expand(-1, -1, 2)) * weight[..., None])
            transform = torch.linalg.solve(poses[index].double(), previous_pose)
            points = torch.einsum('bij,bkj->bki', transform,
                                  torch.cat((points, torch.ones_like(points[..., :1])), -1))[..., :2]
            increments = torch.cat((points[:, :1], points[:, 1:] - points[:, :-1]), 1)
            increments = torch.where(available[..., None], increments, 0.)
            delta[index] = increments.float()
            valid[index] = cumulative_valid(available)
            ages[index] = age.float()
        return delta, valid, ages

    def update(self, scenes, poses, timestamps, delta):
        if delta.shape != (len(scenes), 6, 2) or not bool(torch.isfinite(delta).all()):
            raise ValueError('Only finite six-interval predictions can enter rolling state')
        stamps = timestamps.detach().cpu().tolist() if torch.is_tensor(timestamps) else list(timestamps)
        saved_delta, saved_pose = delta.detach().float().clone(), poses.detach().double().clone()
        for i, scene in enumerate(scenes):
            self.entries[str(scene)] = Plan(saved_delta[i], saved_pose[i], float(stamps[i]))


class QuarterSecondPlanStore(BatchedRollingPlanStore):
    def __init__(self):
        super().__init__(max_gap=.25)

    def prior(self, scenes, poses, timestamps):
        if len(set(map(str, scenes))) != len(scenes):
            raise ValueError('A batch must contain distinct routes to preserve causality')
        stamps = timestamps.detach().cpu().tolist() if torch.is_tensor(timestamps) else list(timestamps)
        for scene, now in zip(scenes, stamps):
            previous = self.entries.get(str(scene))
            if previous is not None and not math.isclose(float(now) - previous.timestamp, .25, abs_tol=1e-6, rel_tol=0):
                # A gap, rewind, new route cycle or mismatched cadence is a cold start.
                self.reset(scene)
        return super().prior(scenes, poses, stamps)

