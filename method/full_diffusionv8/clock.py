"""Every 0.05s simulation tick gets a fresh plan, independent of wall latency."""
import math


class PlanningClock:
    tick_seconds = .05
    period_ticks = 1

    def __init__(self):
        self.anchor_tick = None

    @classmethod
    def tick(cls, timestamp):
        if not math.isfinite(timestamp):
            raise ValueError('A finite simulation timestamp is required')
        tick = round(timestamp / cls.tick_seconds)
        if abs(tick * cls.tick_seconds - timestamp) > 1e-4:
            raise ValueError('This closed-loop evaluation requires CARLA 20Hz simulation ticks')
        return tick

    def due(self, timestamp):
        tick = self.tick(timestamp)
        return self.anchor_tick is None or tick - self.anchor_tick >= self.period_ticks

    def mark(self, timestamp):
        self.anchor_tick = self.tick(timestamp)

    def age(self, timestamp):
        if self.anchor_tick is None:
            raise RuntimeError('No planning anchor has been generated')
        return (self.tick(timestamp) - self.anchor_tick) * self.tick_seconds

    def reset(self):
        self.anchor_tick = None
