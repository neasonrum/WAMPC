"""VP diffusion of ten spatial positions and six temporal XY increments."""

import math

import torch
from diffusers import DDIMScheduler

from method.full_diffusion.diffusion import VPFullDiffusion


class SpatialTemporalCodec(torch.nn.Module):
    def __init__(self, path_points=10, temporal_points=6,
                 temporal_interval_seconds=0.5, mean=None, std=None):
        super().__init__()
        if (path_points, temporal_points) != (10, 6):
            raise ValueError("V3 requires ten spatial points and six temporal increments")
        if not math.isfinite(temporal_interval_seconds) or temporal_interval_seconds <= 0:
            raise ValueError("The temporal interval must be finite and positive")
        self.path_points = path_points
        self.temporal_points = temporal_points
        self.total_points = path_points + temporal_points
        self.temporal_interval_seconds = temporal_interval_seconds
        mean = torch.zeros(self.total_points, 2) if mean is None else torch.tensor(mean, dtype=torch.float32)
        default_std = [[20.0, 20.0]] * path_points + [[5.0, 5.0]] * temporal_points
        std = torch.tensor(default_std if std is None else std, dtype=torch.float32)
        if (mean.shape != (self.total_points, 2) or std.shape != mean.shape
                or not torch.isfinite(mean).all() or not torch.isfinite(std).all()
                or not torch.all(std > 0)):
            raise ValueError("V3 normalization must contain finite 16 x 2 means and positive scales")
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def check_shape(self, value):
        if value.ndim != 3 or value.shape[1:] != (self.total_points, 2):
            raise ValueError("Expected a B x 16 x 2 V3 diffusion state")

    def encode(self, spatial_path, temporal_delta):
        if (spatial_path.ndim != 3 or spatial_path.shape[1:] != (self.path_points, 2)
                or temporal_delta.shape != (len(spatial_path), self.temporal_points, 2)):
            raise ValueError("Expected B x 10 x 2 positions and B x 6 x 2 increments")
        return self.encode_metric(torch.cat((spatial_path, temporal_delta), dim=1))

    def encode_metric(self, metric):
        self.check_shape(metric)
        return (metric - self.mean.to(metric)) / self.std.to(metric)

    def to_metric(self, normalized):
        self.check_shape(normalized)
        return normalized * self.std.to(normalized) + self.mean.to(normalized)

    def decode(self, normalized):
        metric = self.to_metric(normalized)
        return metric[:, :self.path_points], metric[:, self.path_points:]

    def target_speed(self, temporal_delta):
        if temporal_delta.ndim != 3 or temporal_delta.shape[1:] != (self.temporal_points, 2):
            raise ValueError("Expected six temporal XY increments")
        return temporal_delta[:, 1].float().norm(dim=-1) / self.temporal_interval_seconds

    @staticmethod
    def masked_l1(prediction, target, valid, weights=None):
        if prediction.shape != target.shape or valid.shape != prediction.shape[:-1]:
            raise ValueError("Coordinate targets and validity masks do not match")
        valid = valid.to(device=prediction.device, dtype=torch.bool)
        errors = torch.where(valid[..., None], (prediction.float() - target.float()).abs(), 0.0)
        factors = valid.to(errors.dtype)
        if weights is not None:
            weights = torch.as_tensor(weights, device=prediction.device, dtype=errors.dtype)
            if weights.shape != (prediction.shape[1],):
                raise ValueError("Loss weights must match the number of slots")
            factors = factors * weights[None]
        denominator = (factors.sum() * prediction.shape[-1]).clamp_min(1.0)
        return (errors * factors[..., None]).sum() / denominator


class SpatialTemporalVPFullDiffusion(VPFullDiffusion):
    def __init__(self, codec=None, train_steps=1000, beta_min=0.1, beta_d=19.9):
        super().__init__(codec=codec if codec is not None else SpatialTemporalCodec(),
                         train_steps=train_steps, beta_min=beta_min, beta_d=beta_d)

    def corrupt(self, clean, timesteps=None, noise=None):
        self.codec.check_shape(clean)
        if timesteps is None:
            timesteps = torch.randint(self.train_steps, (len(clean),), device=clean.device)
        if noise is None:
            noise = torch.randn_like(clean)
        if noise.shape != clean.shape:
            raise ValueError("Noise must cover all 32 V3 coordinates")
        noisy = self.scheduler.add_noise(clean, noise, timesteps)
        return noisy, (timesteps.float() + 1) / self.train_steps

    @torch.no_grad()
    def sample(self, predict_clean, batch_size, device, generator=None, inference_steps=100):
        scheduler = DDIMScheduler.from_config(self.scheduler.config)
        scheduler.set_timesteps(inference_steps, device=device)
        sample = torch.randn((batch_size, self.codec.total_points, 2),
                             device=device, generator=generator)
        for timestep in scheduler.timesteps:
            normalized_time = (timestep.float().expand(batch_size) + 1) / self.train_steps
            prediction = predict_clean(sample, normalized_time).float()
            self.codec.check_shape(prediction)
            sample = scheduler.step(prediction, timestep, sample, eta=0.0).prev_sample
        return sample