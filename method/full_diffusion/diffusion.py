"""Anchor-free VP diffusion of geometric path points and a speed scalar."""

import torch
import torch.nn.functional as functional
from diffusers import DDIMScheduler


class GeoCodec(torch.nn.Module):
    def __init__(self, path_points=10, path_scale=20.0, speed_scale=20.0, mean=None, std=None):
        super().__init__()
        if path_points < 1 or path_scale <= 0 or speed_scale <= 0:
            raise ValueError("Invalid geometric target dimensions or scales")
        self.path_points = path_points
        self.path_scale = path_scale
        self.speed_scale = speed_scale
        default_std = [[path_scale, path_scale]] * path_points + [[speed_scale, 1.0]]
        mean = torch.zeros(path_points + 1, 2) if mean is None else torch.tensor(mean, dtype=torch.float32)
        std = torch.tensor(default_std if std is None else std, dtype=torch.float32)
        if mean.shape != (path_points + 1, 2) or std.shape != mean.shape or not torch.all(std > 0):
            raise ValueError("Invalid target normalization statistics")
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def encode(self, path, speed):
        if path.ndim != 3 or path.shape[1:] != (self.path_points, 2):
            raise ValueError(f"Expected a B x {self.path_points} x 2 geometric path")
        speed = speed.reshape(-1)
        if speed.shape[0] != path.shape[0]:
            raise ValueError("Path and speed batch sizes differ")
        speed_token = torch.stack((speed, torch.zeros_like(speed)), dim=-1)
        metric = torch.cat((path, speed_token[:, None]), dim=1)
        return (metric - self.mean.to(metric)) / self.std.to(metric)

    def decode(self, sample):
        metric = sample * self.std.to(sample) + self.mean.to(sample)
        return metric[:, :self.path_points], metric[:, -1, 0]

    @staticmethod
    def loss(prediction, clean):
        errors = functional.l1_loss(prediction.float(), clean.float(), reduction="none")
        return (errors[:, :-1].sum() + errors[:, -1, 0].sum()) / (clean.shape[0] * (clean.shape[1] * 2 - 1))


class VPFullDiffusion:
    def __init__(self, codec=None, train_steps=1000, beta_min=0.1, beta_d=19.9):
        self.codec = codec or GeoCodec()
        self.train_steps = train_steps
        boundaries = torch.linspace(0, 1, train_steps + 1, dtype=torch.float64)
        integrated_beta = beta_min * boundaries + 0.5 * beta_d * boundaries.square()
        betas = -torch.expm1(-(integrated_beta[1:] - integrated_beta[:-1]))
        self.scheduler = DDIMScheduler(
            num_train_timesteps=train_steps,
            trained_betas=betas.numpy(),
            prediction_type="sample",
            clip_sample=False,
            set_alpha_to_one=True,
            timestep_spacing="trailing",
        )

    def corrupt(self, clean, timesteps=None, noise=None):
        if timesteps is None:
            timesteps = torch.randint(self.train_steps, (clean.shape[0],), device=clean.device)
        if noise is None:
            noise = torch.randn_like(clean)
        noisy = self.scheduler.add_noise(clean, noise, timesteps)
        noisy = noisy.clone()
        noisy[:, -1, 1] = 0
        return noisy, (timesteps.float() + 1) / self.train_steps

    @torch.no_grad()
    def sample(self, predict_clean, batch_size, device, generator=None, inference_steps=100):
        scheduler = DDIMScheduler.from_config(self.scheduler.config)
        scheduler.set_timesteps(inference_steps, device=device)
        sample = torch.randn(
            (batch_size, self.codec.path_points + 1, 2), device=device, generator=generator
        )
        sample[:, -1, 1] = 0
        for timestep in scheduler.timesteps:
            normalized_time = (timestep.float().expand(batch_size) + 1) / self.train_steps
            prediction = predict_clean(sample, normalized_time).float()
            sample = scheduler.step(prediction, timestep, sample, eta=0.0).prev_sample
            sample[:, -1, 1] = 0
        return sample