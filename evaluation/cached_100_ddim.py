"""Cache DDIM constants, retaining 100 denoiser calls and the original equations."""
import torch
from diffusers import DDIMScheduler


class Cached100DDIM:
    def __init__(self, original):
        self.scheduler = original.scheduler
        self.codec = original.codec
        self.train_steps = original.train_steps
        self._plans = {}
        self.last_denoiser_calls = 0

    def _plan(self, batch_size, device):
        key = (batch_size, str(device))
        if key not in self._plans:
            scheduler = DDIMScheduler.from_config(self.scheduler.config)
            if (scheduler.config.prediction_type != 'sample' or scheduler.config.clip_sample
                    or scheduler.config.thresholding):
                raise ValueError('Cached sampler only supports the unchanged Full Diffusion configuration')
            scheduler.set_timesteps(100)
            rows = []
            times = []
            for timestep in scheduler.timesteps.tolist():
                previous = timestep - scheduler.config.num_train_timesteps // 100
                alpha = scheduler.alphas_cumprod[timestep]
                alpha_previous = (scheduler.alphas_cumprod[previous] if previous >= 0
                                  else scheduler.final_alpha_cumprod)
                beta = 1 - alpha
                # Retain the upstream eta=0 arithmetic when forming constants.
                variance = scheduler._get_variance(timestep, previous)
                std = 0.0 * variance ** 0.5
                # Keep zero-dimensional CPU scalars, as in the original DDIM
                # kernel calls. GPU tensor division has different rounding
                # from PyTorch's CPU-scalar fast path and can accumulate drift.
                rows.append((alpha ** 0.5, beta ** 0.5, alpha_previous ** 0.5,
                             (1 - alpha_previous - std ** 2) ** 0.5))
                time_tensor = torch.tensor(timestep, device=device)
                times.append((time_tensor.float().expand(batch_size) + 1) / self.train_steps)
            self._plans[key] = tuple((times[i], *rows[i]) for i in range(100))
        return self._plans[key]

    @torch.no_grad()
    def sample(self, predict_clean, batch_size, device, generator=None, inference_steps=100):
        if inference_steps != 100:
            raise ValueError('This experiment requires exactly 100 diffusion steps')
        plan = self._plan(batch_size, device)
        sample = torch.randn((batch_size, self.codec.path_points + 1, 2),
                             device=device, generator=generator)
        sample[:, -1, 1] = 0
        self.last_denoiser_calls = 0
        for normalized_time, sqrt_alpha, sqrt_beta, sqrt_previous, direction_scale in plan:
            prediction = predict_clean(sample, normalized_time).float()
            self.last_denoiser_calls += 1
            epsilon = (sample - sqrt_alpha * prediction) / sqrt_beta
            sample = sqrt_previous * prediction + direction_scale * epsilon
            sample[:, -1, 1] = 0
        if self.last_denoiser_calls != 100:
            raise RuntimeError('Incorrect number of denoising iterations')
        return sample
