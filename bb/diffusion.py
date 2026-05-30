from __future__ import annotations

import math
from collections.abc import Iterable

import torch
from torch import nn
from torch.nn import functional as F
from tqdm.auto import tqdm


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(0.0001, 0.9999)


def linear_beta_schedule(timesteps: int) -> torch.Tensor:
    scale = 1000 / timesteps
    return torch.linspace(scale * 0.0001, scale * 0.02, timesteps).clamp(0.0001, 0.9999)


def extract(values: torch.Tensor, timesteps: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    out = values.gather(0, timesteps)
    return out.reshape(timesteps.shape[0], *((1,) * (len(shape) - 1)))


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        timesteps: int = 1000,
        beta_schedule: str = "cosine",
        prediction_type: str = "epsilon",
    ) -> None:
        super().__init__()
        if prediction_type != "epsilon":
            raise ValueError("Only epsilon prediction is implemented.")
        if beta_schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        elif beta_schedule == "linear":
            betas = linear_beta_schedule(timesteps)
        else:
            raise ValueError(f"Unknown beta_schedule: {beta_schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]], dim=0)

        self.timesteps = timesteps
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod",
            torch.sqrt(1.0 / alphas_cumprod - 1),
        )
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer(
            "posterior_log_variance",
            torch.log(posterior_variance.clamp(min=1e-20)),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

    def q_sample(
        self,
        x_start: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)
        return (
            extract(self.sqrt_alphas_cumprod, timesteps, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, timesteps, x_start.shape) * noise
        )

    def training_loss(
        self,
        model: nn.Module,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        timesteps = torch.randint(0, self.timesteps, (images.shape[0],), device=images.device)
        noise = torch.randn_like(images)
        noisy = self.q_sample(images, timesteps, noise)
        pred_noise = model(noisy, timesteps, labels)
        return F.mse_loss(pred_noise.float(), noise.float())

    def predict_xstart_from_eps(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        eps: torch.Tensor,
    ) -> torch.Tensor:
        return (
            extract(self.sqrt_recip_alphas_cumprod, timesteps, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, timesteps, x_t.shape) * eps
        )

    def p_mean_variance(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        labels: torch.Tensor,
        cfg_scale: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if cfg_scale != 1.0 and hasattr(model, "forward_with_cfg"):
            eps = model.forward_with_cfg(x_t, timesteps, labels, cfg_scale)
        else:
            eps = model(x_t, timesteps, labels)
        x_start = self.predict_xstart_from_eps(x_t, timesteps, eps).clamp(-1, 1)
        mean = (
            extract(self.posterior_mean_coef1, timesteps, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, timesteps, x_t.shape) * x_t
        )
        log_variance = extract(self.posterior_log_variance, timesteps, x_t.shape)
        return mean, log_variance

    @torch.no_grad()
    def p_sample(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        labels: torch.Tensor,
        cfg_scale: float = 1.0,
    ) -> torch.Tensor:
        mean, log_variance = self.p_mean_variance(model, x_t, timesteps, labels, cfg_scale)
        noise = torch.randn_like(x_t)
        nonzero_mask = (timesteps != 0).float().reshape(x_t.shape[0], *((1,) * (x_t.ndim - 1)))
        return mean + nonzero_mask * torch.exp(0.5 * log_variance) * noise

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        shape: tuple[int, int, int, int],
        labels: torch.Tensor,
        device: torch.device,
        cfg_scale: float = 1.0,
        progress: bool = True,
    ) -> torch.Tensor:
        x_t = torch.randn(shape, device=device)
        steps: Iterable[int] = range(self.timesteps - 1, -1, -1)
        if progress:
            steps = tqdm(steps, total=self.timesteps, desc="sample", leave=False)
        for step in steps:
            t = torch.full((shape[0],), step, device=device, dtype=torch.long)
            x_t = self.p_sample(model, x_t, t, labels, cfg_scale=cfg_scale)
        return x_t

    @torch.no_grad()
    def ddim_sample(
        self,
        model: nn.Module,
        shape: tuple[int, int, int, int],
        labels: torch.Tensor,
        device: torch.device,
        steps: int = 50,
        cfg_scale: float = 1.0,
        progress: bool = True,
    ) -> torch.Tensor:
        x_t = torch.randn(shape, device=device)
        sequence = torch.linspace(self.timesteps - 1, 0, steps, device=device).long()
        iterator: Iterable[torch.Tensor] = sequence
        if progress:
            iterator = tqdm(sequence, desc="ddim", leave=False)
        for i, timestep in enumerate(iterator):
            t = timestep.expand(shape[0])
            if cfg_scale != 1.0 and hasattr(model, "forward_with_cfg"):
                eps = model.forward_with_cfg(x_t, t, labels, cfg_scale)
            else:
                eps = model(x_t, t, labels)
            alpha_t = extract(self.alphas_cumprod, t, x_t.shape)
            if i == len(sequence) - 1:
                alpha_prev = torch.ones_like(alpha_t)
            else:
                prev_t = sequence[i + 1].expand(shape[0])
                alpha_prev = extract(self.alphas_cumprod, prev_t, x_t.shape)
            pred_x0 = ((x_t - torch.sqrt(1 - alpha_t) * eps) / torch.sqrt(alpha_t)).clamp(-1, 1)
            x_t = torch.sqrt(alpha_prev) * pred_x0 + torch.sqrt(1 - alpha_prev) * eps
        return x_t
