"""Flow math of the Wan Ref2VA stages in Wan's noise-ward convention.

``sigma = 1`` is noise, ``x_sigma = (1 - sigma) x0 + sigma eps``, the model predicts
``v = eps - x0`` from timestep ``1000 * sigma``, and ``x0 = x - sigma v``. H3's data-ward
``t`` is ``1 - sigma`` and its velocity has the opposite sign, so every schedule here is the
H3 one mapped through that change of variables.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from solarwm.training.anyflow import (
    adaptive_rescale_non_diffusion_losses,
    apply_timestep_shift,
    bounded_difference_timesteps,
    central_difference_derivative,
    gaussian_timestep_weights,
)

# H3's keyframe time 0.999 (data-ward): references and given latents sit just off clean.
KEYFRAME_SIGMA = 0.001
NUM_TRAIN_TIMESTEPS = 1000


def shifted_sigmas(num_points: int, shift: float, *, device: Any = None) -> torch.Tensor:
    """H3's ``make_shifted_schedule``: ``num_points`` grid points ending at zero, deduplicated."""

    if int(num_points) < 2:
        raise ValueError("a shifted schedule needs at least two grid points")
    base = torch.linspace(1.0, 0.0, int(num_points), dtype=torch.float32)
    return torch.unique_consecutive(apply_timestep_shift(base, float(shift))).to(device=device)


def sample_training_sigma(generator: Any, *, shift: float, device: Any) -> torch.Tensor:
    """One shifted training noise level (H3 ``shifted_timestep`` as ``1 - t``)."""

    uniform = torch.rand((), generator=generator, device=device, dtype=torch.float32)
    return apply_timestep_shift(uniform, float(shift))


def sgf_score_sigma(
    *, shift: float, min_sigma: float, max_sigma: float, generator: Any = None, device: Any = None
) -> torch.Tensor:
    """H3 SGF score level: an integer raw time in ``[0, 1000)``, shifted and clamped."""

    if not 0.0 < float(min_sigma) < float(max_sigma) < 1.0:
        raise ValueError("SGF score sigma bounds must lie strictly within (0, 1)")
    raw = torch.randint(0, NUM_TRAIN_TIMESTEPS, (), device=device, generator=generator).float()
    return apply_timestep_shift(raw / NUM_TRAIN_TIMESTEPS, float(shift)).clamp(
        float(min_sigma), float(max_sigma)
    )


def add_noise(clean: torch.Tensor, noise: torch.Tensor, sigma: Any) -> torch.Tensor:
    sigma = torch.as_tensor(sigma, device=clean.device, dtype=torch.float32)
    return (1.0 - sigma) * clean.float() + sigma * noise.float()


def keyframe(clean: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    """H3 ``keyframe_noise_augmentation``: ``aug * clean + (1 - aug) * noise``."""

    return add_noise(clean, noise, KEYFRAME_SIGMA)


def predict_x0(noisy: torch.Tensor, velocity: torch.Tensor, sigma: Any) -> torch.Tensor:
    sigma = torch.as_tensor(sigma, device=noisy.device, dtype=torch.float32)
    return noisy.float() - sigma * velocity.float()


def wan_anyflow_v15_loss(
    model_velocity: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    clean: torch.Tensor,
    noise: torch.Tensor,
    sigma_t: torch.Tensor,
    sigma_r: torch.Tensor,
    is_diffusion: torch.Tensor,
    *,
    epsilon: float,
    shift: float,
    process_group: Any = None,
) -> torch.Tensor:
    """TF AnyFlow v1.5 (H3 ``h3_anyflow_v15_loss`` without its sign flip).

    ``model_velocity(sample, raw_t, raw_r)`` returns Wan's ``eps - x0`` for raw times in
    ``[0, 1000]``. The three detached evaluations run before the graph-carrying one, so every
    rank executes the same four forwards.
    """

    if clean.shape != noise.shape or clean.shape[0] != sigma_t.numel():
        raise ValueError("Wan AnyFlow clean/noise/time batch shapes disagree")
    n = NUM_TRAIN_TIMESTEPS
    shape = (clean.shape[0],) + (1,) * (clean.ndim - 1)
    raw_t = sigma_t.float() * n
    raw_r = sigma_r.float() * n
    noisy = (1 - sigma_t.view(shape)) * clean + sigma_t.view(shape) * noise
    with torch.no_grad():
        diagonal = model_velocity(noisy, raw_t, raw_t).float()
        plus_t, minus_t = bounded_difference_timesteps(
            raw_t, raw_r, epsilon=epsilon, num_train_timesteps=n
        )
        plus = noisy + diagonal * ((plus_t - raw_t) / n).view(shape)
        minus = noisy - diagonal * ((raw_t - minus_t) / n).view(shape)
        plus_v = model_velocity(plus, plus_t, raw_r).float()
        minus_v = model_velocity(minus, minus_t, raw_r).float()
        derivative = central_difference_derivative(plus_v, minus_v, plus_t, minus_t)
    del diagonal, plus, minus, plus_v, minus_v
    prediction = model_velocity(noisy, raw_t, raw_r).float()
    residual = prediction + (raw_t - raw_r).view(shape) * derivative - (noise - clean)
    per_sample = residual.square().flatten(1).mean(1)
    per_sample = adaptive_rescale_non_diffusion_losses(
        per_sample, is_diffusion, process_group=process_group
    )
    weights = gaussian_timestep_weights(raw_t, shift=shift, num_train_timesteps=n)
    return (per_sample * weights).mean()


__all__ = [
    "KEYFRAME_SIGMA",
    "NUM_TRAIN_TIMESTEPS",
    "add_noise",
    "keyframe",
    "predict_x0",
    "sample_training_sigma",
    "sgf_score_sigma",
    "shifted_sigmas",
    "wan_anyflow_v15_loss",
]
