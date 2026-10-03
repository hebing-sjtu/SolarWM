"""Strict live/EMA loader for SolarWM H3 Ref2VA proxy checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from solarwm.checkpoint import verify_checkpoint
from solarwm.errors import BackendContractError

from .weights import canonical_lora_key


def load_proxy_checkpoint(
    path: str,
    lora: Any,
    *,
    weight_source: str,
    torch: Any,
) -> str:
    """Load only proxy LoRA weights, never optimizer or reader state."""

    source = str(weight_source).strip().lower()
    if source not in {"live", "ema"}:
        raise BackendContractError("H3 proxy inference weight_source must be live or ema")
    verified = verify_checkpoint(Path(path))
    contract = verified.contract
    if (
        contract.family != "minimax_h3"
        or contract.stage != "stage0p5"
        or contract.causal_mode != "bidirectional"
        or contract.objective != "flow_matching"
        or contract.objective_variant != "data_ward_velocity"
        or contract.parameterization != "peft-lora-r128-alpha128"
        or contract.data_generation != "h3.ref2va-proxy.124f.v1"
        or contract.camera_translation_transform != "none"
    ):
        raise BackendContractError(
            "inference checkpoint is not a SolarWM H3 124f Ref2VA proxy LoRA"
        )

    if source == "live":
        payload = torch.load(
            verified.path / "adapter.pt",
            map_location="cpu",
            mmap=True,
            weights_only=True,
        )
        if payload.get("metadata") != lora.metadata():
            raise BackendContractError("H3 proxy inference LoRA metadata differs")
        values = payload.get("state")
    else:
        payload = torch.load(
            verified.path / "ema.pt",
            map_location="cpu",
            mmap=True,
            weights_only=True,
        )
        if payload.get("schema") != "solarwm.minimax-h3-ema.v1" or not bool(
            payload.get("trainable_only")
        ):
            raise BackendContractError("H3 proxy checkpoint has no compatible EMA")
        values = payload.get("shadow")

    if not isinstance(values, dict):
        raise BackendContractError(f"H3 proxy {source} checkpoint has no tensor mapping")
    translated = {canonical_lora_key(key): value for key, value in values.items()}
    if len(translated) != len(values):
        raise BackendContractError(
            f"H3 proxy {source} checkpoint has colliding canonical LoRA keys"
        )
    expected = lora.parameter_by_key
    if set(translated) != set(expected):
        raise BackendContractError(
            "H3 proxy checkpoint LoRA keys differ: "
            f"missing={sorted(set(expected) - set(translated))[:8]} "
            f"extra={sorted(set(translated) - set(expected))[:8]}"
        )
    b_norm_sq = 0.0
    with torch.no_grad():
        for key, parameter in expected.items():
            value = translated[key]
            if tuple(value.shape) != tuple(parameter.shape) or not value.is_floating_point():
                raise BackendContractError(
                    f"H3 proxy checkpoint tensor descriptor differs for {key!r}"
                )
            parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))
            if ".lora_B." in key:
                b_norm_sq += float(parameter.detach().float().pow(2).sum().item())
    if b_norm_sq == 0.0:
        raise BackendContractError(
            "H3 proxy checkpoint left LoRA-B at zero; refusing to sample base weights"
        )
    return f"{verified.manifest_digest}:{source}:step={verified.step}"


__all__ = ["load_proxy_checkpoint"]
