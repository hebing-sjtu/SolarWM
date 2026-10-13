"""Strict live/EMA loader for SolarWM H3 Ref2VA proxy checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from solarwm.checkpoint import verify_checkpoint
from solarwm.errors import BackendContractError

from .weights import canonical_lora_key

_PROXY_ROUTES = {
    "stage0p5": ("bidirectional", "flow_matching", "data_ward_velocity"),
    "stage1": ("teacher_forcing", "anyflow_forward_map", "v1_5"),
}


def load_proxy_checkpoint(
    path: str,
    lora: Any,
    *,
    weight_source: str,
    torch: Any,
    stage: str = "stage0p5",
    model_stage: str | None = None,
) -> str:
    """Load only proxy LoRA weights of a ``stage`` checkpoint for inference or warm-start.

    A ``model_stage`` other than ``stage`` (a Stage0.5 LoRA starting Stage1) skips the base
    parameter inventory, which Stage1's frozen AnyFlow delta embedding enlarges.
    """

    source = str(weight_source).strip().lower()
    if source not in {"live", "ema"}:
        raise BackendContractError("H3 proxy checkpoint weight_source must be live or ema")
    if stage not in _PROXY_ROUTES:
        raise BackendContractError(f"H3 proxy checkpoints have no {stage!r} route")
    verified = verify_checkpoint(Path(path))
    contract = verified.contract
    if (
        contract.family != "minimax_h3"
        or contract.stage != stage
        or (contract.causal_mode, contract.objective, contract.objective_variant)
        != _PROXY_ROUTES[stage]
        or contract.parameterization != "peft-lora-r128-alpha128"
        or contract.data_generation != "h3.ref2va-proxy.124f.v1"
        or contract.camera_translation_transform != "none"
    ):
        raise BackendContractError(f"checkpoint is not a SolarWM H3 124f Ref2VA proxy {stage} LoRA")

    if source == "live":
        payload = torch.load(
            verified.path / "adapter.pt",
            map_location="cpu",
            mmap=True,
            weights_only=True,
        )
        saved, expected_metadata = payload.get("metadata"), lora.metadata()
        if model_stage not in (None, stage) and isinstance(saved, dict):
            saved, expected_metadata = (
                {
                    **value,
                    "base_identity": {
                        key: item
                        for key, item in value.get("base_identity", {}).items()
                        if key != "parameter_inventory"
                    },
                }
                for value in (saved, expected_metadata)
            )
        if saved != expected_metadata:
            raise BackendContractError("H3 proxy LoRA metadata differs")
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
