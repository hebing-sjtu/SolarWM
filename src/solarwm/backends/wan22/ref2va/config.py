"""Torch-free validation of Wan Ref2VA proxy run configs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from solarwm.errors import BackendContractError

from .data import Ref2VAWanSpec

# Spatial VAE compression and native clip of each family.
FAMILY_GEOMETRY = {
    "wan22_ti2v_5b": {"compression": 16, "pixel_frames": 121, "fps": 24},
    "wan22_i2v_a14b": {"compression": 8, "pixel_frames": 81, "fps": 16},
}


def _require(section: Mapping[str, Any], keys: tuple[str, ...], label: str) -> None:
    missing = [key for key in keys if key not in section]
    if missing:
        raise BackendContractError(f"Wan Ref2VA {label} misses {missing}")


def validate_ref2va_config(config: Mapping[str, Any], *, expected_family: str) -> Ref2VAWanSpec:
    model, data = config["model"], config["data"]
    train = config.get("train") or {}
    family = str(model.get("family"))
    if family != expected_family:
        raise BackendContractError(
            f"Wan Ref2VA config family {family!r} != backend {expected_family!r}"
        )
    _require(model, ("weights", "training_mode"), "model")
    mode = str(model["training_mode"])
    if mode not in ("full", "lora"):
        raise BackendContractError("model.training_mode must be full or lora")
    if mode == "lora":
        _require(model.get("adapter") or {}, ("target", "rank", "alpha"), "model.adapter")
    _require(data, ("data_path", "pixel_frames", "encoded_latents", "height", "width"), "data")
    try:
        spec = Ref2VAWanSpec.from_config(config)
    except (KeyError, ValueError) as exc:
        raise BackendContractError(f"invalid Wan Ref2VA data geometry: {exc}") from exc
    geometry = FAMILY_GEOMETRY[family]
    frames = int(data["pixel_frames"])
    if (frames - 1) % 4 or 1 + (frames - 1) // 4 != spec.latent_frames:
        raise BackendContractError(
            f"{frames} pixel frames do not encode to {spec.latent_frames} latents"
        )
    compression = geometry["compression"]
    if (int(data["height"]), int(data["width"])) != (
        spec.latent_height * compression,
        spec.latent_width * compression,
    ):
        raise BackendContractError(
            "data.height/width disagree with latent_height/width and the VAE compression"
        )
    for name, value in (
        ("latent", (spec.latent_height, spec.latent_width)),
        ("proxy latent", (spec.proxy_latent_height, spec.proxy_latent_width)),
    ):
        if value[0] % 2 or value[1] % 2:
            raise BackendContractError(
                f"Wan Ref2VA {name} grid {value} must be even for the 2x2 patch"
            )
    stage = str(train.get("stage", ""))
    action = str(config.get("action", "")).lower()
    if action == "train":
        _require(train, ("stage", "max_steps", "optimizer"), "train")
        _require(config.get("checkpoint") or {}, ("save_every_steps",), "checkpoint")
        _require(config.get("runtime") or {}, ("output_dir",), "runtime")
        if stage == "stage1":
            _require(
                train,
                ("diffusion_ratio", "consistency_ratio", "finite_difference_epsilon"),
                "train (stage1)",
            )
        if stage == "stage2":
            _require(
                train, ("critic_optimizer", "score_min_sigma", "score_max_sigma"), "train (stage2)"
            )
            initialization = (config.get("checkpoint") or {}).get("initialization") or {}
            _require(
                initialization,
                ("student", "teacher", "critic"),
                "checkpoint.initialization (stage2)",
            )
            if spec.train_target_latents != spec.latent_frames:
                raise BackendContractError(
                    "Wan Ref2VA SGF needs train_target_latents == encoded_latents"
                )
        if stage == "stage1" and spec.train_target_latents % spec.chunk_latents:
            raise BackendContractError(
                "Wan Ref2VA causal stages train whole chunks of chunk_latents"
            )
    elif action == "infer":
        inference = config.get("inference") or {}
        _require(inference, ("stage", "checkpoint", "num_inference_steps"), "inference")
        _require(model, ("vae_weights",), "model")
    return spec


__all__ = ["FAMILY_GEOMETRY", "validate_ref2va_config"]
