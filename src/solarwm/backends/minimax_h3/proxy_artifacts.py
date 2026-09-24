"""Strict reader and contract for FastVideo-compatible H3 proxy caches."""

from __future__ import annotations

import hashlib
import json
import math
import pickle
import random
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from solarwm.errors import DataContractError
from solarwm.preencode import EncoderContract, TensorSpec
from solarwm.runtime import Topology

from .geometry import PROXY_STAGE0P5_GEOMETRY

H3_PROXY_PREENCODE_VERSION = "h3.ref2va-proxy.124f.v1"
H3_PROXY_DATASET_NAME = "h3_proxy_pt"
_CWM_GIVEN_FRAMES = {"w0": 1, "wn": 10}


def h3_proxy_encoder_contract(
    *,
    qwen_video_fps: float,
    cwm_system: str,
    align_proxy_reference_time: bool,
    anchor_short_edge: int,
) -> EncoderContract:
    """Describe the cached Ref2VA semantics without claiming native H3 compatibility."""

    role = str(cwm_system).strip().lower()
    if role not in _CWM_GIVEN_FRAMES:
        raise DataContractError(f"unsupported proxy CWM role {cwm_system!r}")
    geometry = PROXY_STAGE0P5_GEOMETRY
    return EncoderContract(
        schema="solarwm.encoder.v1",
        family="minimax_h3",
        format_version=H3_PROXY_PREENCODE_VERSION,
        pixel_frames=geometry.pixel_frames,
        latent_frames=geometry.encoded_latents,
        height=geometry.height,
        width=geometry.width,
        camera_convention="none",
        tensors=(
            TensorSpec("target_latents", (24, 37, 48, 84), "float32"),
            TensorSpec("proxy_latents", (24, 37, 12, 21), "float32"),
            TensorSpec("anchor_latents", (24, 1, None, None), "float32"),
            TensorSpec("prompt_embeds", (None, 5120), "float32"),
            TensorSpec("text_token_tags", (None,), "int64"),
        ),
        extras={
            "storage": "fastvideo-pt-cache-v1",
            "conditioning_mode": "ref2va_proxy",
            "reference_order": ["picture_anchor", "video_proxy"],
            "qwen_presentation": "<Picture 1> then <Video 1> plus CWM user caption",
            "qwen_hidden_state": 50,
            "qwen_video_fps": float(qwen_video_fps),
            "cwm_system": role,
            "num_given_latent_frames": _CWM_GIVEN_FRAMES[role],
            "align_proxy_reference_time": bool(align_proxy_reference_time),
            "anchor_short_edge": int(anchor_short_edge),
            "camera_conditioning": "none",
            "audio_conditioning": "zero-placeholder-no-loss",
        },
    )


@dataclass(frozen=True)
class H3ProxyArtifactBatch:
    sample_id: str
    start_frame: int
    plan_fingerprint: str
    target_latents: Any
    proxy_latents: Any
    anchor_latents: Any
    prompt_embeds: Any
    text_token_tags: Any
    cwm_system: str
    num_given_latent_frames: int
    audio_latents: Any = None
    dataset_source: str = "fastvideo_proxy_pt"


def _sample_paths(data_path: str) -> tuple[Path, ...]:
    source = Path(str(data_path)).expanduser()
    if not source.is_absolute():
        raise DataContractError("H3 proxy data_path must be absolute")
    if source.is_dir():
        paths = tuple(sorted(source.glob("*.pt")))
    elif source.is_file():
        base = source.parent
        rows = []
        try:
            for raw in source.read_text(encoding="utf-8").splitlines():
                value = raw.strip()
                if value and not value.startswith("#"):
                    path = Path(value)
                    rows.append(path if path.is_absolute() else base / path)
        except (OSError, UnicodeError) as exc:
            raise DataContractError(f"cannot read H3 proxy manifest {source}: {exc}") from exc
        paths = tuple(rows)
    else:
        raise DataContractError(f"H3 proxy data_path does not exist: {source}")
    if not paths:
        raise DataContractError(f"H3 proxy data_path contains no .pt samples: {source}")
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise DataContractError(f"H3 proxy manifest references missing samples: {missing[:4]}")
    return paths


class H3ProxyPtStream:
    """Infinite deterministic rank-partitioned stream over trusted local ``.pt`` files."""

    state_schema = "solarwm.minimax-h3-proxy-pt-reader.v1"

    def __init__(self, config: Mapping[str, Any], topology: Topology) -> None:
        data = config["data"]
        self.topology = topology
        self.seed = int(data.get("seed", 42))
        self.paths = _sample_paths(str(data["data_path"]))
        self.expected_role = str(data["cwm_system"]).strip().lower()
        self.expected_qwen_fps = float(data["qwen_video_fps"])
        self.encoder_profile = h3_proxy_encoder_contract(
            qwen_video_fps=self.expected_qwen_fps,
            cwm_system=self.expected_role,
            align_proxy_reference_time=bool(data["align_proxy_reference_time"]),
            anchor_short_edge=int(data["anchor_short_edge"]),
        ).as_dict()
        self.epoch = 0
        self.cursor = 0
        self._owned: tuple[Path, ...] = ()
        self._reset_epoch()

    def _reset_epoch(self) -> None:
        ordered = list(self.paths)
        random.Random(self.seed + self.epoch).shuffle(ordered)
        dp_world_size = int(self.topology.dp_world_size)
        samples_per_rank = math.ceil(len(ordered) / dp_world_size)
        total_size = samples_per_rank * dp_world_size
        if total_size > len(ordered):
            ordered.extend(ordered[: total_size - len(ordered)])
        self._owned = tuple(
            path
            for index, path in enumerate(ordered)
            if index % dp_world_size == int(self.topology.dp_rank)
        )
        if not self._owned:
            raise DataContractError(
                "H3 proxy rank owns no samples; cache size must be at least logical DP world size"
            )
        self.cursor = min(self.cursor, len(self._owned))

    @staticmethod
    def _float_tensor(value: Any, key: str) -> Any:
        import torch

        if not isinstance(value, torch.Tensor) or not torch.is_floating_point(value):
            raise DataContractError(f"H3 proxy {key} must be a floating tensor")
        return value.detach().cpu().float().contiguous()

    @staticmethod
    def _shape(value: Any, expected: tuple[int | None, ...], key: str) -> None:
        shape = tuple(int(edge) for edge in value.shape)
        if len(shape) != len(expected) or any(
            wanted is not None and observed != wanted
            for observed, wanted in zip(shape, expected, strict=True)
        ):
            raise DataContractError(f"H3 proxy {key} shape {shape} differs from {expected}")

    def _read(self, path: Path) -> H3ProxyArtifactBatch:
        import torch

        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except (EOFError, OSError, pickle.UnpicklingError, RuntimeError) as exc:
            raise DataContractError(f"cannot read H3 proxy sample {path}: {exc}") from exc
        if not isinstance(payload, Mapping):
            raise DataContractError(f"H3 proxy sample {path} is not a mapping")
        required = {
            "vae_latent",
            "proxy_latent",
            "anchor_latent",
            "text_embedding",
            "text_token_tags",
        }
        missing = sorted(required - set(payload))
        if missing:
            raise DataContractError(f"H3 proxy sample {path} misses {missing}")

        target = self._float_tensor(payload["vae_latent"], "vae_latent")
        proxy = self._float_tensor(payload["proxy_latent"], "proxy_latent")
        anchor = self._float_tensor(payload["anchor_latent"], "anchor_latent")
        prompt = self._float_tensor(payload["text_embedding"], "text_embedding")
        tags = payload["text_token_tags"]
        if not isinstance(tags, torch.Tensor) or tags.dtype != torch.int64:
            raise DataContractError("H3 proxy text_token_tags must be an int64 tensor")
        tags = tags.detach().cpu().contiguous()
        self._shape(target, (24, 37, 48, 84), "vae_latent")
        self._shape(proxy, (24, 37, 12, 21), "proxy_latent")
        self._shape(anchor, (24, 1, None, None), "anchor_latent")
        self._shape(prompt, (None, 5120), "text_embedding")
        self._shape(tags, (None,), "text_token_tags")
        if int(prompt.shape[0]) != int(tags.shape[0]):
            raise DataContractError("H3 proxy prompt rows and token tags differ")
        if anchor.shape[-2] < 2 or anchor.shape[-1] < 2:
            raise DataContractError("H3 proxy anchor latent grid is empty")
        if not bool(((tags == 0) | (tags == 1)).all().item()):
            raise DataContractError("H3 proxy text_token_tags may contain only 0 or 1")

        info = payload.get("info", {})
        if not isinstance(info, Mapping):
            raise DataContractError("H3 proxy info must be a mapping")
        role = str(info.get("cwm_system") or "").strip().lower()
        if role != self.expected_role:
            raise DataContractError(
                f"H3 proxy sample {path} has cwm_system={role!r}, expected {self.expected_role!r}"
            )
        qwen_fps = info.get("qwen_video_fps")
        if qwen_fps is not None and float(qwen_fps) != self.expected_qwen_fps:
            raise DataContractError(
                f"H3 proxy sample {path} has qwen_video_fps={qwen_fps}, "
                f"expected {self.expected_qwen_fps}"
            )
        audio = payload.get("audio_latent")
        if audio is not None:
            audio = self._float_tensor(audio, "audio_latent")
            self._shape(audio, (2, 32, None), "audio_latent")

        identity = hashlib.blake2s(
            f"{path.resolve()}\0{path.stat().st_size}".encode(), digest_size=16
        ).hexdigest()
        return H3ProxyArtifactBatch(
            sample_id=path.stem,
            start_frame=0,
            plan_fingerprint=identity,
            target_latents=target,
            proxy_latents=proxy,
            anchor_latents=anchor,
            prompt_embeds=prompt,
            text_token_tags=tags,
            cwm_system=role,
            num_given_latent_frames=_CWM_GIVEN_FRAMES[role],
            audio_latents=audio,
        )

    def next(self) -> H3ProxyArtifactBatch:
        if self.cursor >= len(self._owned):
            self.epoch += 1
            self.cursor = 0
            self._reset_epoch()
        path = self._owned[self.cursor]
        self.cursor += 1
        return self._read(path)

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema": self.state_schema,
            "epoch": self.epoch,
            "cursor": self.cursor,
            "seed": self.seed,
            "world_size": int(self.topology.raw_world_size),
            "dp_world_size": int(self.topology.dp_world_size),
            "dp_rank": int(self.topology.dp_rank),
            "paths_digest": hashlib.sha256(
                "\n".join(str(path.resolve()) for path in self.paths).encode()
            ).hexdigest(),
            "encoder_profile": json.loads(json.dumps(self.encoder_profile, sort_keys=True)),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        expected = self.state_dict()
        for key in ("schema", "seed", "world_size", "dp_world_size", "dp_rank", "paths_digest"):
            if state.get(key) != expected[key]:
                raise DataContractError(f"H3 proxy reader resume field {key!r} differs")
        self.epoch = int(state["epoch"])
        requested_cursor = int(state["cursor"])
        if requested_cursor < 0:
            raise DataContractError("H3 proxy reader resume cursor is out of range")
        self.cursor = 0
        self._reset_epoch()
        if requested_cursor > len(self._owned):
            raise DataContractError("H3 proxy reader resume cursor is out of range")
        self.cursor = requested_cursor

    def close(self) -> None:
        return None


__all__ = [
    "H3_PROXY_DATASET_NAME",
    "H3_PROXY_PREENCODE_VERSION",
    "H3ProxyArtifactBatch",
    "H3ProxyPtStream",
    "h3_proxy_encoder_contract",
]
