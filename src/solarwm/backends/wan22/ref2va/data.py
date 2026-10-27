"""Wan Ref2VA cache samples, their geometry contract and the training stream.

A cache sample is one ``.pt`` mapping written by FastVideo-oc's
``scripts/h3_proxy/prepare_data/encode_wan_proxy_samples.py``::

    target_latent   [C, T, H, W]     the target clip
    proxy_latent    [C, T, h, w]     the one proxy video, same latent frames as the target
    anchor_latent   [C, 1, Ha, Wa]   <Picture 1>, the target's first frame at its own size
    oc_latent       [C, 1, Ho, Wo]   optional <Picture 2> identity sheet
    i2v_latent      [16, T, H, W]    A14B only: VAE of [first frame, zeros...]
    prompt_embeds   [n, 4096]        umT5 rows of the document (with the OC line if any)
    prompt_embeds_no_oc [m, 4096]    the same document without the OC line (only with oc_latent)
    info            dict

``C`` is 48 for TI2V-5B and 16 for I2V-A14B.
"""

from __future__ import annotations

import hashlib
import math
import random
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA = "solarwm.wan22-ref2va-sample.v1"
FAMILY_LATENT_CHANNELS = {"wan22_ti2v_5b": 48, "wan22_i2v_a14b": 16}


@dataclass(frozen=True)
class Ref2VAWanSpec:
    family: str
    latent_channels: int
    latent_frames: int
    latent_height: int
    latent_width: int
    proxy_latent_height: int
    proxy_latent_width: int
    given_latents: int
    chunk_latents: int
    window_chunks: int
    train_target_latents: int
    align_proxy_reference_time: bool
    i2v_y: bool

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> Ref2VAWanSpec:
        family = str(config["model"]["family"])
        if family not in FAMILY_LATENT_CHANNELS:
            raise ValueError(
                f"Wan Ref2VA supports {sorted(FAMILY_LATENT_CHANNELS)}, got {family!r}"
            )
        data = config["data"]
        frames = int(data["encoded_latents"])
        spec = cls(
            family=family,
            latent_channels=FAMILY_LATENT_CHANNELS[family],
            latent_frames=frames,
            latent_height=int(data["latent_height"]),
            latent_width=int(data["latent_width"]),
            proxy_latent_height=int(data["proxy_latent_height"]),
            proxy_latent_width=int(data["proxy_latent_width"]),
            given_latents=int(data.get("num_given_latent_frames", 1)),
            chunk_latents=int(data.get("chunk_latents", 5)),
            window_chunks=int(data.get("window_chunks", 6)),
            train_target_latents=int(data.get("train_target_latents", frames)),
            align_proxy_reference_time=bool(data.get("align_proxy_reference_time", False)),
            i2v_y=family == "wan22_i2v_a14b",
        )
        if not 0 <= spec.given_latents <= frames:
            raise ValueError("num_given_latent_frames must lie in [0, encoded_latents]")
        if spec.train_target_latents > frames:
            raise ValueError("train_target_latents exceeds encoded_latents")
        return spec

    @property
    def rollout_chunks(self) -> int:
        return -(-self.latent_frames // self.chunk_latents)

    @property
    def rollout_latents(self) -> int:
        return self.rollout_chunks * self.chunk_latents


@dataclass
class Ref2VAWanSample:
    sample_id: str
    target: Any
    proxy: Any
    pictures: tuple[Any, ...]
    prompt: Any
    i2v_latent: Any | None
    info: Mapping[str, Any]
    fingerprint: str
    oc_dropped: bool = False


def sample_paths(data_path: str) -> tuple[Path, ...]:
    source = Path(str(data_path)).expanduser()
    if source.is_dir():
        paths = tuple(sorted(source.glob("*.pt")))
    elif source.is_file():
        rows = []
        for raw in source.read_text(encoding="utf-8").splitlines():
            value = raw.strip()
            if value and not value.startswith("#"):
                path = Path(value)
                rows.append(path if path.is_absolute() else source.parent / path)
        paths = tuple(rows)
    else:
        raise FileNotFoundError(f"Wan Ref2VA data_path does not exist: {source}")
    if not paths:
        raise ValueError(f"Wan Ref2VA data_path holds no samples: {source}")
    return paths


def _check(value: Any, expected: tuple[int | None, ...], key: str, path: Path) -> None:
    shape = tuple(int(edge) for edge in value.shape)
    if len(shape) != len(expected) or any(
        w is not None and s != w for s, w in zip(shape, expected, strict=True)
    ):
        raise ValueError(f"{path}: {key} has shape {shape}, expected {expected}")


def read_sample(
    path: Path, spec: Ref2VAWanSpec, *, use_oc: bool, drop_oc: bool = False
) -> Ref2VAWanSample:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping) or payload.get("schema") != SCHEMA:
        raise ValueError(f"{path} is not a {SCHEMA} sample")
    if payload.get("family") != spec.family:
        raise ValueError(
            f"{path} was encoded for {payload.get('family')!r}, the run is {spec.family!r}"
        )
    c = spec.latent_channels
    target = payload["target_latent"].float()
    proxy = payload["proxy_latent"].float()
    anchor = payload["anchor_latent"].float()
    _check(
        target,
        (c, spec.latent_frames, spec.latent_height, spec.latent_width),
        "target_latent",
        path,
    )
    _check(
        proxy,
        (c, spec.latent_frames, spec.proxy_latent_height, spec.proxy_latent_width),
        "proxy_latent",
        path,
    )
    _check(anchor, (c, 1, None, None), "anchor_latent", path)
    pictures = (anchor,)
    prompt = payload["prompt_embeds"]
    oc = payload.get("oc_latent")
    dropped = False
    if oc is not None:
        if use_oc and not drop_oc:
            _check(oc, (c, 1, None, None), "oc_latent", path)
            pictures = (anchor, oc.float())
        else:
            prompt = payload["prompt_embeds_no_oc"]
            dropped = bool(use_oc)
    _check(prompt, (None, 4096), "prompt_embeds", path)
    i2v = payload.get("i2v_latent")
    if spec.i2v_y:
        if i2v is None:
            raise ValueError(f"{path} lacks the A14B i2v_latent")
        i2v = i2v.float()
        _check(
            i2v, (16, spec.latent_frames, spec.latent_height, spec.latent_width), "i2v_latent", path
        )
    else:
        i2v = None
    fingerprint = hashlib.blake2s(
        f"{path.resolve()}\0{path.stat().st_size}".encode(), digest_size=16
    ).hexdigest()
    return Ref2VAWanSample(
        sample_id=path.stem,
        target=target,
        proxy=proxy,
        pictures=pictures,
        prompt=prompt.float(),
        i2v_latent=i2v,
        info=dict(payload.get("info") or {}),
        fingerprint=fingerprint,
        oc_dropped=dropped,
    )


class Ref2VAWanStream:
    """Infinite deterministic stream; each data-parallel rank owns a disjoint slice per epoch."""

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        rank: int,
        world_size: int,
        training: bool = True,
    ) -> None:
        data = config["data"]
        self.spec = Ref2VAWanSpec.from_config(config)
        self.paths = sample_paths(str(data["data_path"]))
        self.rank, self.world_size = int(rank), int(world_size)
        self.seed = int(data.get("seed", 42))
        self.use_oc = bool(data.get("oc_reference", False))
        drop = float(data.get("oc_drop_prob", 0.0))
        if not 0.0 <= drop < 1.0:
            raise ValueError("data.oc_drop_prob must lie in [0, 1)")
        self.oc_drop_prob = drop if training else 0.0
        self.epoch = 0
        self.cursor = 0
        self._owned: tuple[Path, ...] = ()
        self._reset()

    def _reset(self) -> None:
        ordered = list(self.paths)
        random.Random(self.seed + self.epoch).shuffle(ordered)
        per_rank = math.ceil(len(ordered) / self.world_size)
        ordered.extend(ordered[: per_rank * self.world_size - len(ordered)])
        self._owned = tuple(ordered[self.rank :: self.world_size])

    def _drops_oc(self, path: Path) -> bool:
        if not self.oc_drop_prob:
            return False
        key = f"{self.seed}\0{self.epoch}\0{self.cursor}\0{path.name}".encode()
        draw = int.from_bytes(hashlib.blake2s(key, digest_size=8).digest(), "big") / 2.0**64
        return draw < self.oc_drop_prob

    def next(self) -> Ref2VAWanSample:
        if self.cursor >= len(self._owned):
            self.epoch += 1
            self.cursor = 0
            self._reset()
        path = self._owned[self.cursor]
        drop = self._drops_oc(path)
        self.cursor += 1
        return read_sample(path, self.spec, use_oc=self.use_oc, drop_oc=drop)

    def state_dict(self) -> dict[str, Any]:
        digest = hashlib.sha256("\n".join(str(path) for path in self.paths).encode()).hexdigest()
        return {
            "epoch": self.epoch,
            "cursor": self.cursor,
            "seed": self.seed,
            "world_size": self.world_size,
            "paths": digest,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Resume; a state saved at another world size continues from the same position of the
        epoch's shuffled order (all ranks share one cursor), rounded up to the new world size."""

        current = self.state_dict()
        for key in ("seed", "paths"):
            if state.get(key) != current[key]:
                raise ValueError(f"Wan Ref2VA reader resume field {key!r} differs")
        self.epoch = int(state["epoch"])
        self._reset()
        position = int(state["cursor"]) * int(state["world_size"])
        self.cursor = -(-position // self.world_size)


__all__ = [
    "FAMILY_LATENT_CHANNELS",
    "SCHEMA",
    "Ref2VAWanSample",
    "Ref2VAWanSpec",
    "Ref2VAWanStream",
    "read_sample",
    "sample_paths",
]
