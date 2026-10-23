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

from .geometry import PROXY_STAGE0P5_GEOMETRY, H3ProxyStage0p5Geometry, proxy_geometry_from_data

H3_PROXY_PREENCODE_VERSION = "h3.ref2va-proxy.124f.v1"
H3_PROXY_DATASET_NAME = "h3_proxy_pt"
# Given latent frames promised by each FastVideo CWM system prompt.
CWM_GIVEN_FRAMES = {"w0": 1, "wn": 10, "w0_depth_semantic": 1, "w0_omni": 1, "wn_omni": 10}
_CWM_GIVEN_FRAMES = CWM_GIVEN_FRAMES
# The role of a long take's later segments, which open on the previous segment's tail, for a
# run whose documents use the key role.
CWM_CONTINUATION_ROLES = {"w0": "wn", "w0_omni": "wn_omni"}
# What `encode_proxy_samples.py --proxy-references` can cache, and the default single DUV video.
PROXY_REFERENCE_KINDS = ("duv", "depth", "semantic", "style")
LEGACY_PROXY_REFERENCES = ("duv",)
OMNI_MIXED_ROLE = "w0_omni"
OMNI_ROLES = (OMNI_MIXED_ROLE, "wn_omni")
# Video references each role's system prompt names, in <Video N> order. Unlisted roles name one.
CWM_ROLE_REFERENCES = {"w0_depth_semantic": ("depth", "semantic")}


def proxy_references_from_data(data: Mapping[str, Any]) -> tuple[str, ...]:
    """Validated ``data.proxy_references``; absent means the single packed DUV video."""

    value = data.get("proxy_references", list(LEGACY_PROXY_REFERENCES))
    if not isinstance(value, (list, tuple)) or not value:
        raise DataContractError("data.proxy_references must be a non-empty list")
    references = tuple(str(item).strip().lower() for item in value)
    unknown = sorted(set(references) - set(PROXY_REFERENCE_KINDS))
    if unknown or len(set(references)) != len(references):
        raise DataContractError(
            f"data.proxy_references must be distinct values of {list(PROXY_REFERENCE_KINDS)}, "
            f"got {list(value)}"
        )
    return references


def proxy_variants_from_data(data: Mapping[str, Any]) -> tuple[str, ...]:
    """Allowed modalities for one mixed-single-reference omni run."""

    value = data.get("proxy_variants")
    if value is None:
        return ()
    if "proxy_references" in data:
        raise DataContractError(
            "data.proxy_variants and data.proxy_references are mutually exclusive"
        )
    if not isinstance(value, (list, tuple)) or not value:
        raise DataContractError("data.proxy_variants must be a non-empty list")
    variants = tuple(str(item).strip().lower() for item in value)
    unknown = sorted(set(variants) - set(PROXY_REFERENCE_KINDS))
    if unknown or len(set(variants)) != len(variants):
        raise DataContractError(
            f"data.proxy_variants must be distinct values of {list(PROXY_REFERENCE_KINDS)}, "
            f"got {list(value)}"
        )
    return variants


def check_role_references(role: str, references: tuple[str, ...]) -> None:
    """Refuse a CWM role whose system prompt describes other videos than the cache carries."""

    expected = CWM_ROLE_REFERENCES.get(role)
    if role in OMNI_ROLES:
        if len(references) != 1 or references[0] not in PROXY_REFERENCE_KINDS:
            raise DataContractError(
                f"CWM role {role!r} requires exactly one typed proxy video, got "
                f"{list(references)}"
            )
        return
    if expected is None and len(references) != 1:
        raise DataContractError(
            f"CWM role {role!r} describes one proxy video, but proxy_references is "
            f"{list(references)}; use w0_depth_semantic for depth+semantic"
        )
    if expected is not None and references != expected:
        raise DataContractError(
            f"CWM role {role!r} describes the video references {list(expected)}, "
            f"but proxy_references is {list(references)}"
        )


def proxy_latent_shape(
    geometry: H3ProxyStage0p5Geometry, references: tuple[str, ...]
) -> tuple[int, ...]:
    """Cached proxy tensor shape: ``proxy_latent`` for one DUV, ``proxy_latents`` otherwise."""

    shape = (
        geometry.latent_channels,
        geometry.encoded_latents,
        geometry.proxy_latent_height,
        geometry.proxy_latent_width,
    )
    return shape if references == LEGACY_PROXY_REFERENCES else (len(references), *shape)


def h3_proxy_encoder_contract(
    *,
    qwen_video_fps: float,
    cwm_system: str,
    align_proxy_reference_time: bool,
    anchor_short_edge: int,
    geometry: H3ProxyStage0p5Geometry = PROXY_STAGE0P5_GEOMETRY,
    proxy_references: tuple[str, ...] = LEGACY_PROXY_REFERENCES,
    proxy_variants: tuple[str, ...] = (),
    proxy_fit: str | None = None,
    proxy_code_resize: str | None = None,
    oc_reference: bool = False,
) -> EncoderContract:
    """Describe the cached Ref2VA semantics without claiming native H3 compatibility.

    The legacy single-DUV profile is emitted exactly as before, so existing checkpoints keep
    resuming; other reference sets add their names to the extras.
    """

    role = str(cwm_system).strip().lower()
    if role not in _CWM_GIVEN_FRAMES:
        raise DataContractError(f"unsupported proxy CWM role {cwm_system!r}")
    variants = tuple(proxy_variants)
    references = tuple(proxy_references)
    if variants:
        if role not in OMNI_ROLES:
            raise DataContractError(
                f"mixed proxy variants require a CWM role of {list(OMNI_ROLES)}, got {role!r}"
            )
        if references != ("proxy",):
            raise DataContractError("mixed proxy variants use one generic video reference")
    else:
        check_role_references(role, references)
    legacy = references == LEGACY_PROXY_REFERENCES
    target_shape = (
        geometry.latent_channels,
        geometry.encoded_latents,
        geometry.latent_height,
        geometry.latent_width,
    )
    if variants:
        reference_extras: dict[str, Any] = {
            "reference_order": ["picture_anchor", "video_proxy"],
            "qwen_presentation": (
                "<Picture 1> then one modality-labelled <Video 1> plus CWM user caption"
            ),
            "proxy_variants": list(variants),
            "proxy_fit": proxy_fit,
            "proxy_code_resize": proxy_code_resize,
        }
    elif legacy:
        reference_extras = {
            "reference_order": ["picture_anchor", "video_proxy"],
            "qwen_presentation": "<Picture 1> then <Video 1> plus CWM user caption",
        }
    else:
        reference_extras = {
            "reference_order": ["picture_anchor", *(f"video_{kind}" for kind in references)],
            "qwen_presentation": "<Picture 1> then "
            + ", ".join(f"<Video {index + 1}>" for index in range(len(references)))
            + " plus CWM user caption",
            "proxy_references": list(references),
        }
    if oc_reference:
        order = reference_extras["reference_order"]
        reference_extras = {
            **reference_extras,
            "reference_order": [order[0], "picture_oc_optional", *order[1:]],
            "qwen_presentation": reference_extras["qwen_presentation"].replace(
                "<Picture 1> then", "<Picture 1>, an optional <Picture 2> OC sheet, then"
            ),
            "oc_reference": True,
        }
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
            TensorSpec("target_latents", target_shape, "float32"),
            TensorSpec("proxy_latents", proxy_latent_shape(geometry, references), "float32"),
            TensorSpec("anchor_latents", (24, 1, None, None), "float32"),
            TensorSpec("prompt_embeds", (None, 5120), "float32"),
            TensorSpec("text_token_tags", (None,), "int64"),
        ),
        extras={
            "storage": "fastvideo-pt-cache-v1",
            "conditioning_mode": "ref2va_proxy",
            **reference_extras,
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
    proxy_references: tuple[str, ...] = LEGACY_PROXY_REFERENCES
    dataset_source: str = "fastvideo_proxy_pt"
    # ``<Picture 2>`` identity sheet, and the Qwen rows of the same document without it.
    oc_latents: Any = None
    prompt_embeds_no_oc: Any = None
    text_token_tags_no_oc: Any = None
    # Preference pairs: ``target_latents`` is the preferred video, this the rejected one.
    target_latents_lose: Any = None

    def image_latents(self) -> tuple[Any, ...]:
        """Reference pictures in ``<Picture N>`` order: the anchor, then the OC sheet if any."""

        if self.oc_latents is None:
            return (self.anchor_latents,)
        return (self.anchor_latents, self.oc_latents)


def batch_given_frames(
    data: Mapping[str, Any], batch: H3ProxyArtifactBatch, *, continuation: bool = False
) -> int:
    """The batch's given latents, once its role is the run's, or with ``continuation`` the
    role of a long take's later segments."""

    role = str(data["cwm_system"]).strip().lower()
    allowed = {role}
    if continuation and role in CWM_CONTINUATION_ROLES:
        allowed.add(CWM_CONTINUATION_ROLES[role])
    if batch.cwm_system not in allowed:
        raise RuntimeError(
            f"H3 proxy CWM role {batch.cwm_system!r} is not one of this run's {sorted(allowed)}"
        )
    given = int(batch.num_given_latent_frames)
    if given != CWM_GIVEN_FRAMES[batch.cwm_system]:
        raise RuntimeError("H3 proxy given-frame count differs from cached CWM role")
    return given


def oc_reference_from_data(data: Mapping[str, Any]) -> tuple[bool, float]:
    """``data.oc_reference`` and the training-only ``data.oc_drop_prob``."""

    enabled = data.get("oc_reference", False)
    if not isinstance(enabled, bool):
        raise DataContractError("data.oc_reference must be a boolean")
    drop = data.get("oc_drop_prob", 0.0)
    if isinstance(drop, bool) or not isinstance(drop, (int, float)) or not 0.0 <= drop < 1.0:
        raise DataContractError("data.oc_drop_prob must be a number in [0, 1)")
    if drop and not enabled:
        raise DataContractError("data.oc_drop_prob requires data.oc_reference=true")
    return enabled, float(drop)


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

    def __init__(
        self,
        config: Mapping[str, Any],
        topology: Topology,
        *,
        selection_seed: int | None = None,
        oc_dropout: bool = False,
        continuation: bool = False,
    ) -> None:
        data = config["data"]
        self.topology = topology
        self.continuation = bool(continuation)
        self.seed = int(data.get("seed", 42) if selection_seed is None else selection_seed)
        self.oc_reference, drop = oc_reference_from_data(data)
        # Validation and inference always show the sheet a sample carries.
        self.oc_drop_prob = drop if oc_dropout else 0.0
        self.paths = _sample_paths(str(data["data_path"]))
        self.expected_role = str(data["cwm_system"]).strip().lower()
        self.expected_qwen_fps = float(data["qwen_video_fps"])
        try:
            self.geometry = proxy_geometry_from_data(data)
        except ValueError as exc:
            raise DataContractError(str(exc)) from exc
        self.variants = proxy_variants_from_data(data)
        self.references = ("proxy",) if self.variants else proxy_references_from_data(data)
        self.expected_fit = str(data.get("proxy_fit") or "")
        self.expected_code_resize = str(data.get("proxy_code_resize") or "")
        self.encoder_profile = h3_proxy_encoder_contract(
            qwen_video_fps=self.expected_qwen_fps,
            cwm_system=self.expected_role,
            align_proxy_reference_time=bool(data["align_proxy_reference_time"]),
            anchor_short_edge=int(data["anchor_short_edge"]),
            geometry=self.geometry,
            proxy_references=self.references,
            proxy_variants=self.variants,
            proxy_fit=self.expected_fit or None,
            proxy_code_resize=self.expected_code_resize or None,
            oc_reference=self.oc_reference,
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

    def _oc_fields(self, payload: Mapping[str, Any], path: Path, *, drop: bool) -> dict[str, Any]:
        """The OC latent and both Qwen row sets, resolved to what this sample presents.

        A cache row with a sheet stores the document's Qwen rows with ``<Picture 2>`` as
        ``text_embedding`` and without it as ``text_embedding_no_oc``. Runs without
        ``oc_reference`` read the latter, so an OC cache serves them exactly as the cache it
        was refreshed from.
        """

        import torch

        if "oc_latent" not in payload:
            return {}
        missing = sorted({"text_embedding_no_oc", "text_token_tags_no_oc"} - set(payload))
        if missing:
            raise DataContractError(f"H3 proxy sample {path} has oc_latent but misses {missing}")
        oc = self._float_tensor(payload["oc_latent"], "oc_latent")
        self._shape(oc, (24, 1, None, None), "oc_latent")
        if oc.shape[-2] < 2 or oc.shape[-1] < 2:
            raise DataContractError("H3 proxy OC latent grid is empty")
        prompt = self._float_tensor(payload["text_embedding_no_oc"], "text_embedding_no_oc")
        tags = payload["text_token_tags_no_oc"]
        if not isinstance(tags, torch.Tensor) or tags.dtype != torch.int64:
            raise DataContractError("H3 proxy text_token_tags_no_oc must be an int64 tensor")
        self._shape(prompt, (None, 5120), "text_embedding_no_oc")
        self._shape(tags, (None,), "text_token_tags_no_oc")
        if int(prompt.shape[0]) != int(tags.shape[0]):
            raise DataContractError("H3 proxy no-OC prompt rows and token tags differ")
        tags = tags.detach().cpu().contiguous()
        if not self.oc_reference or drop:
            return {"prompt_embeds": prompt, "text_token_tags": tags}
        return {"oc_latents": oc, "prompt_embeds_no_oc": prompt, "text_token_tags_no_oc": tags}

    def _drops_oc(self, path: Path) -> bool:
        """A per-visit coin, identical on every rank that reads this visit (SP peers do)."""

        if not self.oc_drop_prob:
            return False
        key = f"{self.seed}\0{self.epoch}\0{self.cursor}\0{path.name}".encode()
        draw = int.from_bytes(hashlib.blake2s(key, digest_size=8).digest(), "big") / 2.0**64
        return draw < self.oc_drop_prob

    def _read(self, path: Path, *, drop_oc: bool = False) -> H3ProxyArtifactBatch:
        import torch

        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except (EOFError, OSError, pickle.UnpicklingError, RuntimeError) as exc:
            raise DataContractError(f"cannot read H3 proxy sample {path}: {exc}") from exc
        if not isinstance(payload, Mapping):
            raise DataContractError(f"H3 proxy sample {path} is not a mapping")
        legacy = self.references == LEGACY_PROXY_REFERENCES
        proxy_key = "proxy_latent" if legacy else "proxy_latents"
        required = {
            "vae_latent",
            proxy_key,
            "anchor_latent",
            "text_embedding",
            "text_token_tags",
        }
        missing = sorted(required - set(payload))
        if missing:
            raise DataContractError(f"H3 proxy sample {path} misses {missing}")

        target = self._float_tensor(payload["vae_latent"], "vae_latent")
        proxy = self._float_tensor(payload[proxy_key], proxy_key)
        anchor = self._float_tensor(payload["anchor_latent"], "anchor_latent")
        prompt = self._float_tensor(payload["text_embedding"], "text_embedding")
        tags = payload["text_token_tags"]
        if not isinstance(tags, torch.Tensor) or tags.dtype != torch.int64:
            raise DataContractError("H3 proxy text_token_tags must be an int64 tensor")
        tags = tags.detach().cpu().contiguous()
        geometry = self.geometry
        self._shape(
            target,
            (
                geometry.latent_channels,
                geometry.encoded_latents,
                geometry.latent_height,
                geometry.latent_width,
            ),
            "vae_latent",
        )
        self._shape(proxy, proxy_latent_shape(geometry, self.references), proxy_key)
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
        allowed = {self.expected_role}
        if self.continuation and self.expected_role in CWM_CONTINUATION_ROLES:
            allowed.add(CWM_CONTINUATION_ROLES[self.expected_role])
        if role not in allowed:
            raise DataContractError(
                f"H3 proxy sample {path} has cwm_system={role!r}, expected one of {sorted(allowed)}"
            )
        qwen_fps = info.get("qwen_video_fps")
        if qwen_fps is not None and float(qwen_fps) != self.expected_qwen_fps:
            raise DataContractError(
                f"H3 proxy sample {path} has qwen_video_fps={qwen_fps}, "
                f"expected {self.expected_qwen_fps}"
            )
        recorded = tuple(info.get("proxy_references") or LEGACY_PROXY_REFERENCES)
        if self.variants:
            modality = str(info.get("proxy_modality") or "").strip().lower()
            recorded_variants = tuple(info.get("proxy_variants") or ())
            if (
                modality not in self.variants
                or recorded != (modality,)
                or recorded_variants != self.variants
                or str(info.get("fit") or "") != self.expected_fit
                or str(info.get("code_resize") or "") != self.expected_code_resize
            ):
                raise DataContractError(
                    f"H3 proxy sample {path} has modality={modality!r} and "
                    f"proxy_references={list(recorded)}, variants={list(recorded_variants)}, "
                    f"fit={info.get('fit')!r}, code_resize={info.get('code_resize')!r}; "
                    f"expected exactly one of {list(self.variants)}, "
                    f"variants={list(self.variants)}, "
                    f"fit={self.expected_fit!r}, code_resize={self.expected_code_resize!r}"
                )
            sample_references = recorded
        else:
            if recorded != self.references:
                raise DataContractError(
                    f"H3 proxy sample {path} has proxy_references={list(recorded)}, "
                    f"expected {list(self.references)}"
                )
            sample_references = self.references
        audio = payload.get("audio_latent")
        if audio is not None:
            audio = self._float_tensor(audio, "audio_latent")
            self._shape(audio, (2, 32, None), "audio_latent")
        lose = payload.get("vae_latent_lose")
        if lose is not None:
            lose = self._float_tensor(lose, "vae_latent_lose")
            self._shape(lose, tuple(target.shape), "vae_latent_lose")

        identity = hashlib.blake2s(
            f"{path.resolve()}\0{path.stat().st_size}".encode(), digest_size=16
        ).hexdigest()
        fields: dict[str, Any] = {
            "prompt_embeds": prompt,
            "text_token_tags": tags,
            **self._oc_fields(payload, path, drop=drop_oc),
        }
        return H3ProxyArtifactBatch(
            sample_id=path.stem,
            start_frame=0,
            plan_fingerprint=identity,
            target_latents=target,
            proxy_latents=proxy,
            anchor_latents=anchor,
            cwm_system=role,
            num_given_latent_frames=_CWM_GIVEN_FRAMES[role],
            audio_latents=audio,
            proxy_references=sample_references,
            target_latents_lose=lose,
            **fields,
        )

    def read(self, path: Path) -> H3ProxyArtifactBatch:
        """One sample of this reader's contract, outside the shuffled stream."""

        return self._read(Path(path))

    def next(self) -> H3ProxyArtifactBatch:
        if self.cursor >= len(self._owned):
            self.epoch += 1
            self.cursor = 0
            self._reset_epoch()
        path = self._owned[self.cursor]
        drop_oc = self._drops_oc(path)
        self.cursor += 1
        return self._read(path, drop_oc=drop_oc)

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
    "CWM_CONTINUATION_ROLES",
    "CWM_GIVEN_FRAMES",
    "CWM_ROLE_REFERENCES",
    "H3_PROXY_DATASET_NAME",
    "H3_PROXY_PREENCODE_VERSION",
    "LEGACY_PROXY_REFERENCES",
    "OMNI_MIXED_ROLE",
    "OMNI_ROLES",
    "PROXY_REFERENCE_KINDS",
    "H3ProxyArtifactBatch",
    "H3ProxyPtStream",
    "batch_given_frames",
    "check_role_references",
    "h3_proxy_encoder_contract",
    "oc_reference_from_data",
    "proxy_latent_shape",
    "proxy_references_from_data",
    "proxy_variants_from_data",
]
