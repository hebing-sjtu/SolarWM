"""A Wan2.2 transformer that reads explicitly positioned, per-token-timed latent segments.

Parameter names are exactly the official Wan2.2 ones (``patch_embedding``,
``text_embedding``, ``time_embedding``, ``time_projection``, ``blocks.N.{self_attn, cross_attn,
norm3, ffn, modulation}``, ``head``), so TI2V-5B and the I2V-A14B experts load strictly. What
differs from the official forward is the input contract: a list of latent segments, an
explicit ``(t, h, w)`` RoPE coordinate per token, a timestep per token, and an attention plan.
Tokens with equal timesteps form runs, and modulation is applied per run instead of
materializing a per-token ``[L, 6, dim]`` embedding.
"""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from ..runtime.modeling.model import WanLayerNorm, WanRMSNorm, sinusoidal_embedding_1d
from .attention import dense_attention

ARCHITECTURES: dict[str, dict[str, Any]] = {
    "wan22_ti2v_5b": {
        "in_dim": 48,
        "out_dim": 48,
        "dim": 3072,
        "ffn_dim": 14336,
        "num_heads": 24,
        "num_layers": 30,
    },
    "wan22_i2v_a14b": {
        "in_dim": 36,
        "out_dim": 16,
        "dim": 5120,
        "ffn_dim": 13824,
        "num_heads": 40,
        "num_layers": 40,
    },
}
_COMMON = {"freq_dim": 256, "text_dim": 4096, "text_len": 512, "eps": 1e-6, "patch_size": (1, 2, 2)}


def time_runs(
    timesteps: Any, r_timesteps: Any | None
) -> tuple[Any, Any, tuple[tuple[int, int, int], ...]]:
    """Unique ``(t, r)`` values and ``(start, stop, index)`` runs of equal consecutive tokens."""

    pairs = torch.stack(
        (timesteps, timesteps if r_timesteps is None else r_timesteps), dim=1
    ).float()
    unique, inverse = torch.unique(pairs, dim=0, return_inverse=True)
    values, counts = torch.unique_consecutive(inverse, return_counts=True)
    stops = torch.cumsum(counts, 0)
    runs = tuple(
        (int(stop - count), int(stop), int(value))
        for value, count, stop in zip(values.tolist(), counts.tolist(), stops.tolist(), strict=True)
    )
    return unique[:, 0].contiguous(), unique[:, 1].contiguous(), runs


def _per_run(tensor: Any, table: Any, runs: tuple[tuple[int, int, int], ...], fn: Any) -> Any:
    if len(runs) == 1:
        return fn(tensor, table[runs[0][2]])
    return torch.cat(
        [fn(tensor[:, start:stop], table[index]) for start, stop, index in runs], dim=1
    )


def rotate(x: Any, cos: Any, sin: Any) -> Any:
    """Wan RoPE on interleaved pairs: ``(a + ib) * (cos + i sin)`` per head-dim pair."""

    shape = x.shape
    pairs = x.float().reshape(*shape[:-1], -1, 2)
    a, b = pairs[..., 0], pairs[..., 1]
    cos, sin = cos[None, :, None, :], sin[None, :, None, :]
    return torch.stack((a * cos - b * sin, a * sin + b * cos), dim=-1).reshape(shape).type_as(x)


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps)
        self.norm_k = WanRMSNorm(dim, eps=eps)

    def forward(self, x: Any, cos: Any, sin: Any, plan: Any, layer: int) -> Any:
        b, s = x.shape[:2]
        shape = (b, s, self.num_heads, self.head_dim)
        q = rotate(self.norm_q(self.q(x)).view(shape), cos, sin)
        k = rotate(self.norm_k(self.k(x)).view(shape), cos, sin)
        v = self.v(x).view(shape)
        out = dense_attention(q, k, v) if plan is None else plan.attend(q, k, v, layer=layer)
        return self.o(out.flatten(2))


class CrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps)
        self.norm_k = WanRMSNorm(dim, eps=eps)

    def forward(self, x: Any, context: Any) -> Any:
        b = x.shape[0]
        q = self.norm_q(self.q(x)).view(b, -1, self.num_heads, self.head_dim)
        k = self.norm_k(self.k(context)).view(b, -1, self.num_heads, self.head_dim)
        v = self.v(context).view(b, -1, self.num_heads, self.head_dim)
        return self.o(dense_attention(q, k, v).flatten(2))


class Ref2VAWanBlock(nn.Module):
    def __init__(self, dim: int, ffn_dim: int, num_heads: int, eps: float) -> None:
        super().__init__()
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = SelfAttention(dim, num_heads, eps)
        self.norm3 = WanLayerNorm(dim, eps, elementwise_affine=True)
        self.cross_attn = CrossAttention(dim, num_heads, eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate="tanh"), nn.Linear(ffn_dim, dim)
        )
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x: Any,
        e0: Any,
        runs: tuple[tuple[int, int, int], ...],
        cos: Any,
        sin: Any,
        context: Any,
        plan: Any = None,
        layer: int = 0,
    ) -> Any:
        mod = (self.modulation.float() + e0.float()).unbind(1)  # six [U, dim]

        def affine(h: Any, scale_shift: Any) -> Any:
            scale, shift = scale_shift
            return (h.float() * (1 + scale) + shift).type_as(x)

        def gate(h: Any, value: Any) -> Any:
            return (h.float() * value).type_as(x)

        ss1 = torch.stack((mod[1], mod[0]), dim=1)
        ss2 = torch.stack((mod[4], mod[3]), dim=1)
        h = _per_run(self.norm1(x), ss1, runs, lambda a, t: affine(a, t.unbind(0)))
        x = x + _per_run(self.self_attn(h, cos, sin, plan, layer), mod[2], runs, gate)
        x = x + self.cross_attn(self.norm3(x), context)
        h = _per_run(self.norm2(x), ss2, runs, lambda a, t: affine(a, t.unbind(0)))
        return x + _per_run(self.ffn(h), mod[5], runs, gate)


class Head(nn.Module):
    def __init__(
        self, dim: int, out_dim: int, patch_size: tuple[int, int, int], eps: float
    ) -> None:
        super().__init__()
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, math.prod(patch_size) * out_dim)
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x: Any, e: Any, runs: tuple[tuple[int, int, int], ...]) -> Any:
        mod = (self.modulation.float() + e.float()[:, None]).unbind(1)  # shift, scale: [U, dim]
        table = torch.stack((mod[1], mod[0]), dim=1)

        def affine(h: Any, scale_shift: Any) -> Any:
            scale, shift = scale_shift.unbind(0)
            return (h.float() * (1 + scale) + shift).type_as(x)

        return self.head(_per_run(self.norm(x), table, runs, affine))


class Ref2VAWanModel(nn.Module):
    """The Wan2.2 DiT over Ref2VA segment documents."""

    def __init__(self, family: str, *, anyflow_gate: float | None = None, **overrides: Any) -> None:
        super().__init__()
        if family not in ARCHITECTURES:
            raise ValueError(f"unknown Wan Ref2VA family {family!r}")
        config = {**_COMMON, **ARCHITECTURES[family], **overrides}
        self.family = family
        self.config = config
        dim, heads = int(config["dim"]), int(config["num_heads"])
        self.dim, self.num_heads = dim, heads
        self.freq_dim = int(config["freq_dim"])
        self.text_len = int(config["text_len"])
        self.in_dim, self.out_dim = int(config["in_dim"]), int(config["out_dim"])
        self.patch_size = tuple(int(v) for v in config["patch_size"])
        self.patch_embedding = nn.Conv3d(
            self.in_dim, dim, kernel_size=self.patch_size, stride=self.patch_size
        )
        self.text_embedding = nn.Sequential(
            nn.Linear(int(config["text_dim"]), dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(dim, dim),
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(self.freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))
        self.blocks = nn.ModuleList(
            Ref2VAWanBlock(dim, int(config["ffn_dim"]), heads, float(config["eps"]))
            for _ in range(int(config["num_layers"]))
        )
        self.head = Head(dim, self.out_dim, self.patch_size, float(config["eps"]))
        self.anyflow_gate = None if anyflow_gate is None else float(anyflow_gate)
        if self.anyflow_gate is not None:
            self.delta_embedding = copy.deepcopy(self.time_embedding)

    @property
    def uses_anyflow(self) -> bool:
        return self.anyflow_gate is not None

    def initialize_anyflow_delta_from_time(self) -> None:
        if self.uses_anyflow:
            self.delta_embedding.load_state_dict(self.time_embedding.state_dict())

    def _rope(self, coords: Any) -> tuple[Any, Any]:
        """Per-token ``cos``/``sin`` ``[L, head_dim/2]`` of Wan's split ``(t, h, w)`` RoPE."""

        head_dim = self.dim // self.num_heads
        axis_dims = (head_dim - 4 * (head_dim // 6), 2 * (head_dim // 6), 2 * (head_dim // 6))
        if int(coords.max()) >= 1024 or int(coords.min()) < 0:
            raise ValueError("Wan RoPE coordinates must lie in [0, 1024)")
        angles = []
        for axis, axis_dim in enumerate(axis_dims):
            inverse = 1.0 / torch.pow(
                10000.0,
                torch.arange(0, axis_dim, 2, dtype=torch.float64, device=coords.device) / axis_dim,
            )
            angles.append(coords[:, axis].to(torch.float64)[:, None] * inverse[None])
        angles = torch.cat(angles, dim=1)
        return torch.cos(angles).float(), torch.sin(angles).float()

    def _time(self, timesteps: Any, r_timesteps: Any) -> tuple[Any, Any]:
        def embed(module: Any, values: Any) -> Any:
            return module(
                sinusoidal_embedding_1d(self.freq_dim, values).to(module[0].weight.dtype)
            ).float()

        e = embed(self.time_embedding, timesteps)
        if self.uses_anyflow:
            e = (1.0 - self.anyflow_gate) * e + self.anyflow_gate * embed(
                self.delta_embedding, r_timesteps
            )
        projection = self.time_projection[1].weight.dtype
        e0 = self.time_projection(e.to(projection)).unflatten(1, (6, self.dim))
        return e, e0.float()

    def embed(self, segments: Sequence[Any]) -> Any:
        tokens = [
            self.patch_embedding(segment.to(self.patch_embedding.weight.dtype))
            .flatten(2)
            .transpose(1, 2)
            for segment in segments
        ]
        return torch.cat(tokens, dim=1)

    def unpatchify(self, tokens: Any, frames: int, height: int, width: int) -> Any:
        """``[B, F*h*w, prod(patch)*C]`` to ``[B, C, F, 2h, 2w]`` for token grid ``h x w``."""

        b = tokens.shape[0]
        pt, ph, pw = self.patch_size
        x = tokens.view(b, frames, height, width, pt, ph, pw, self.out_dim)
        x = torch.einsum("bfhwpqrc->bcfphqwr", x)
        return x.reshape(b, self.out_dim, frames * pt, height * ph, width * pw)

    def forward(
        self,
        segments: Sequence[Any],
        *,
        coords: Any,
        timesteps: Any,
        context: Any,
        r_timesteps: Any | None = None,
        plan: Any = None,
        output: tuple[int, int] | None = None,
    ) -> Any:
        """Velocity tokens ``[B, L_out, prod(patch)*C]`` of rows ``output`` (all rows by default).

        ``segments`` are ``[B, in_dim, F, H, W]`` latents in layout order, ``coords`` ``[L, 3]``
        integer RoPE coordinates, ``timesteps``/``r_timesteps`` ``[L]`` Wan times in
        ``[0, 1000]`` (``r`` only for AnyFlow models), ``context`` ``[B, <=512, 4096]`` umT5 rows.
        """

        if self.uses_anyflow and r_timesteps is None:
            r_timesteps = timesteps
        x = self.embed(segments)
        if x.shape[1] != coords.shape[0] or timesteps.shape[0] != coords.shape[0]:
            raise ValueError(
                f"Wan Ref2VA tokens {x.shape[1]}, coords {coords.shape[0]} and timesteps "
                f"{timesteps.shape[0]} differ"
            )
        t_values, r_values, runs = time_runs(timesteps, r_timesteps if self.uses_anyflow else None)
        e, e0 = self._time(t_values, r_values)
        cos, sin = self._rope(coords)
        pad = self.text_len - int(context.shape[1])
        if pad < 0:
            raise ValueError(f"umT5 context has {context.shape[1]} rows, more than {self.text_len}")
        context = torch.cat(
            (context, context.new_zeros(context.shape[0], pad, context.shape[2])), dim=1
        )
        context = self.text_embedding(context.to(self.text_embedding[0].weight.dtype))
        for index, block in enumerate(self.blocks):
            x = block(x, e0, runs, cos, sin, context, plan=plan, layer=index)
        if output is not None:
            start, stop = output
            if stop <= start:
                return x.new_zeros((x.shape[0], 0, self.head.head.out_features))
            x = x[:, start:stop]
            runs = tuple(
                (max(a, start) - start, min(b, stop) - start, u)
                for a, b, u in runs
                if a < stop and b > start
            )
        return self.head(x, e, runs)


def _shards(directory: Path) -> tuple[Path, ...]:
    shards = tuple(sorted(directory.glob("*.safetensors")))
    if not shards:
        raise FileNotFoundError(f"no .safetensors shards under {directory}")
    return shards


def load_official_weights(model: Ref2VAWanModel, directory: str | Path) -> dict[str, Any]:
    """Strictly load an official Wan2.2 transformer directory; only AnyFlow's delta may lack."""

    from safetensors.torch import load_file

    directory = Path(directory)
    config_path = directory / "config.json"
    if config_path.is_file():
        declared = json.loads(config_path.read_text())
        for key in ("dim", "ffn_dim", "num_heads", "num_layers", "in_dim", "out_dim"):
            if key in declared and int(declared[key]) != int(model.config[key]):
                raise ValueError(
                    f"{config_path} declares {key}={declared[key]}, model has {model.config[key]}"
                )
    expected = set(model.state_dict())
    seen: set[str] = set()
    for shard in _shards(directory):
        state = load_file(str(shard), device="cpu")
        state = {key.removeprefix("model."): value for key, value in state.items()}
        unexpected = sorted(set(state) - expected)
        if unexpected:
            raise ValueError(f"{shard} has keys the Wan Ref2VA model lacks: {unexpected[:8]}")
        model.load_state_dict(state, strict=False)
        seen.update(state)
        del state
    missing = sorted(key for key in expected - seen if not key.startswith("delta_embedding."))
    if missing:
        raise ValueError(f"official Wan weights under {directory} miss {missing[:8]}")
    model.initialize_anyflow_delta_from_time()
    return {"directory": str(directory), "keys": len(seen)}


def build_model(
    family: str,
    *,
    weights: str | Path | None,
    anyflow_gate: float | None = None,
    dtype: Any = torch.bfloat16,
    overrides: Mapping[str, Any] | None = None,
) -> tuple[Ref2VAWanModel, dict[str, Any]]:
    """Build a Wan Ref2VA model in ``dtype`` on CPU; official weights avoid a random init."""

    if weights is None:
        with torch.device("cpu"):
            model = Ref2VAWanModel(family, anyflow_gate=anyflow_gate, **dict(overrides or {}))
        return model.to(dtype=dtype), {"directory": None, "keys": 0}
    with torch.device("meta"):
        model = Ref2VAWanModel(family, anyflow_gate=anyflow_gate, **dict(overrides or {}))
    model = model.to(dtype=dtype).to_empty(device="cpu")
    receipt = load_official_weights(model, weights)
    return model, receipt


__all__ = [
    "ARCHITECTURES",
    "Ref2VAWanBlock",
    "Ref2VAWanModel",
    "build_model",
    "load_official_weights",
    "rotate",
    "time_runs",
]
