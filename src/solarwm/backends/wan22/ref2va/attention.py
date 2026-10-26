"""Attention plans for the Wan Ref2VA core: dense, W-chunk masked, and cached rollout.

The core calls ``plan.attend(q, k, v, layer=i)`` with rotated ``[B, L, N, D]`` tensors.
``None`` is full bidirectional attention. :class:`WindowMask` runs FlexAttention under the
Stage1 visibility of :func:`layout.window_allows`, built tile by tile from the document's
role/chunk runs so no ``L x L`` mask is ever materialized. :class:`RolloutAttention` reads and
writes the key/value cache of a chunked causal rollout.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .layout import (
    CLEAN_ROLE,
    NOISY_ROLE,
    PAD_ROLE,
    PICTURE_ROLE,
    PROXY_ROLE,
    Ref2VAWanLayout,
    window_allows,
)

TILE = 128
_COMPILED_FLEX: Any = None


def dense_attention(query: Any, key: Any, value: Any) -> Any:
    """Unmasked attention on ``[B, L, N, D]``; FlashAttention when available."""

    import torch

    if query.is_cuda and query.dtype in (torch.float16, torch.bfloat16):
        try:
            from flash_attn import flash_attn_func

            return flash_attn_func(query, key, value)
        except ImportError:
            pass
    output = torch.nn.functional.scaled_dot_product_attention(
        query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)
    )
    return output.transpose(1, 2)


def _flex() -> Any:
    global _COMPILED_FLEX
    if _COMPILED_FLEX is None:
        import torch
        import torch._dynamo
        from torch.nn.attention.flex_attention import flex_attention

        # One static compile per document length: training, replay and every window rollout size.
        torch._dynamo.config.cache_size_limit = max(int(torch._dynamo.config.cache_size_limit), 64)
        _COMPILED_FLEX = torch.compile(flex_attention, dynamic=False)
    return _COMPILED_FLEX


def _runs(values: np.ndarray) -> list[tuple[int, int, int]]:
    """``(start, stop, value)`` runs of equal consecutive values."""

    edges = np.flatnonzero(np.diff(values)) + 1
    starts = np.concatenate(([0], edges))
    stops = np.concatenate((edges, [values.size]))
    return [(int(a), int(b), int(values[a])) for a, b in zip(starts, stops, strict=True)]


@dataclass
class WindowMask:
    """A FlexAttention ``BlockMask`` over a padded Ref2VA document."""

    length: int
    padded_length: int
    block_mask: Any
    roles: Any
    chunks: Any

    def attend(self, query: Any, key: Any, value: Any, *, layer: int) -> Any:
        del layer
        import torch

        pad = self.padded_length - self.length
        if pad:
            zeros = query.new_zeros((query.shape[0], pad, *query.shape[2:]))
            query, key, value = (
                torch.cat((tensor, zeros), dim=1) for tensor in (query, key, value)
            )
        output = _flex()(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            block_mask=self.block_mask,
        )
        return output.transpose(1, 2)[:, : self.length]


def build_window_mask(
    layout: Ref2VAWanLayout,
    *,
    window_chunks: int,
    windowed_clean: bool,
    device: Any,
) -> WindowMask:
    """Tile-classify the document from its role/chunk runs, then attach the exact ``mask_mod``."""

    import torch
    from torch.nn.attention.flex_attention import BlockMask

    length = layout.length
    padded = -(-length // TILE) * TILE
    roles = np.concatenate((layout.roles, np.full(padded - length, PAD_ROLE, dtype=np.int64)))
    chunks = np.concatenate((layout.chunks, np.full(padded - length, -1, dtype=np.int64)))
    pairs, pair_ids = np.unique(np.stack((roles, chunks), axis=1), axis=0, return_inverse=True)
    pair_ids = pair_ids.reshape(-1)
    allowed = np.array(
        [
            [
                window_allows(
                    int(qr),
                    int(qc),
                    int(kr),
                    int(kc),
                    window_chunks=window_chunks,
                    windowed_clean=windowed_clean,
                )
                for kr, kc in pairs
            ]
            for qr, qc in pairs
        ],
        dtype=np.float64,
    )
    tiles = padded // TILE
    presence = np.zeros((tiles, len(pairs)), dtype=np.float64)
    for start, stop, pair in _runs(pair_ids):
        for tile in range(start // TILE, (stop - 1) // TILE + 1):
            presence[tile, pair] = 1.0
    allowed_count = presence @ allowed @ presence.T
    total = presence.sum(1)[:, None] * presence.sum(1)[None, :]
    full = allowed_count == total
    partial = (allowed_count > 0) & ~full

    def indices(selected: np.ndarray) -> tuple[Any, Any]:
        counts = selected.sum(1).astype(np.int32)
        order = np.argsort(~selected, axis=1, kind="stable").astype(np.int32)
        return (
            torch.from_numpy(counts).to(device)[None, None],
            torch.from_numpy(order).to(device)[None, None],
        )

    roles_t = torch.from_numpy(roles).to(device)
    chunks_t = torch.from_numpy(chunks).to(device)
    window = int(window_chunks)

    def mask_mod(_batch: Any, _head: Any, q: Any, k: Any) -> Any:
        qr, qc, kr, kc = roles_t[q], chunks_t[q], roles_t[k], chunks_t[k]
        delta = qc - kc
        pad = (qr == PAD_ROLE) & (kr == PAD_ROLE)
        real = (qr != PAD_ROLE) & (kr != PAD_ROLE)
        picture_key = kr == PICTURE_ROLE
        proxy_window = (kr == PROXY_ROLE) & (delta >= 0) & (delta < window)
        clean_history = (kr == CLEAN_ROLE) & (delta >= 0)
        if windowed_clean:
            clean_history = clean_history & (delta < window)
        allowed_ = (
            ((qr == PICTURE_ROLE) & picture_key)
            | ((qr == PROXY_ROLE) & (picture_key | proxy_window))
            | ((qr == CLEAN_ROLE) & (picture_key | proxy_window | clean_history))
            | (
                (qr == NOISY_ROLE)
                & (
                    picture_key
                    | proxy_window
                    | ((kr == CLEAN_ROLE) & (delta >= 1) & (delta < window))
                    | ((kr == NOISY_ROLE) & (delta == 0))
                )
            )
        )
        return pad | (real & allowed_)

    kv_num, kv_indices = indices(partial)
    full_num, full_indices = indices(full)
    block_mask = BlockMask.from_kv_blocks(
        kv_num, kv_indices, full_num, full_indices, BLOCK_SIZE=TILE, mask_mod=mask_mod
    )
    return WindowMask(length, padded, block_mask, roles_t, chunks_t)


@dataclass
class RolloutCache:
    """Rotated keys/values of one rollout: pictures, proxy chunks and the recent clean chunks."""

    window_chunks: int
    pictures: dict[int, tuple[Any, Any]] = field(default_factory=dict)
    proxy: dict[int, list[tuple[Any, Any]]] = field(default_factory=dict)
    clean: dict[int, list[tuple[int, Any, Any]]] = field(default_factory=dict)

    def visible(self, layer: int, chunk: int) -> tuple[Any, Any]:
        """Everything a target row of ``chunk`` reads besides its own chunk."""

        import torch

        first = max(0, chunk - (self.window_chunks - 1))
        history = self.clean.get(layer, [])
        if [entry[0] for entry in history] != list(range(first, chunk)):
            raise RuntimeError(f"Wan Ref2VA KV history differs at layer={layer}, chunk={chunk}")
        proxy = self.proxy[layer][first : chunk + 1]
        keys = [self.pictures[layer][0], *(k for k, _ in proxy), *(entry[1] for entry in history)]
        values = [self.pictures[layer][1], *(v for _, v in proxy), *(entry[2] for entry in history)]
        return torch.cat(keys, dim=1), torch.cat(values, dim=1)

    def commit(self, layer: int, chunk: int, key: Any, value: Any) -> None:
        first = max(0, chunk - (self.window_chunks - 1))
        history = [entry for entry in self.clean.get(layer, []) if entry[0] >= first]
        entry = (chunk, key.detach().clone(), value.detach().clone())
        self.clean[layer] = (
            [*history, entry][-(self.window_chunks - 1) :] if self.window_chunks > 1 else []
        )

    def clear(self) -> None:
        self.pictures.clear()
        self.proxy.clear()
        self.clean.clear()


@dataclass(frozen=True)
class RolloutAttention:
    """One forward of a cached rollout.

    ``prefill`` runs ``[pictures | proxy]`` of the whole timeline under the window visibility
    and caches it. ``step`` runs one target chunk against the cache; ``commit`` does the same
    and also caches the chunk as clean history.
    """

    mode: str
    cache: RolloutCache
    chunk: int = 0
    picture_tokens: int = 0
    proxy_chunk_spans: tuple[tuple[int, int], ...] = ()

    def attend(self, query: Any, key: Any, value: Any, *, layer: int) -> Any:
        import torch

        if self.mode == "prefill":
            pictures = self.picture_tokens
            output = torch.zeros_like(query)
            output[:, :pictures] = dense_attention(
                query[:, :pictures], key[:, :pictures], value[:, :pictures]
            )
            proxy = [(key[:, a:b], value[:, a:b]) for a, b in self.proxy_chunk_spans]
            window = self.cache.window_chunks
            for chunk, (start, stop) in enumerate(self.proxy_chunk_spans):
                recent = proxy[max(0, chunk - (window - 1)) : chunk + 1]
                visible_k = torch.cat((key[:, :pictures], *(k for k, _ in recent)), dim=1)
                visible_v = torch.cat((value[:, :pictures], *(v for _, v in recent)), dim=1)
                output[:, start:stop] = dense_attention(query[:, start:stop], visible_k, visible_v)
            self.cache.pictures[layer] = (key[:, :pictures].detach(), value[:, :pictures].detach())
            self.cache.proxy[layer] = [(k.detach(), v.detach()) for k, v in proxy]
            return output
        if self.mode not in ("step", "commit"):
            raise ValueError(f"unknown Wan Ref2VA rollout attention mode {self.mode!r}")
        cached_k, cached_v = self.cache.visible(layer, self.chunk)
        visible_k = torch.cat((cached_k.to(key.dtype), key), dim=1)
        visible_v = torch.cat((cached_v.to(value.dtype), value), dim=1)
        output = dense_attention(query, visible_k, visible_v)
        if self.mode == "commit":
            self.cache.commit(layer, self.chunk, key, value)
        return output


def proxy_chunk_spans(layout: Ref2VAWanLayout) -> tuple[tuple[int, int], ...]:
    """Token spans of each timeline chunk inside the proxy segment, from chunk zero."""

    proxy = layout.only(PROXY_ROLE)
    if proxy.first_frame != 0:
        raise ValueError("a rollout prefill needs the proxy from latent zero")
    per = proxy.tokens_per_frame
    spans = []
    for first in range(0, proxy.frames, layout.chunk_latents):
        last = min(first + layout.chunk_latents, proxy.frames)
        spans.append((proxy.start + first * per, proxy.start + last * per))
    return tuple(spans)


__all__ = [
    "TILE",
    "RolloutAttention",
    "RolloutCache",
    "WindowMask",
    "build_window_mask",
    "dense_attention",
    "proxy_chunk_spans",
]
