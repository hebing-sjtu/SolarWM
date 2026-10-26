"""Token layouts of Wan Ref2VA documents.

A document is a sequence of latent segments, each patchified ``(1, 2, 2)`` in frame-major
raster order: ``[picture_1 .. picture_P | proxy | target]`` for the bidirectional model and
``[picture_1 .. picture_P | proxy | clean target | noisy target]`` for teacher forcing.
Every segment uses its own spatial grid starting at zero. Time is one clock: picture ``i``
sits at ``i``, proxy latent ``f`` at ``P + f`` and target latent ``f`` at
``P + proxy_frames + f``; with ``align_proxy_reference_time`` the proxy shares the target's
times instead. Clean and noisy copies of a latent share its coordinates.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

PICTURE_ROLE = 0
PROXY_ROLE = 1
CLEAN_ROLE = 2
NOISY_ROLE = 3
PAD_ROLE = 4


@dataclass(frozen=True)
class Segment:
    """One latent block of the document, in token units."""

    role: int
    frames: int
    height: int
    width: int
    start: int
    first_frame: int = 0

    @property
    def tokens_per_frame(self) -> int:
        return self.height * self.width

    @property
    def length(self) -> int:
        return self.frames * self.tokens_per_frame

    @property
    def stop(self) -> int:
        return self.start + self.length

    @property
    def slice(self) -> slice:
        return slice(self.start, self.stop)


@dataclass(frozen=True)
class Ref2VAWanLayout:
    segments: tuple[Segment, ...]
    coords: np.ndarray
    roles: np.ndarray
    chunks: np.ndarray
    chunk_latents: int

    @property
    def length(self) -> int:
        return int(self.coords.shape[0])

    def role_segments(self, role: int) -> tuple[Segment, ...]:
        return tuple(segment for segment in self.segments if segment.role == role)

    def only(self, role: int) -> Segment:
        found = self.role_segments(role)
        if len(found) != 1:
            raise ValueError(f"layout has {len(found)} segments of role {role}, expected one")
        return found[0]

    @property
    def condition_length(self) -> int:
        """Tokens of the pictures and the proxy, which lead every document."""

        return sum(
            segment.length
            for segment in self.segments
            if segment.role in (PICTURE_ROLE, PROXY_ROLE)
        )


def _token_grid(latent_height: int, latent_width: int) -> tuple[int, int]:
    if latent_height % 2 or latent_width % 2 or min(latent_height, latent_width) <= 0:
        raise ValueError(
            f"Wan Ref2VA latent grids must be positive and even, got {latent_height}x{latent_width}"
        )
    return latent_height // 2, latent_width // 2


def build_ref2va_layout(
    *,
    picture_latent_sizes: Sequence[tuple[int, int]],
    proxy_latent_size: tuple[int, int],
    proxy_frames: int,
    target_latent_size: tuple[int, int],
    frame_range: tuple[int, int],
    teacher_forcing: bool,
    chunk_latents: int,
    align_proxy_reference_time: bool = False,
    noisy_first: int | None = None,
) -> Ref2VAWanLayout:
    """Lay out pictures, the proxy latents and target latents ``frame_range`` of the timeline.

    ``proxy_frames`` is the cached proxy length; the proxy segment holds those of its latents
    that fall inside ``frame_range``. ``frame_range`` may run past the proxy, as the last
    rollout chunk does. Chunk ids are timeline chunks of ``chunk_latents`` latents.
    With ``noisy_first`` a teacher-forced document holds clean latents ``[first, noisy_first)``
    and noisy latents ``[noisy_first, stop)`` only, which is all a window rollout step reads.
    """

    if not picture_latent_sizes:
        raise ValueError("a Ref2VA document needs at least the anchor picture")
    first, stop = (int(value) for value in frame_range)
    if not 0 <= first < stop:
        raise ValueError(f"invalid Ref2VA frame range {frame_range}")
    if noisy_first is not None and (not teacher_forcing or not first <= int(noisy_first) < stop):
        raise ValueError("noisy_first needs teacher forcing and must lie inside the frame range")
    if int(chunk_latents) <= 0:
        raise ValueError("chunk_latents must be positive")
    if align_proxy_reference_time and proxy_frames < stop:
        raise ValueError("a time-aligned proxy must cover the whole frame range")

    pictures = len(picture_latent_sizes)
    proxy_origin = pictures
    target_origin = pictures if align_proxy_reference_time else pictures + int(proxy_frames)
    proxy_stop = min(stop, int(proxy_frames))

    segments: list[Segment] = []
    coords: list[np.ndarray] = []
    roles: list[np.ndarray] = []
    chunks: list[np.ndarray] = []
    cursor = 0

    def add(
        role: int,
        frames: int,
        grid: tuple[int, int],
        times: np.ndarray,
        ids: np.ndarray,
        first_frame: int,
    ) -> None:
        nonlocal cursor
        height, width = grid
        segment = Segment(role, frames, height, width, cursor, first_frame)
        t = np.repeat(times.astype(np.int64), height * width)
        h = np.tile(np.repeat(np.arange(height, dtype=np.int64), width), frames)
        w = np.tile(np.arange(width, dtype=np.int64), frames * height)
        coords.append(np.stack((t, h, w), axis=1))
        roles.append(np.full(segment.length, role, dtype=np.int64))
        chunks.append(np.repeat(ids.astype(np.int64), height * width))
        segments.append(segment)
        cursor = segment.stop

    for index, size in enumerate(picture_latent_sizes):
        grid = _token_grid(*size)
        add(PICTURE_ROLE, 1, grid, np.array([index]), np.array([-1]), 0)
    if proxy_stop > first:
        frames = np.arange(first, proxy_stop)
        add(
            PROXY_ROLE,
            proxy_stop - first,
            _token_grid(*proxy_latent_size),
            (target_origin if align_proxy_reference_time else proxy_origin) + frames,
            frames // int(chunk_latents),
            first,
        )
    target_grid = _token_grid(*target_latent_size)
    split = first if noisy_first is None else int(noisy_first)
    spans = (
        ((CLEAN_ROLE, first, stop if noisy_first is None else split), (NOISY_ROLE, split, stop))
        if teacher_forcing
        else ((NOISY_ROLE, first, stop),)
    )
    for role, low, high in spans:
        if high > low:
            frames = np.arange(low, high)
            add(
                role,
                high - low,
                target_grid,
                target_origin + frames,
                frames // int(chunk_latents),
                low,
            )

    return Ref2VAWanLayout(
        segments=tuple(segments),
        coords=np.concatenate(coords, axis=0),
        roles=np.concatenate(roles, axis=0),
        chunks=np.concatenate(chunks, axis=0),
        chunk_latents=int(chunk_latents),
    )


def window_allows(
    query_role: int,
    query_chunk: int,
    key_role: int,
    key_chunk: int,
    *,
    window_chunks: int,
    windowed_clean: bool,
) -> bool:
    """The W-chunk causal visibility of H3's Ref2VA Stage1, without text or audio rows.

    Pictures read only pictures. A proxy chunk reads the pictures and the proxy of its own and
    the previous ``W-1`` chunks; every target row reads that same proxy window. Clean rows read
    the clean history up to their own chunk (only the window's with ``windowed_clean``), noisy
    rows the clean chunks of the ``W-1`` previous chunks and their own noisy chunk.
    """

    if query_role == PAD_ROLE or key_role == PAD_ROLE:
        return query_role == key_role == PAD_ROLE
    if query_role == PICTURE_ROLE:
        return key_role == PICTURE_ROLE
    if key_role == PICTURE_ROLE:
        return True
    delta = query_chunk - key_chunk
    if key_role == PROXY_ROLE:
        return 0 <= delta < window_chunks
    if query_role == PROXY_ROLE:
        return False
    if query_role == CLEAN_ROLE:
        return (
            key_role == CLEAN_ROLE
            and delta >= 0
            and not (windowed_clean and delta >= window_chunks)
        )
    if query_role == NOISY_ROLE:
        if key_role == CLEAN_ROLE:
            return 1 <= delta < window_chunks
        return key_role == NOISY_ROLE and delta == 0
    raise ValueError(f"unknown Ref2VA role {query_role}")


def dense_window_mask(
    layout: Ref2VAWanLayout, *, window_chunks: int, windowed_clean: bool
) -> np.ndarray:
    """The full boolean ``[L, L]`` visibility, for tests and small documents."""

    roles, chunks = layout.roles.tolist(), layout.chunks.tolist()
    return np.array(
        [
            [
                window_allows(
                    qr, qc, kr, kc, window_chunks=window_chunks, windowed_clean=windowed_clean
                )
                for kr, kc in zip(roles, chunks, strict=True)
            ]
            for qr, qc in zip(roles, chunks, strict=True)
        ],
        dtype=bool,
    )


__all__ = [
    "CLEAN_ROLE",
    "NOISY_ROLE",
    "PAD_ROLE",
    "PICTURE_ROLE",
    "PROXY_ROLE",
    "Ref2VAWanLayout",
    "Segment",
    "build_ref2va_layout",
    "dense_window_mask",
    "window_allows",
]
