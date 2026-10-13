"""Ref2VA proxy packed-row layout, ported from FastVideo's Apache-2.0 H3 path."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import frame_position_grid, temporal_position_grid
from .layout import (
    AUDIO_ROLE,
    AUDIO_TAG,
    CLEAN_VIDEO_ROLE,
    CONDITION_ROLE,
    NOISY_VIDEO_ROLE,
    PROXY_VIDEO_ROLE,
    VIDEO_TAG,
    H3PackedLayout,
)


@dataclass(frozen=True)
class H3ReferenceGeometry:
    media_type: str
    num_latent_frames: int
    latent_height: int
    latent_width: int
    time_aligned: bool = False


def _indices(start: int, stop: int) -> np.ndarray:
    return np.arange(start, stop, dtype=np.int64)


def _empty() -> np.ndarray:
    return np.empty((0,), dtype=np.int64)


def _padded(value: int, patch: int) -> int:
    return int(value) + (-int(value)) % int(patch)


def _temporal_span(num_latent_frames: int) -> float:
    return float(
        sum((5.0 / 3.0) * (1, 4, 4, 4, 4)[index % 5] for index in range(int(num_latent_frames)))
    )


def _clock_advance(reference: H3ReferenceGeometry) -> float:
    return 1.0 if reference.media_type == "image" else _temporal_span(reference.num_latent_frames)


def _frame_grid(
    height: int, width: int, patch_h: int, patch_w: int
) -> tuple[np.ndarray, np.ndarray]:
    return frame_position_grid(height, width, patch_h, patch_w)


def _strided_indices(target: int, reference: int) -> np.ndarray:
    if target <= 0 or reference <= 0:
        raise ValueError("Ref2VA strided grid dimensions must be positive")
    if reference == 1:
        return np.zeros((1,), dtype=np.int64)
    stride = max(1, round(target / reference))
    return np.minimum(np.arange(reference, dtype=np.int64) * stride, target - 1)


def _aligned_frame_grid(
    reference_height: int,
    reference_width: int,
    target_height: int,
    target_width: int,
    patch_h: int,
    patch_w: int,
) -> tuple[np.ndarray, np.ndarray]:
    target_grid, target_width_axis = _frame_grid(target_height, target_width, patch_h, patch_w)
    ref_h, ref_w = reference_height // patch_h, reference_width // patch_w
    tgt_h, tgt_w = target_height // patch_h, target_width // patch_w
    h_indices = _strided_indices(tgt_h, ref_h)
    w_indices = _strided_indices(tgt_w, ref_w)
    sampled = target_grid.reshape(tgt_h, tgt_w, 2)[h_indices][:, w_indices]
    return sampled.reshape(-1, 2), target_width_axis[w_indices]


def _fill_audio(
    position_ids: np.ndarray,
    rows: slice,
    num_audio_latents: int,
    rotary_time: float,
    width_axis: np.ndarray,
) -> None:
    times = rotary_time + np.arange(num_audio_latents, dtype=np.float64)
    position_ids[rows, 0] = np.tile(times, 2)
    position_ids[rows, 2] = np.concatenate(
        (
            np.full(num_audio_latents, width_axis[0], dtype=np.float64),
            np.full(num_audio_latents, width_axis[-1], dtype=np.float64),
        )
    )


def build_ref2va_proxy_layout(
    text_token_tags: object,
    *,
    anchor_height: int,
    anchor_width: int,
    proxy_frames: int = 37,
    proxy_height: int = 12,
    proxy_width: int = 21,
    target_frames: int = 37,
    target_height: int = 48,
    target_width: int = 84,
    num_audio_latents: int = 207,
    align_proxy_reference_time: bool = False,
    patch_size: tuple[int, int, int] = (1, 2, 2),
    num_proxy_references: int = 1,
) -> H3PackedLayout:
    """Build ``[text | anchor | proxy_1 .. proxy_N | target audio | target video]``.

    Every proxy video shares one latent geometry. Each advances the reference clock by its own
    temporal span, which is how H3 orders ``<Video 1>``, ``<Video 2>``... in Ref2VA.
    """

    tags = np.asarray(text_token_tags, dtype=np.int64)
    if tags.ndim != 1 or not np.isin(tags, (0, 1)).all():
        raise ValueError("Ref2VA text_token_tags must be one-dimensional 0/1 values")
    if patch_size != (1, 2, 2):
        raise ValueError("MiniMax-H3 Ref2VA requires patch_size=(1,2,2)")
    if proxy_frames != target_frames and align_proxy_reference_time:
        raise ValueError("time-aligned proxy and target must have equal latent frame counts")
    if min(anchor_height, anchor_width, proxy_height, proxy_width) <= 0:
        raise ValueError("Ref2VA reference latent grids must be positive")
    if int(num_proxy_references) < 1:
        raise ValueError("Ref2VA proxy layout needs at least one proxy video reference")

    _, patch_h, patch_w = patch_size
    anchor_h, anchor_w = _padded(anchor_height, patch_h), _padded(anchor_width, patch_w)
    proxy_h, proxy_w = _padded(proxy_height, patch_h), _padded(proxy_width, patch_w)
    proxy_reference = H3ReferenceGeometry(
        "video",
        proxy_frames,
        proxy_h,
        proxy_w,
        time_aligned=bool(align_proxy_reference_time),
    )
    references = (
        H3ReferenceGeometry("image", 1, anchor_h, anchor_w),
        *((proxy_reference,) * int(num_proxy_references)),
    )
    target_rows_per_frame = (target_height // patch_h) * (target_width // patch_w)
    reference_rows = (anchor_h // patch_h) * (anchor_w // patch_w) + int(
        num_proxy_references
    ) * proxy_frames * (proxy_h // patch_h) * (proxy_w // patch_w)
    num_text = int(tags.size)
    num_audio_rows = int(num_audio_latents) * 2
    num_target_rows = int(target_frames) * target_rows_per_frame
    audio_start = num_text + reference_rows
    target_start = audio_start + num_audio_rows
    sequence_length = target_start + num_target_rows

    position_ids = np.zeros((sequence_length, 3), dtype=np.float64)
    position_ids[:num_text, 0] = np.arange(num_text, dtype=np.float64)
    target_grid, target_width_axis = _frame_grid(target_height, target_width, patch_h, patch_w)
    target_origin = float(num_text)
    for reference in references:
        target_origin += _clock_advance(reference)

    cursor = num_text
    rotary_time = float(num_text)
    condition_parts: list[np.ndarray] = []
    for reference in references:
        rows_per_frame = (reference.latent_height // patch_h) * (reference.latent_width // patch_w)
        count = reference.num_latent_frames * rows_per_frame
        rows = slice(cursor, cursor + count)
        condition_parts.append(_indices(rows.start, rows.stop))
        if reference.time_aligned:
            frame_grid, _ = _aligned_frame_grid(
                reference.latent_height,
                reference.latent_width,
                target_height,
                target_width,
                patch_h,
                patch_w,
            )
            times = temporal_position_grid(reference.num_latent_frames, target_origin)
        else:
            frame_grid, _ = _frame_grid(
                reference.latent_height,
                reference.latent_width,
                patch_h,
                patch_w,
            )
            if reference.media_type == "image":
                times = np.asarray([rotary_time], dtype=np.float64)
            else:
                times = temporal_position_grid(reference.num_latent_frames, rotary_time)
        position_ids[rows, 0] = np.repeat(times, rows_per_frame)
        position_ids[rows, 1:] = np.tile(frame_grid, (reference.num_latent_frames, 1))
        cursor = rows.stop
        rotary_time += _clock_advance(reference)

    if rotary_time != target_origin:
        raise AssertionError("Ref2VA sequential clock drifted from its prepass")
    _fill_audio(
        position_ids,
        slice(audio_start, target_start),
        num_audio_latents,
        target_origin,
        target_width_axis,
    )
    target_times = temporal_position_grid(target_frames, target_origin)
    position_ids[target_start:, 0] = np.repeat(target_times, target_rows_per_frame)
    position_ids[target_start:, 1:] = np.tile(target_grid, (target_frames, 1))

    text_indices = _indices(0, num_text)
    condition_video_indices = np.concatenate(condition_parts)
    audio_indices = _indices(audio_start, target_start)
    target_video_indices = _indices(target_start, sequence_length)
    video_indices = np.concatenate((condition_video_indices, target_video_indices))
    token_tags = np.empty(sequence_length, dtype=np.int64)
    token_tags[text_indices] = tags
    token_tags[video_indices] = VIDEO_TAG
    token_tags[audio_indices] = AUDIO_TAG
    roles = np.full(sequence_length, CONDITION_ROLE, dtype=np.int64)
    roles[audio_indices] = AUDIO_ROLE
    roles[target_video_indices] = NOISY_VIDEO_ROLE
    return H3PackedLayout(
        position_ids=position_ids,
        token_tags=token_tags,
        video_indices=video_indices,
        audio_indices=audio_indices,
        text_indices=text_indices,
        condition_indices=np.concatenate((text_indices, condition_video_indices)),
        target_indices=np.concatenate((audio_indices, target_video_indices)),
        condition_video_indices=condition_video_indices,
        audio_condition_indices=_empty(),
        clean_video_indices=_empty(),
        noisy_video_indices=target_video_indices,
        target_video_indices=target_video_indices,
        target_audio_indices=audio_indices,
        camera_video_indices=_empty(),
        camera_frame_ids=_empty(),
        row_roles=roles,
        target_video_chunk_ids=np.full(sequence_length, -1, dtype=np.int64),
        num_condition_video_rows=int(condition_video_indices.size),
        num_condition_audio_rows=0,
        num_clean_video_rows=0,
        num_noisy_video_rows=int(target_video_indices.size),
        rows_per_video_frame=target_rows_per_frame,
        latent_height=target_height,
        latent_width=target_width,
        patch_size=patch_size,
        stage="stage0p5",
    )


def build_ref2va_stage1_layout(
    text_token_tags: object,
    *,
    anchor_height: int,
    anchor_width: int,
    proxy_height: int,
    proxy_width: int,
    proxy_frames: int = 37,
    num_latent_frames: int = 35,
    target_height: int = 48,
    target_width: int = 84,
    num_audio_latents: int = 207,
    align_proxy_reference_time: bool = False,
    num_proxy_references: int = 1,
    chunk_latent_frames: int = 5,
    window_chunks: int = 6,
    chunks: tuple[int, int] | None = None,
) -> H3PackedLayout:
    """Build ``[text | anchor | proxy | audio | clean target | noisy target]`` for Stage1.

    Every position is the Stage0.5 Ref2VA document's for a ``proxy_frames``-latent proxy, so
    a Stage0.5 LoRA keeps its geometry; target positions continue past the cached length when
    ``num_latent_frames`` (whole chunks) is longer. Proxy rows are condition video at the fixed
    keyframe time, chunked like the target, the last chunk shorter when the proxy ends inside
    it. ``chunks=(first, stop)`` keeps only those chunks at their absolute positions, as one
    rollout step sees them; chunk ids are local to the kept window.
    """

    if chunk_latent_frames <= 0 or window_chunks <= 0:
        raise ValueError("Ref2VA Stage1 chunk and window sizes must be positive")
    if num_latent_frames <= 0 or num_latent_frames % chunk_latent_frames:
        raise ValueError(
            f"Ref2VA Stage1 length {num_latent_frames} is not whole "
            f"{chunk_latent_frames}-latent chunks"
        )
    total_chunks = num_latent_frames // chunk_latent_frames
    first, stop = (0, total_chunks) if chunks is None else (int(chunks[0]), int(chunks[1]))
    if not 0 <= first < stop <= total_chunks:
        raise ValueError(f"Ref2VA Stage1 window {chunks} is outside [0, {total_chunks}]")
    first_frame, stop_frame = first * chunk_latent_frames, stop * chunk_latent_frames
    if first_frame >= proxy_frames:
        raise ValueError(f"Ref2VA Stage1 window {chunks} starts after the proxy ends")
    full = build_ref2va_proxy_layout(
        text_token_tags,
        anchor_height=anchor_height,
        anchor_width=anchor_width,
        proxy_frames=proxy_frames,
        proxy_height=proxy_height,
        proxy_width=proxy_width,
        target_frames=proxy_frames,
        target_height=target_height,
        target_width=target_width,
        num_audio_latents=num_audio_latents,
        align_proxy_reference_time=align_proxy_reference_time,
        num_proxy_references=num_proxy_references,
    )
    num_text = int(full.text_indices.size)
    anchor_rows = (_padded(anchor_height, 2) // 2) * (_padded(anchor_width, 2) // 2)
    proxy_frame_rows = (_padded(proxy_height, 2) // 2) * (_padded(proxy_width, 2) // 2)
    target_frame_rows = int(full.rows_per_video_frame)
    proxy_stop_frame = min(stop_frame, proxy_frames)

    proxy_start = num_text + anchor_rows
    proxy_source = np.concatenate(
        [
            _indices(
                proxy_start + (index * proxy_frames + first_frame) * proxy_frame_rows,
                proxy_start + (index * proxy_frames + proxy_stop_frame) * proxy_frame_rows,
            )
            for index in range(int(num_proxy_references))
        ]
    )
    target_rows = full.target_video_indices
    target_times = temporal_position_grid(stop_frame, float(full.position_ids[target_rows[0], 0]))
    target_positions = np.empty((stop_frame - first_frame, target_frame_rows, 3), np.float64)
    target_positions[:, :, 0] = target_times[first_frame:, None]
    target_positions[:, :, 1:] = full.position_ids[target_rows[:target_frame_rows], 1:][None]
    target_positions = target_positions.reshape(-1, 3)
    prefix = np.concatenate((_indices(0, proxy_start), proxy_source, full.audio_indices))
    num_target = int(target_positions.shape[0])
    audio_start = proxy_start + int(proxy_source.size)
    clean_start = audio_start + int(full.audio_indices.size)
    noisy_start = clean_start + num_target
    sequence_length = noisy_start + num_target

    text_indices = _indices(0, num_text)
    proxy_indices = _indices(proxy_start, audio_start)
    audio_indices = _indices(audio_start, clean_start)
    clean_indices = _indices(clean_start, noisy_start)
    noisy_indices = _indices(noisy_start, sequence_length)
    condition_video_indices = _indices(num_text, audio_start)
    roles = np.full(sequence_length, CONDITION_ROLE, dtype=np.int64)
    roles[proxy_indices] = PROXY_VIDEO_ROLE
    roles[audio_indices] = AUDIO_ROLE
    roles[clean_indices] = CLEAN_VIDEO_ROLE
    roles[noisy_indices] = NOISY_VIDEO_ROLE
    local_chunks = np.arange(stop_frame - first_frame, dtype=np.int64) // chunk_latent_frames
    chunk_ids = np.full(sequence_length, -1, dtype=np.int64)
    chunk_ids[proxy_indices] = np.tile(
        np.repeat(local_chunks[: proxy_stop_frame - first_frame], proxy_frame_rows),
        int(num_proxy_references),
    )
    chunk_ids[clean_indices] = np.repeat(local_chunks, target_frame_rows)
    chunk_ids[noisy_indices] = chunk_ids[clean_indices]
    token_tags = np.full(sequence_length, VIDEO_TAG, dtype=np.int64)
    token_tags[: prefix.size] = full.token_tags[prefix]
    return H3PackedLayout(
        position_ids=np.concatenate(
            (full.position_ids[prefix], target_positions, target_positions), axis=0
        ),
        token_tags=token_tags,
        video_indices=np.concatenate((condition_video_indices, clean_indices, noisy_indices)),
        audio_indices=audio_indices,
        text_indices=text_indices,
        condition_indices=np.concatenate((text_indices, condition_video_indices)),
        target_indices=np.concatenate((audio_indices, noisy_indices)),
        condition_video_indices=condition_video_indices,
        audio_condition_indices=_empty(),
        clean_video_indices=clean_indices,
        noisy_video_indices=noisy_indices,
        target_video_indices=noisy_indices,
        target_audio_indices=audio_indices,
        camera_video_indices=_empty(),
        camera_frame_ids=_empty(),
        row_roles=roles,
        target_video_chunk_ids=chunk_ids,
        num_condition_video_rows=int(condition_video_indices.size),
        num_condition_audio_rows=0,
        num_clean_video_rows=num_target,
        num_noisy_video_rows=num_target,
        rows_per_video_frame=target_frame_rows,
        latent_height=target_height,
        latent_width=target_width,
        patch_size=(1, 2, 2),
        stage="stage1",
        window_chunks=window_chunks,
        chunk_latent_frames=chunk_latent_frames,
    )


__all__ = ["H3ReferenceGeometry", "build_ref2va_proxy_layout", "build_ref2va_stage1_layout"]
