"""Long-take rollout: cached 124-frame Ref2VA proxy segments chained on their overlaps.

A take longer than one document is cut by FastVideo's ``split_proxy_take.py`` into 124-frame
segments that start every ``STRIDE_FRAMES`` frames, so neighbours share ``OVERLAP_FRAMES``.
Segment 0 is an ordinary opening document (``w0_omni``: one given latent, the anchor frame).
Every later segment is a continuation document (``wn_omni``) whose first ``GIVEN_LATENTS``
latents are given: the rollout decodes the previous segment, VAE-encodes its last
``OVERLAP_FRAMES`` frames exactly as the cache encoder encodes a clip (39 frames are 12
latents), and hands the first ten to the next segment. All segments keep the take's global
anchor and their own window of the proxy video.

The stitched take is segment 0 up to the frame the next segment takes over, then each later
segment from ``GIVEN_FRAMES`` on: the frames before that only reproduce the previous tail.
"""

from __future__ import annotations

import copy
import dataclasses
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from solarwm.errors import BackendContractError, DataContractError

from .proxy_artifacts import CWM_CONTINUATION_ROLES, CWM_GIVEN_FRAMES

TAKE_SCHEMA = "h3-proxy-take.v1"
SEGMENT_FRAMES = 124
# 39 = 5 + 17 + 17 frames is the shortest clip whose encoding covers ten latents (2 + 5 + 5).
OVERLAP_FRAMES = 39
OVERLAP_LATENTS = 12
GIVEN_LATENTS = 10
# The pixel frames the ten given latents stand for (the ``wn`` prompt's "first 34 frames").
GIVEN_FRAMES = 34
STRIDE_FRAMES = SEGMENT_FRAMES - OVERLAP_FRAMES
KEYFRAME_ENCODE_SEED = 42


@dataclass(frozen=True)
class TakeSegment:
    index: int
    name: str
    path: Path
    start: int
    cwm_system: str


@dataclass(frozen=True)
class Take:
    name: str
    fps: int
    frames: int
    target: str  # "real": the cached targets are the take's own video; else placeholders
    segments: tuple[TakeSegment, ...]
    source: Mapping[str, Any]


def take_frames(num_segments: int) -> int:
    """Stitched length of a take of ``num_segments`` segments."""

    if num_segments < 1:
        raise ValueError("a take has at least one segment")
    return SEGMENT_FRAMES + STRIDE_FRAMES * (num_segments - 1)


def stitch_spans(num_segments: int) -> tuple[tuple[int, int], ...]:
    """Per segment, the ``[first, stop)`` local frames it adds to the stitched take."""

    spans = []
    for index in range(num_segments):
        first = 0 if index == 0 else GIVEN_FRAMES
        stop = SEGMENT_FRAMES if index == num_segments - 1 else STRIDE_FRAMES + GIVEN_FRAMES
        spans.append((first, stop))
    return tuple(spans)


def read_take(path: Path, *, base_role: str, cache_dir: Path | None = None) -> Take:
    """Validate a take manifest against the rollout geometry and the run's CWM role."""

    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise DataContractError(f"cannot read H3 proxy take {path}: {exc}") from exc
    if payload.get("schema") != TAKE_SCHEMA:
        raise DataContractError(f"H3 proxy take {path} is not {TAKE_SCHEMA}")
    geometry = {
        "segment_frames": SEGMENT_FRAMES,
        "overlap_frames": OVERLAP_FRAMES,
        "given_frames": GIVEN_FRAMES,
        "stride_frames": STRIDE_FRAMES,
    }
    for key, expected in geometry.items():
        if payload.get(key) != expected:
            raise DataContractError(f"H3 proxy take {key}={payload.get(key)!r}, expected {expected}")
    role = str(base_role).strip().lower()
    if role not in CWM_CONTINUATION_ROLES:
        raise DataContractError(f"CWM role {role!r} has no continuation role for a long take")
    continuation = CWM_CONTINUATION_ROLES[role]
    if CWM_GIVEN_FRAMES[continuation] != GIVEN_LATENTS:
        raise DataContractError(f"continuation role {continuation!r} does not give ten latents")
    rows = payload.get("segments")
    if not isinstance(rows, list) or not rows:
        raise DataContractError(f"H3 proxy take {path} lists no segments")
    root = Path(cache_dir) if cache_dir is not None else Path(path).parent
    segments = []
    for index, row in enumerate(rows):
        name = str(row.get("name") or "")
        expected_role = role if index == 0 else continuation
        if not name or row.get("start") != index * STRIDE_FRAMES:
            raise DataContractError(
                f"take segment {index} must be named and start at frame {index * STRIDE_FRAMES}"
            )
        if str(row.get("cwm_system") or "").strip().lower() != expected_role:
            raise DataContractError(
                f"take segment {index} must use CWM role {expected_role!r}, "
                f"got {row.get('cwm_system')!r}"
            )
        sample = Path(str(row["pt"])) if row.get("pt") else root / f"{name}.pt"
        if not sample.is_file():
            raise DataContractError(f"take segment {index} sample is missing: {sample}")
        segments.append(TakeSegment(index, name, sample, index * STRIDE_FRAMES, expected_role))
    frames = take_frames(len(segments))
    if payload.get("frames") != frames:
        raise DataContractError(
            f"H3 proxy take frames={payload.get('frames')!r}, but {len(segments)} segments "
            f"stitch to {frames}"
        )
    target = str(payload.get("target") or "placeholder")
    if target not in {"real", "placeholder"}:
        raise DataContractError("H3 proxy take target must be 'real' or 'placeholder'")
    return Take(
        name=str(payload.get("name") or Path(path).stem),
        fps=int(payload.get("fps", 24)),
        frames=frames,
        target=target,
        segments=tuple(segments),
        source=dict(payload.get("source") or {}),
    )


def with_given_latents(batch: Any, given: Any) -> Any:
    """``batch`` whose first target latents are the previous segment's encoded tail."""

    import torch

    head = given.detach().to("cpu", dtype=torch.float32)
    if head.ndim == 5:
        head = head[0]
    target = batch.target_latents.clone()
    count = int(batch.num_given_latent_frames)
    if tuple(head.shape) != (int(target.shape[0]), count, *target.shape[2:]):
        raise BackendContractError(
            f"given latents {tuple(head.shape)} do not fit the target {tuple(target.shape)}"
        )
    target[:, :count] = head
    return dataclasses.replace(batch, target_latents=target.contiguous())


def encode_h3_pixels(video_vae: Any, pixels: Any, *, device: Any) -> Any:
    """``[1, 3, T, H, W]`` pixels in ``[0, 1]`` to normalized latents, as the cache encodes."""

    import torch

    pixel_mean = torch.tensor((0.485, 0.456, 0.406), device=device).view(1, -1, 1, 1, 1)
    pixel_std = torch.tensor((0.229, 0.224, 0.225), device=device).view(1, -1, 1, 1, 1)
    normalized = (pixels.to(device=device, dtype=torch.float32) - pixel_mean) / pixel_std
    with (
        torch.no_grad(),
        torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"),
    ):
        posterior = video_vae.encode(normalized, return_dict=False)[0]
    latent = posterior.sample(generator=torch.Generator("cpu").manual_seed(KEYFRAME_ENCODE_SEED))
    latent = latent.to(torch.float16).float()
    mean = torch.tensor(video_vae.config.latents_mean, device=latent.device).view(1, -1, 1, 1, 1)
    std = torch.tensor(video_vae.config.latents_std, device=latent.device).view(1, -1, 1, 1, 1)
    return ((latent - mean) / std).contiguous()


def _frames_uint8(video: Any) -> Any:
    """``[1, 3, T, H, W]`` in ``[0, 1]`` to a ``[T, H, W, 3]`` uint8 numpy array."""

    import torch

    return video[0].permute(1, 2, 3, 0).mul(255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()


def _proxy_pixels(video_vae: Any, proxy_latents: Any, size: tuple[int, int], *, device: Any) -> Any:
    """The first proxy reference decoded and resized to the generated frame size."""

    import torch.nn.functional as F

    from .inference import decode_h3_latents

    proxies = proxy_latents.unsqueeze(0) if proxy_latents.ndim == 4 else proxy_latents
    decoded = decode_h3_latents(video_vae, proxies[:1], device=device)
    frames = decoded.permute(0, 2, 1, 3, 4).reshape(-1, 3, *decoded.shape[-2:])
    frames = F.interpolate(frames, size=size, mode="bilinear", align_corners=False)
    return frames.reshape(1, -1, 3, *size).permute(0, 2, 1, 3, 4)


class _Mp4:
    def __init__(self, path: Path, fps: int) -> None:
        import imageio.v2 as imageio

        path.parent.mkdir(parents=True, exist_ok=True)
        self.writer = imageio.get_writer(
            str(path),
            fps=int(fps),
            quality=8,
            macro_block_size=1,
            ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
        )

    def extend(self, frames: Any) -> None:
        for frame in frames:
            self.writer.append_data(frame)

    def close(self) -> None:
        self.writer.close()


def run_proxy_rollout(config: Mapping[str, Any]) -> int:
    """Roll one take out on every logical DP rank, each with its own noise seed."""

    import numpy as np

    from .distributed import broadcast_sp_tensor, get_sp_rank
    from .optional import load_conditioners
    from .proxy_artifacts import H3ProxyPtStream
    from .proxy_inference import load_proxy_runtime
    from .runtime import _collective_call, _collective_failures

    inference = config["inference"]
    data = config["data"]
    cache_dir = inference.get("proxy_take_cache")
    take = read_take(
        Path(str(inference["proxy_take"])),
        base_role=str(data["cwm_system"]),
        cache_dir=Path(str(cache_dir)) if cache_dir else None,
    )
    runtime = load_proxy_runtime(config)
    torch, dist, topology, device, core = (
        runtime.torch,
        runtime.dist,
        runtime.topology,
        runtime.device,
        runtime.core,
    )
    core.continuation = True
    reader_config = copy.deepcopy(dict(config))
    reader_config["data"] = {**data, "data_path": str(take.segments[0].path.parent)}
    reader = H3ProxyPtStream(reader_config, topology, continuation=True)
    steps = int(config["validation"]["num_inference_steps"])
    seed = int(config["validation"]["noise_seed"]) + int(topology.dp_rank)
    root = (
        Path(str(config["runtime"]["output_dir"])).resolve()
        / "proxy-rollout"
        / f"dp-rank-{topology.dp_rank:05d}"
    )
    writer_rank = get_sp_rank() == 0
    spans = stitch_spans(len(take.segments))
    given_shape = (1, 24, GIVEN_LATENTS, int(data["latent_height"]), int(data["latent_width"]))

    conditioners = None
    movies: dict[str, _Mp4] = {}
    records: list[dict[str, Any]] = []
    given = None
    try:
        for segment, (first, stop) in zip(take.segments, spans, strict=True):
            batch = _collective_call(
                lambda segment=segment: reader.read(segment.path),
                dist=dist,
                topology=topology,
                label=f"take segment {segment.index} read",
            )
            if batch.cwm_system != segment.cwm_system:
                raise BackendContractError(
                    f"take segment {segment.index} is cached as {batch.cwm_system!r}, "
                    f"the take lists {segment.cwm_system!r}"
                )
            if given is not None:
                batch = with_given_latents(batch, given)
            segment_seed = seed + 1000 * segment.index
            latents = _collective_call(
                lambda batch=batch, segment_seed=segment_seed: core.generate(
                    batch, noise_seed=segment_seed, num_inference_steps=steps
                ),
                dist=dist,
                topology=topology,
                label=f"take segment {segment.index} generation",
            )
            last = segment.index == len(take.segments) - 1
            given = None if last else torch.zeros(given_shape, device=device, dtype=torch.float32)
            local_error = ""
            if writer_rank:
                try:
                    from safetensors.torch import save_file

                    from .inference import _mp4_bytes, decode_h3_latents

                    if conditioners is None:
                        conditioners = load_conditioners(
                            config["model"],
                            device=device,
                            qwen=False,
                            video_vae=True,
                            audio_vae=False,
                            schedulers=False,
                        )
                    vae = conditioners.video_vae
                    generated = decode_h3_latents(vae, latents, device=device)
                    if not last:
                        tail = generated[:, :, STRIDE_FRAMES:SEGMENT_FRAMES]
                        encoded = encode_h3_pixels(vae, tail, device=device)
                        if int(encoded.shape[2]) != OVERLAP_LATENTS:
                            raise BackendContractError(
                                f"{OVERLAP_FRAMES} tail frames encoded to "
                                f"{int(encoded.shape[2])} latents, expected {OVERLAP_LATENTS}"
                            )
                        given.copy_(encoded[:, :, :GIVEN_LATENTS])
                    size = (int(generated.shape[-2]), int(generated.shape[-1]))
                    panels = [_frames_uint8(_proxy_pixels(vae, batch.proxy_latents, size, device=device))]
                    produced = _frames_uint8(generated)
                    panels.append(produced)
                    if take.target == "real":
                        target = batch.target_latents.unsqueeze(0)
                        panels.append(_frames_uint8(decode_h3_latents(vae, target, device=device)))
                    compare = np.concatenate(panels, axis=2)
                    folder = root / "segments" / f"seg-{segment.index:02d}"
                    folder.mkdir(parents=True, exist_ok=True)
                    save_file(
                        {"latents": latents.detach().to("cpu", dtype=torch.bfloat16).contiguous()},
                        str(folder / "generated.safetensors"),
                    )
                    (folder / "generated.mp4").write_bytes(_mp4_bytes(generated, fps=take.fps))
                    segment_movie = _Mp4(folder / "compare.mp4", take.fps)
                    segment_movie.extend(compare)
                    segment_movie.close()
                    if not movies:
                        movies = {
                            "take": _Mp4(root / "take.mp4", take.fps),
                            "compare": _Mp4(root / "take_compare.mp4", take.fps),
                        }
                    movies["take"].extend(produced[first:stop])
                    movies["compare"].extend(compare[first:stop])
                    records.append(
                        {
                            "index": segment.index,
                            "name": segment.name,
                            "cwm_system": segment.cwm_system,
                            "noise_seed": segment_seed,
                            "take_frames": [segment.start + first, segment.start + stop],
                            "segment_frames": [first, stop],
                            "sample": str(segment.path),
                        }
                    )
                    print(
                        f"[h3-proxy-rollout] dp={topology.dp_rank} segment {segment.index + 1}/"
                        f"{len(take.segments)} frames {segment.start + first}..{segment.start + stop}",
                        flush=True,
                    )
                except Exception as exc:
                    local_error = f"rank {topology.raw_rank}: {type(exc).__name__}: {exc}"
            failures = _collective_failures(dist, topology, local_error)
            if failures:
                raise BackendContractError("H3 proxy take rollout failed: " + " | ".join(failures))
            if given is not None:
                broadcast_sp_tensor(given)
    finally:
        for movie in movies.values():
            movie.close()
        reader.close()

    local_error = ""
    if writer_rank:
        try:
            payload = {
                "schema": "solarwm.minimax-h3-proxy-rollout.v1",
                "take": take.name,
                "take_manifest": str(inference["proxy_take"]),
                "frames": take.frames,
                "fps": take.fps,
                "target": take.target,
                "source": dict(take.source),
                "weights_id": runtime.weights_id,
                "weight_source": runtime.weight_source,
                "checkpoint": runtime.checkpoint_path or None,
                "stage": str(config["train"]["stage"]),
                "num_inference_steps": steps,
                "noise_seed": seed,
                "panel_order": ["proxy", "prediction", *(["target"] if take.target == "real" else [])],
                **runtime.generation,
                "segments": records,
            }
            (root / "take-result.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
        except Exception as exc:
            local_error = f"rank {topology.raw_rank}: {type(exc).__name__}: {exc}"
    failures = _collective_failures(dist, topology, local_error)
    if failures:
        raise BackendContractError("H3 proxy take result failed: " + " | ".join(failures))
    return 0


__all__ = [
    "GIVEN_FRAMES",
    "GIVEN_LATENTS",
    "OVERLAP_FRAMES",
    "STRIDE_FRAMES",
    "TAKE_SCHEMA",
    "Take",
    "TakeSegment",
    "encode_h3_pixels",
    "read_take",
    "run_proxy_rollout",
    "stitch_spans",
    "take_frames",
    "with_given_latents",
]
