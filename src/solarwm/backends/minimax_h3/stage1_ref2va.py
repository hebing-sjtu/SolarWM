"""Teacher-forced W6 AnyFlow on Ref2VA proxy documents, and its chunked NFE4 rollout.

The document is ``[text | anchor | proxy | audio | clean target | noisy target]`` at the
positions of the Stage0.5 Ref2VA document, so a Stage0.5 proxy LoRA initializes it directly.
Training covers the cached target's whole five-latent chunks; the rollout adds one more chunk
so the decoded video has the cached ``5n+2`` latent length.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

from solarwm.training.anyflow import apply_timestep_shift, sample_anyflow_time_pairs

from .anyflow_loss import h3_anyflow_v15_loss
from .distributed import get_dp_group, get_dp_rank, get_dp_world_size, is_sequence_parallel_enabled
from .layout import patchify_video, unpatchify_video
from .proxy_artifacts import H3ProxyArtifactBatch, batch_given_frames
from .ref2va_layout import build_ref2va_stage1_layout
from .stage0p5_ref2va import (
    H3Ref2VAStage0p5Core,
    _condition_rows,
    _images,
    _pad_spatial,
    _picture_inputs,
    _proxy_videos,
)
from .torch_flow import make_shifted_schedule, scale_noise
from .torch_mask import build_ref2va_stage1_block_mask

KEYFRAME_TIME = 0.999
CHUNK_LATENTS = 5
WINDOW_CHUNKS = 6


def given_in_chunk(given: int, chunk: int) -> tuple[int, int]:
    """``(start, count)`` of the given latents inside ``chunk``, in timeline latents."""

    start = int(chunk) * CHUNK_LATENTS
    return start, min(max(int(given) - start, 0), CHUNK_LATENTS)


def ref2va_row_time_pairs(
    layout: Any,
    *,
    video_t: Any,
    video_r: Any,
    audio_t: Any,
    current_chunk: int | None,
    given_rows: int,
    device: Any,
    given_start_row: int = 0,
) -> tuple[Any, Any, Any]:
    """Distinct native ``(t, r)`` pairs and each row's index into them.

    Anchor and proxy rows keep the keyframe time, text and clean history ``t=1``, audio its
    own time, all with ``r=t``. Noisy rows of ``current_chunk`` (every chunk when ``None``)
    carry the interval, except ``given_rows`` noisy rows from ``given_start_row`` on: CWM
    given latents held at the keyframe time.
    """

    import torch

    t = torch.ones(layout.sequence_length, dtype=torch.float32, device=device)
    t[layout.condition_video_indices] = KEYFRAME_TIME
    t[layout.audio_indices] = torch.as_tensor(audio_t, dtype=torch.float32, device=device).reshape(
        ()
    )
    r = t.clone()
    noisy = layout.noisy_video_indices
    if current_chunk is not None:
        chunks = layout.target_video_chunk_ids.index_select(0, noisy)
        noisy = noisy[chunks == int(current_chunk)]
    t[noisy] = torch.as_tensor(video_t, dtype=torch.float32, device=device).reshape(())
    r[noisy] = torch.as_tensor(video_r, dtype=torch.float32, device=device).reshape(())
    if given_rows:
        start = int(given_start_row)
        fixed = layout.noisy_video_indices[start : start + int(given_rows)]
        t[fixed] = KEYFRAME_TIME
        r[fixed] = KEYFRAME_TIME
    pairs, inverse = torch.unique(
        torch.stack((t, r), dim=1), dim=0, sorted=True, return_inverse=True
    )
    return pairs[:, 0].contiguous(), pairs[:, 1].contiguous(), inverse


class H3Ref2VAStage1Core(H3Ref2VAStage0p5Core):
    """Train and roll out the Ref2VA partition causally, five latents per chunk."""

    windowed_clean = False

    def __init__(self, model: Any, device: Any, config: Any) -> None:
        super().__init__(model, device, config)
        self.train = config["train"]
        self.train_frames = int(config["data"]["train_target_latents"])
        self.rollout_chunks = -(-self.target_frames // CHUNK_LATENTS)
        self._windows: OrderedDict[tuple[Any, ...], tuple[Any, Any]] = OrderedDict()
        self._check_train_frames()

    def _check_train_frames(self) -> None:
        if self.train_frames % CHUNK_LATENTS or self.train_frames > self.target_frames:
            raise ValueError(
                f"Ref2VA Stage1 trains whole {CHUNK_LATENTS}-latent chunks of the cached "
                f"{self.target_frames} latents, got train_target_latents={self.train_frames}"
            )

    def window(
        self, tags: Any, anchor: Any, proxy: Any, *, frames: int, chunks: tuple[int, int]
    ) -> tuple[Any, Any]:
        """The layout and W6 ``BlockMask`` of ``chunks`` on a ``frames``-latent timeline."""

        anchor, *extra = _images(anchor)
        extra_sizes = tuple((int(image.shape[-2]), int(image.shape[-1])) for image in extra)
        key = (
            tags.detach().cpu().long().numpy().tobytes(),
            tuple(int(value) for value in anchor.shape[-2:]),
            extra_sizes,
            tuple(int(value) for value in proxy.shape[-3:]),
            int(proxy.shape[0]),
            int(frames),
            tuple(chunks),
        )
        cached = self._windows.get(key)
        if cached is not None:
            self._windows.move_to_end(key)
            return cached
        layout = build_ref2va_stage1_layout(
            tags.detach().cpu().long().numpy(),
            anchor_height=int(anchor.shape[-2]),
            anchor_width=int(anchor.shape[-1]),
            proxy_frames=int(proxy.shape[-3]),
            proxy_height=int(proxy.shape[-2]),
            proxy_width=int(proxy.shape[-1]),
            num_latent_frames=int(frames),
            target_height=self.target_height,
            target_width=self.target_width,
            num_audio_latents=207,
            align_proxy_reference_time=bool(self.config["data"]["align_proxy_reference_time"]),
            num_proxy_references=int(proxy.shape[0]),
            chunk_latent_frames=CHUNK_LATENTS,
            window_chunks=WINDOW_CHUNKS,
            chunks=chunks,
            extra_image_sizes=extra_sizes,
        ).to(self.device)
        cached = (
            layout,
            build_ref2va_stage1_block_mask(
                layout, device=self.device, windowed_clean=self.windowed_clean
            ),
        )
        self._windows[key] = cached
        while len(self._windows) > 2 * WINDOW_CHUNKS:
            self._windows.popitem(last=False)
        return cached

    def _inputs(self, batch: H3ProxyArtifactBatch) -> tuple[Any, ...]:
        import torch

        given = batch_given_frames(self.config["data"], batch, continuation=self.continuation)
        clean = batch.target_latents.to(
            self.device, dtype=torch.float32, non_blocking=True
        ).unsqueeze(0)
        pictures = _picture_inputs(batch, self.device)
        proxy = _pad_spatial(
            _proxy_videos(
                batch.proxy_latents.to(self.device, dtype=torch.float32, non_blocking=True)
            )
        )
        if int(clean.shape[2]) != self.target_frames or int(proxy.shape[-3]) != self.target_frames:
            raise RuntimeError(
                f"Ref2VA Stage1 needs {self.target_frames}-latent target and proxy caches, got "
                f"{int(clean.shape[2])} and {int(proxy.shape[-3])}"
            )
        prompt = batch.prompt_embeds.to(
            self.device, dtype=torch.bfloat16, non_blocking=True
        ).unsqueeze(0)
        tags = batch.text_token_tags.to(self.device, dtype=torch.long)
        self._broadcast(clean, *pictures, proxy, prompt, tags)
        return clean, pictures, proxy, prompt, tags, given

    def _model(
        self,
        layout: Any,
        mask: Any,
        *,
        rows: Any,
        audio: Any,
        prompt: Any,
        times: tuple[Any, Any, Any],
    ) -> Any:
        timestep, r_timestep, indices = times
        prediction, _ = self.model(
            hidden_states=rows,
            audio_hidden_states=audio,
            encoder_hidden_states=prompt,
            timestep=timestep,
            r_timestep=r_timestep,
            timestep_indices=indices,
            attention_mask=mask,
            fused_prope=False,
            packed_sequence_parallel=is_sequence_parallel_enabled(),
            return_dict=False,
            **layout.transformer_kwargs(),
        )
        return prediction[:, layout.noisy_video_output_slice]

    def forward_loss(self, batch: H3ProxyArtifactBatch, *, noise_seed: int | None) -> Any:
        import torch

        generator = (
            None
            if noise_seed is None
            else torch.Generator(device=self.device).manual_seed(int(noise_seed))
        )

        def noise_like(value: Any) -> Any:
            return torch.randn(
                value.shape, generator=generator, device=self.device, dtype=torch.float32
            )

        clean, pictures, proxy, prompt, tags, given = self._inputs(batch)
        clean = clean[:, :, : self.train_frames].contiguous()
        layout, mask = self.window(
            tags,
            pictures,
            proxy,
            frames=self.train_frames,
            chunks=(0, self.train_frames // CHUNK_LATENTS),
        )
        aug = float(self.train["keyframe_noise_augmentation"])
        condition = _condition_rows(pictures, proxy[:, :, : self.train_frames])
        condition_noise = noise_like(condition)
        audio_clean = self._audio_rows(batch)
        audio_noise = noise_like(audio_clean)
        audio_t = self.shifted_timestep(
            generator, shift=float(self.train["audio_timestep_shift"]), device=self.device
        )
        noise = noise_like(clean)
        pairs = sample_anyflow_time_pairs(
            1,
            logical_dp_rank=get_dp_rank(),
            logical_dp_world_size=get_dp_world_size(),
            diffusion_ratio=float(self.train["diffusion_ratio"]),
            consistency_ratio=float(self.train["consistency_ratio"]),
            generator=generator,
            device=self.device,
        )
        self._broadcast(condition_noise, audio_noise, audio_t, noise, *pairs)
        condition = aug * condition + (1.0 - aug) * condition_noise
        audio = audio_t * audio_clean + (1.0 - audio_t) * audio_noise
        shift = float(self.train["video_timestep_shift"])
        sigma_t = apply_timestep_shift(pairs.t, shift)
        sigma_r = apply_timestep_shift(pairs.r, shift)
        clean_rows, noise_rows = patchify_video(clean), patchify_video(noise)
        given_rows = given * int(layout.rows_per_video_frame)
        given_input = aug * clean_rows[:, :given_rows] + (1.0 - aug) * noise_rows[:, :given_rows]
        # A given latent's data-ward velocity is exactly clean - noise, so its AnyFlow residual
        # is zero, as its flow-matching residual is in Stage0.5.
        given_velocity = clean_rows[:, :given_rows] - noise_rows[:, :given_rows]

        def velocity(noisy: Any, video_t: Any, video_r: Any) -> Any:
            if given_rows:
                noisy = torch.cat((given_input.to(noisy.dtype), noisy[:, given_rows:]), dim=1)
            prediction = self._model(
                layout,
                mask,
                rows=torch.cat((condition, clean_rows, noisy), dim=1),
                audio=audio,
                prompt=prompt,
                times=ref2va_row_time_pairs(
                    layout,
                    video_t=video_t,
                    video_r=video_r,
                    audio_t=audio_t,
                    current_chunk=None,
                    given_rows=given_rows,
                    device=self.device,
                ),
            )
            if given_rows:
                prediction = torch.cat(
                    (given_velocity.to(prediction.dtype), prediction[:, given_rows:]), dim=1
                )
            return prediction

        return h3_anyflow_v15_loss(
            velocity,
            clean_rows,
            noise_rows,
            sigma_t,
            sigma_r,
            pairs.is_diffusion,
            epsilon=float(self.train["finite_difference_epsilon"]),
            shift=shift,
            process_group=get_dp_group(),
        )

    def generate(
        self,
        batch: H3ProxyArtifactBatch,
        *,
        noise_seed: int,
        num_inference_steps: int,
    ) -> Any:
        """Roll the cached document out chunk by chunk with four flow-map steps per chunk."""

        import torch

        if int(num_inference_steps) != 4:
            raise ValueError("Ref2VA Stage1 rollout uses NFE4 per chunk")
        generator = torch.Generator(device=self.device).manual_seed(int(noise_seed))

        def noise_like(value: Any) -> Any:
            noise = torch.randn(
                value.shape, generator=generator, device=self.device, dtype=torch.float32
            )
            self._broadcast(noise)
            return noise

        clean, pictures, proxy, prompt, tags, given = self._inputs(batch)
        aug = float(self.train["keyframe_noise_augmentation"])
        anchor_rows = torch.cat(
            [patchify_video(scale_noise(image, noise_like(image), aug)) for image in pictures],
            dim=1,
        )
        proxy = scale_noise(proxy, noise_like(proxy), aug)
        audio_clean = self._audio_rows(batch)
        audio_noise = noise_like(audio_clean)
        audio_schedule = make_shifted_schedule(
            int(num_inference_steps) + 1,
            shift=float(self.train["audio_timestep_shift"]),
            device=self.device,
        )
        sigmas = torch.linspace(
            1.0, 0.0, int(num_inference_steps) + 1, device=self.device, dtype=torch.float32
        )
        frames = self.rollout_chunks * CHUNK_LATENTS
        rows_per_frame = (self.target_height // 2) * (self.target_width // 2)
        chunk_shape = (1, 24, CHUNK_LATENTS, self.target_height, self.target_width)
        given_value = None
        if given:
            given_noise = noise_like(clean[:, :, :given])
            given_value = aug * clean[:, :, :given] + (1.0 - aug) * given_noise
        generated: list[Any] = []
        with torch.no_grad():
            for chunk in range(self.rollout_chunks):
                low, fixed = given_in_chunk(given, chunk)
                held = given_value[:, :, low : low + fixed] if fixed else None
                if fixed == CHUNK_LATENTS:
                    # Wholly given: it is history as it stands, and every rank skips it alike.
                    generated.append(held.contiguous())
                    continue
                first = max(0, chunk - (WINDOW_CHUNKS - 1))
                layout, mask = self.window(
                    tags, pictures, proxy, frames=frames, chunks=(first, chunk + 1)
                )
                start, stop = first * CHUNK_LATENTS, (chunk + 1) * CHUNK_LATENTS
                condition = torch.cat(
                    (
                        anchor_rows,
                        *(
                            patchify_video(proxy[index : index + 1, :, start:stop])
                            for index in range(int(proxy.shape[0]))
                        ),
                    ),
                    dim=1,
                )
                placeholder = torch.zeros(chunk_shape, device=self.device, dtype=torch.float32)
                history = patchify_video(torch.cat((*generated[first:chunk], placeholder), dim=2))
                noisy = torch.zeros_like(history)
                current = noise_like(placeholder)
                if fixed:
                    current[:, :, :fixed] = held
                for step in range(int(num_inference_steps)):
                    noisy[:, -CHUNK_LATENTS * rows_per_frame :] = patchify_video(current)
                    audio_t = audio_schedule.timesteps[step]
                    prediction = self._model(
                        layout,
                        mask,
                        rows=torch.cat((condition, history, noisy), dim=1),
                        audio=scale_noise(audio_clean, audio_noise, audio_t),
                        prompt=prompt,
                        times=ref2va_row_time_pairs(
                            layout,
                            video_t=1.0 - sigmas[step],
                            video_r=1.0 - sigmas[step + 1],
                            audio_t=audio_t,
                            current_chunk=chunk - first,
                            given_rows=fixed * rows_per_frame,
                            given_start_row=(chunk - first) * CHUNK_LATENTS * rows_per_frame,
                            device=self.device,
                        ),
                    )
                    velocity = unpatchify_video(
                        prediction[:, -CHUNK_LATENTS * rows_per_frame :],
                        CHUNK_LATENTS,
                        self.target_height,
                        self.target_width,
                        channels=24,
                    ).float()
                    current = current + (sigmas[step] - sigmas[step + 1]) * velocity
                    if fixed:
                        current[:, :, :fixed] = held
                if not bool(torch.isfinite(current).all().item()):
                    raise FloatingPointError(
                        f"H3 Ref2VA Stage1 rollout chunk {chunk} is non-finite"
                    )
                generated.append(current.contiguous())
        return torch.cat(generated, dim=2)[:, :, : self.target_frames].contiguous()


__all__ = ["H3Ref2VAStage1Core", "given_in_chunk", "ref2va_row_time_pairs"]
