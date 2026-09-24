"""Ref2VA proxy Stage0.5 forward for FastVideo-compatible 124-frame caches."""

from __future__ import annotations

import hashlib
from typing import Any

from .distributed import broadcast_sp_tensor, is_sequence_parallel_enabled
from .layout import build_row_timesteps, patchify_video
from .proxy_artifacts import H3ProxyArtifactBatch
from .ref2va_layout import build_ref2va_proxy_layout
from .stage0p5 import H3TorchLayout


def _pad_spatial(latents: Any) -> Any:
    import torch.nn.functional as F

    pad_h = (-int(latents.shape[-2])) % 2
    pad_w = (-int(latents.shape[-1])) % 2
    return F.pad(latents, (0, pad_w, 0, pad_h)) if pad_h or pad_w else latents


class H3Ref2VAStage0p5Core:
    """Train the Ref2VA partition on ``[text|anchor|proxy|audio|target]``."""

    def __init__(self, model: Any, device: Any, config: Any) -> None:
        self.model = model
        self.device = device
        self.config = config
        self._layouts: dict[str, H3TorchLayout] = {}

    def layout(self, tags: Any, anchor: Any, proxy: Any) -> H3TorchLayout:
        import torch

        tags_cpu = tags.detach().cpu().long().numpy()
        dimensions = (
            int(anchor.shape[-2]),
            int(anchor.shape[-1]),
            int(proxy.shape[-2]),
            int(proxy.shape[-1]),
        )
        key = hashlib.blake2s(
            tags_cpu.tobytes()
            + repr(dimensions).encode()
            + str(bool(self.config["data"]["align_proxy_reference_time"])).encode()
        ).hexdigest()
        cached = self._layouts.get(key)
        if cached is not None:
            return cached
        source = build_ref2va_proxy_layout(
            tags_cpu,
            anchor_height=dimensions[0],
            anchor_width=dimensions[1],
            proxy_frames=int(proxy.shape[-3]),
            proxy_height=dimensions[2],
            proxy_width=dimensions[3],
            target_frames=37,
            target_height=48,
            target_width=84,
            num_audio_latents=207,
            align_proxy_reference_time=bool(self.config["data"]["align_proxy_reference_time"]),
        )
        layout = H3TorchLayout(
            source=source,
            position_ids=torch.from_numpy(source.position_ids).to(self.device),
            token_tags=torch.from_numpy(source.token_tags).long().to(self.device),
            video_indices=torch.from_numpy(source.video_indices).long().to(self.device),
            audio_indices=torch.from_numpy(source.audio_indices).long().to(self.device),
            text_indices=torch.from_numpy(source.text_indices).long().to(self.device),
            camera_video_indices=torch.empty(0, dtype=torch.long, device=self.device),
            camera_frame_ids=torch.empty(0, dtype=torch.long, device=self.device),
        )
        self._layouts[key] = layout
        while len(self._layouts) > 4:
            self._layouts.pop(next(iter(self._layouts)))
        return layout

    @staticmethod
    def shifted_timestep(generator: Any | None, *, shift: float, device: Any) -> Any:
        import torch

        sigma = torch.rand((), generator=generator, device=device, dtype=torch.float32)
        shifted = float(shift) * sigma / (1.0 + (float(shift) - 1.0) * sigma)
        return 1.0 - shifted

    @staticmethod
    def _broadcast(*values: Any) -> None:
        if is_sequence_parallel_enabled():
            for value in values:
                broadcast_sp_tensor(value)

    def _audio_rows(self, batch: H3ProxyArtifactBatch) -> Any:
        import torch

        if batch.audio_latents is None:
            audio = torch.zeros((1, 2, 32, 207), device=self.device, dtype=torch.float32)
        else:
            audio = batch.audio_latents.to(
                self.device, dtype=torch.float32, non_blocking=True
            ).unsqueeze(0)
            if int(audio.shape[-1]) < 207:
                raise RuntimeError("H3 proxy audio_latents has fewer than 207 positions")
            audio = audio[..., :207]
        return audio.permute(0, 1, 3, 2).reshape(1, -1, 32).contiguous()

    def _forward(
        self,
        *,
        video_rows: Any,
        audio_rows: Any,
        prompt: Any,
        layout: H3TorchLayout,
        video_t: Any,
        audio_t: Any,
        fixed_rows: int,
    ) -> Any:
        import torch

        times, inverse = build_row_timesteps(
            layout.source,
            float(video_t.item()),
            float(audio_t.item()),
            condition_video_timestep=max(float(video_t.item()), 0.999),
            num_fixed_video_rows=int(fixed_rows),
        )
        video, _audio = self.model(
            hidden_states=video_rows,
            audio_hidden_states=audio_rows,
            encoder_hidden_states=prompt,
            timestep=torch.from_numpy(times).to(self.device, dtype=torch.float32),
            timestep_indices=torch.from_numpy(inverse).to(self.device, dtype=torch.long),
            token_tags=layout.token_tags,
            position_ids=layout.position_ids,
            video_indices=layout.video_indices,
            audio_indices=layout.audio_indices,
            text_indices=layout.text_indices,
            attention_mask=None,
            fused_prope=False,
            stage0p5_sequence_parallel=is_sequence_parallel_enabled(),
            return_dict=False,
        )
        return video[:, layout.target_video_output_slice]

    def forward_loss(self, batch: H3ProxyArtifactBatch, *, noise_seed: int | None) -> Any:
        import torch
        import torch.nn.functional as F

        expected_role = str(self.config["data"]["cwm_system"])
        if batch.cwm_system != expected_role:
            raise RuntimeError(
                f"H3 proxy CWM role changed inside one run: {batch.cwm_system!r} "
                f"!= {expected_role!r}"
            )
        expected_given = int(self.config["data"]["num_given_latent_frames"])
        if batch.num_given_latent_frames != expected_given:
            raise RuntimeError("H3 proxy given-frame count differs from cached CWM role")
        generator = (
            None
            if noise_seed is None
            else torch.Generator(device=self.device).manual_seed(int(noise_seed))
        )
        clean = batch.target_latents.to(
            self.device, dtype=torch.float32, non_blocking=True
        ).unsqueeze(0)
        anchor = _pad_spatial(
            batch.anchor_latents.to(self.device, dtype=torch.float32, non_blocking=True).unsqueeze(
                0
            )
        )
        proxy = _pad_spatial(
            batch.proxy_latents.to(self.device, dtype=torch.float32, non_blocking=True).unsqueeze(0)
        )
        prompt = batch.prompt_embeds.to(
            self.device, dtype=torch.bfloat16, non_blocking=True
        ).unsqueeze(0)
        tags = batch.text_token_tags.to(self.device, dtype=torch.long)
        self._broadcast(clean, anchor, proxy, prompt, tags)
        layout = self.layout(tags, anchor, proxy)

        condition_rows = torch.cat((patchify_video(anchor), patchify_video(proxy)), dim=1)
        condition_noise = torch.randn(
            condition_rows.shape,
            generator=generator,
            device=self.device,
            dtype=torch.float32,
        )
        aug = float(self.config["train"]["keyframe_noise_augmentation"])
        condition_rows = aug * condition_rows + (1.0 - aug) * condition_noise

        video_noise = torch.randn(
            clean.shape, generator=generator, device=self.device, dtype=torch.float32
        )
        video_t = self.shifted_timestep(
            generator,
            shift=float(self.config["train"]["video_timestep_shift"]),
            device=self.device,
        )
        video_noisy = video_t * clean + (1.0 - video_t) * video_noise
        given = int(batch.num_given_latent_frames)
        if given:
            video_noisy[:, :, :given] = (
                aug * clean[:, :, :given] + (1.0 - aug) * video_noise[:, :, :given]
            )
            video_noise[:, :, :given] = clean[:, :, :given]

        audio_clean = self._audio_rows(batch)
        audio_noise = torch.randn(
            audio_clean.shape, generator=generator, device=self.device, dtype=torch.float32
        )
        audio_t = self.shifted_timestep(
            generator,
            shift=float(self.config["train"]["audio_timestep_shift"]),
            device=self.device,
        )
        audio_noisy = audio_t * audio_clean + (1.0 - audio_t) * audio_noise
        self._broadcast(
            condition_noise,
            video_noise,
            video_t,
            audio_noise,
            audio_t,
        )

        target_rows = patchify_video(clean - video_noise)
        fixed_rows = given * layout.source.rows_per_video_frame
        prediction = self._forward(
            video_rows=torch.cat((condition_rows, patchify_video(video_noisy)), dim=1),
            audio_rows=audio_noisy,
            prompt=prompt,
            layout=layout,
            video_t=video_t,
            audio_t=audio_t,
            fixed_rows=fixed_rows,
        )
        if given:
            prediction = torch.cat(
                (
                    torch.zeros_like(prediction[:, :fixed_rows]),
                    prediction[:, fixed_rows:],
                ),
                dim=1,
            )
        if tuple(prediction.shape) != tuple(target_rows.shape):
            raise RuntimeError(
                f"H3 proxy predicted rows {tuple(prediction.shape)} != "
                f"target rows {tuple(target_rows.shape)}"
            )
        return F.mse_loss(prediction.float(), target_rows.float())


__all__ = ["H3Ref2VAStage0p5Core"]
