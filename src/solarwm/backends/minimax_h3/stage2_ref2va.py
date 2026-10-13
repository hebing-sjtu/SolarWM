"""Self-gradient forcing (DMD) on Ref2VA proxy documents.

The causal student rolls ``[text | anchor | proxy | audio | target]`` out five latents at a
time, four denoiser evaluations per chunk, reading the conditions and the last five committed
chunks through a key/value cache. A teacher-forced replay of the same document then recomputes
the exit-step predictions with gradients; its clean history is limited to the same six-chunk
window, so replay and rollout see identical keys. Bidirectional Stage0.5 Ref2VA models score
the result. Every row keeps its absolute position in the Stage0.5 Ref2VA document.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .distributed import is_sequence_parallel_enabled
from .layout import patchify_video, unpatchify_video
from .lora_roles import student_role_split
from .mask import AUDIO_ROLE, CONDITION_ROLE, PROXY_VIDEO_ROLE
from .proxy_artifacts import H3ProxyArtifactBatch
from .sgf import h3_sgf_schedule, h3_sgf_score_timesteps
from .sgf_rollout import H3SGFRollout
from .stage0p5_ref2va import _condition_rows
from .stage1_ref2va import CHUNK_LATENTS, KEYFRAME_TIME, WINDOW_CHUNKS, H3Ref2VAStage1Core
from .torch_flow import predict_clean_sample, sample_shifted_timestep, scale_noise
from .torch_mask import _role_chunk_runs


@dataclass
class H3Ref2VAKVCache:
    """Rotated keys/values of this rank's heads: conditions, proxy chunks, recent clean chunks."""

    window_chunks: int = WINDOW_CHUNKS
    common: dict[int, tuple[Any, Any]] = field(default_factory=dict)
    proxy: dict[int, list[tuple[Any, Any]]] = field(default_factory=dict)
    clean: dict[int, list[tuple[int, Any, Any]]] = field(default_factory=dict)

    def _history(self, layer: int, chunk: int) -> list[tuple[int, Any, Any]]:
        first = max(0, chunk - (self.window_chunks - 1))
        history = self.clean.get(layer, [])
        if [entry[0] for entry in history] != list(range(first, chunk)):
            raise RuntimeError(f"H3 Ref2VA KV history differs at layer={layer}, chunk={chunk}")
        return history

    def keys(self, layer: int, chunk: int) -> tuple[Any, Any]:
        """Everything a target row of ``chunk`` reads besides its own chunk."""

        import torch

        if layer not in self.common:
            raise RuntimeError(f"H3 Ref2VA KV cache has no prefill for layer={layer}")
        first = max(0, chunk - (self.window_chunks - 1))
        history = self._history(layer, chunk)
        proxy = self.proxy[layer][first : chunk + 1]
        keys = [self.common[layer][0], *(k for k, _ in proxy), *(entry[1] for entry in history)]
        values = [self.common[layer][1], *(v for _, v in proxy), *(entry[2] for entry in history)]
        return torch.cat(keys, dim=1), torch.cat(values, dim=1)

    def commit(self, layer: int, chunk: int, key: Any, value: Any) -> None:
        import torch

        if torch.is_grad_enabled():
            raise RuntimeError("H3 Ref2VA KV commits are detached rollout state")
        history = self._history(layer, chunk)
        entry = (chunk, key.detach().clone(), value.detach().clone())
        self.clean[layer] = [*history, entry][-(self.window_chunks - 1) :]

    def clear(self) -> None:
        self.common.clear()
        self.proxy.clear()
        self.clean.clear()


@dataclass(frozen=True)
class H3Ref2VARolloutAttention:
    """Attention of one rollout forward on full-sequence, local-head ``[B,S,H,D]`` tensors.

    ``prefill`` runs the condition document ``[text | anchor | proxy | audio]`` under the
    Stage1 visibility and caches its keys. ``step`` and ``commit`` run one target chunk behind a
    single placeholder text row that reads only itself; ``commit`` also caches the chunk.
    """

    native_rotary = True

    mode: str
    cache: H3Ref2VAKVCache
    chunk: int = 0
    condition_rows: int = 0
    proxy_spans: tuple[tuple[tuple[int, int], ...], ...] = ()
    audio_span: tuple[int, int] = (0, 0)

    def attend(self, query: Any, key: Any, value: Any, *, layer: int, attend: Any) -> Any:
        import torch

        if self.mode == "prefill":
            return self._prefill(query, key, value, layer=layer, attend=attend)
        if self.mode not in ("step", "commit"):
            raise ValueError(f"unknown H3 Ref2VA rollout attention mode {self.mode!r}")
        rows_k, rows_v = key[:, 1:], value[:, 1:]
        cached_k, cached_v = self.cache.keys(layer, self.chunk)
        visible_k = torch.cat((cached_k.to(rows_k.dtype), rows_k), dim=1)
        visible_v = torch.cat((cached_v.to(rows_v.dtype), rows_v), dim=1)
        output = torch.cat(
            (
                attend(query[:, :1], key[:, :1], value[:, :1]),
                attend(query[:, 1:], visible_k, visible_v),
            ),
            dim=1,
        )
        if self.mode == "commit":
            self.cache.commit(layer, self.chunk, rows_k, rows_v)
        return output

    def _prefill(self, query: Any, key: Any, value: Any, *, layer: int, attend: Any) -> Any:
        import torch

        def gather(tensor: Any, spans: tuple[tuple[int, int], ...]) -> Any:
            return torch.cat([tensor[:, start:stop] for start, stop in spans], dim=1)

        condition = self.condition_rows
        audio = (self.audio_span,)
        output = torch.zeros_like(query)
        output[:, :condition] = attend(
            query[:, :condition], key[:, :condition], value[:, :condition]
        )
        common_k = torch.cat((key[:, :condition], gather(key, audio)), dim=1)
        common_v = torch.cat((value[:, :condition], gather(value, audio)), dim=1)
        start, stop = self.audio_span
        output[:, start:stop] = attend(query[:, start:stop], common_k, common_v)
        proxy = [(gather(key, spans), gather(value, spans)) for spans in self.proxy_spans]
        for chunk, spans in enumerate(self.proxy_spans):
            recent = proxy[max(0, chunk - (self.cache.window_chunks - 1)) : chunk + 1]
            visible_k = torch.cat((key[:, :condition], *(k for k, _ in recent)), dim=1)
            visible_v = torch.cat((value[:, :condition], *(v for _, v in recent)), dim=1)
            result = attend(gather(query, spans), visible_k, visible_v)
            offset = 0
            for span_start, span_stop in spans:
                rows = span_stop - span_start
                output[:, span_start:span_stop] = result[:, offset : offset + rows]
                offset += rows
        self.cache.common[layer] = (common_k, common_v)
        self.cache.proxy[layer] = proxy
        return output


def prefill_spans(layout: Any) -> tuple[int, tuple[tuple[tuple[int, int], ...], ...], tuple]:
    """Condition-row count, per-chunk proxy spans (one per reference) and the audio span."""

    runs, _ = _role_chunk_runs(layout)
    prefix = int(layout.clean_video_indices[0])
    runs = [run for run in runs if run[0] < prefix]
    if runs[0][2] != CONDITION_ROLE or runs[-1][2] != AUDIO_ROLE or runs[-1][1] != prefix:
        raise ValueError("a Ref2VA rollout prefix is [text | anchor | proxy | audio]")
    proxy: dict[int, list[tuple[int, int]]] = {}
    for start, stop, role, chunk in runs[1:-1]:
        if role != PROXY_VIDEO_ROLE:
            raise ValueError("a Ref2VA rollout prefix has proxy rows between anchor and audio")
        proxy.setdefault(chunk, []).append((start, stop))
    if sorted(proxy) != list(range(len(proxy))):
        raise ValueError("Ref2VA proxy chunks are not contiguous from zero")
    return (
        runs[0][1],
        tuple(tuple(proxy[chunk]) for chunk in range(len(proxy))),
        (runs[-1][0], runs[-1][1]),
    )


@dataclass
class H3Ref2VASGFInputs:
    prompt: Any
    tags: Any
    anchor: Any
    proxy: Any
    audio_clean: Any
    audio: Any
    audio_timestep: Any
    given: int
    given_value: Any


class H3Ref2VASGFCore(H3Ref2VAStage1Core):
    """Rollout, gradient replay and Stage0.5 scoring of the Ref2VA proxy partition."""

    windowed_clean = True

    def __init__(self, model: Any, device: Any, config: Any) -> None:
        super().__init__(model, device, config)
        if student_role_split(model):
            raise ValueError("H3 Ref2VA SGF trains one shared student adapter")
        self.rollout_frames = self.rollout_chunks * CHUNK_LATENTS
        self.video_shift = float(self.train["video_timestep_shift"])
        self.audio_shift = float(self.train["audio_timestep_shift"])

    def _check_train_frames(self) -> None:
        if self.train_frames != self.target_frames:
            raise ValueError(
                f"Ref2VA SGF scores all {self.target_frames} cached latents, "
                f"got train_target_latents={self.train_frames}"
            )

    def random_like(self, value: Any, *, generator: Any = None) -> Any:
        import torch

        noise = torch.randn(
            value.shape, device=value.device, dtype=torch.float32, generator=generator
        )
        self._broadcast(noise)
        return noise

    def prepare_inputs(
        self, batch: H3ProxyArtifactBatch, *, generator: Any = None
    ) -> H3Ref2VASGFInputs:
        import torch

        clean, anchor, proxy, prompt, tags, given = self._inputs(batch)
        aug = float(self.train["keyframe_noise_augmentation"])
        anchor = scale_noise(anchor, self.random_like(anchor, generator=generator), aug)
        proxy = scale_noise(proxy, self.random_like(proxy, generator=generator), aug)
        given_value = None
        if given:
            head = clean[:, :, :given]
            given_value = scale_noise(head, self.random_like(head, generator=generator), aug)
        audio_clean = self._audio_rows(batch)
        audio_time = sample_shifted_timestep(
            1, self.audio_shift, self.device, torch.float32, generator=generator
        ).reshape(())
        self._broadcast(audio_time)
        audio = scale_noise(
            audio_clean, self.random_like(audio_clean, generator=generator), audio_time
        )
        return H3Ref2VASGFInputs(
            prompt, tags, anchor, proxy, audio_clean, audio, audio_time, given, given_value
        )

    def full_window(self, inputs: H3Ref2VASGFInputs) -> tuple[Any, Any]:
        return self.window(
            inputs.tags,
            inputs.anchor,
            inputs.proxy,
            frames=self.rollout_frames,
            chunks=(0, self.rollout_chunks),
        )

    def _rollout_forward(
        self,
        *,
        layout: Any,
        rows: Any,
        index: Any,
        video: Any,
        audio_rows: Any,
        audio: Any,
        prompt: Any,
        text: Any,
        times: Any,
        control: H3Ref2VARolloutAttention,
    ) -> Any:
        import torch

        timestep, inverse = torch.unique(times, sorted=True, return_inverse=True)
        prediction, _ = self.model(
            hidden_states=rows,
            audio_hidden_states=audio_rows,
            encoder_hidden_states=prompt,
            timestep=timestep,
            timestep_indices=inverse,
            token_tags=layout.token_tags.index_select(0, index),
            position_ids=layout.position_ids.index_select(0, index),
            video_indices=video,
            audio_indices=audio,
            text_indices=text,
            sgf_attention=control,
            packed_sequence_parallel=is_sequence_parallel_enabled(),
            return_dict=False,
        )
        return prediction

    def _prefill(self, inputs: H3Ref2VASGFInputs, layout: Any, cache: H3Ref2VAKVCache) -> None:
        import torch

        condition_rows, proxy_spans, audio_span = prefill_spans(layout)
        prefix = int(layout.clean_video_indices[0])
        index = torch.arange(prefix, device=self.device)
        times = torch.ones(prefix, device=self.device, dtype=torch.float32)
        times[layout.condition_video_indices] = KEYFRAME_TIME
        times[layout.audio_indices] = inputs.audio_timestep
        self._rollout_forward(
            layout=layout,
            rows=_condition_rows(inputs.anchor, inputs.proxy),
            index=index,
            video=layout.condition_video_indices,
            audio_rows=inputs.audio,
            audio=layout.audio_indices,
            prompt=inputs.prompt,
            text=layout.text_indices,
            times=times,
            control=H3Ref2VARolloutAttention(
                "prefill",
                cache,
                condition_rows=condition_rows,
                proxy_spans=proxy_spans,
                audio_span=audio_span,
            ),
        )

    def _chunk_forward(
        self,
        inputs: H3Ref2VASGFInputs,
        layout: Any,
        cache: H3Ref2VAKVCache,
        *,
        chunk: int,
        latents: Any,
        timestep: Any,
        fixed: int,
        commit: bool,
    ) -> Any:
        import torch

        rows_per_frame = int(layout.rows_per_video_frame)
        chunk_rows = CHUNK_LATENTS * rows_per_frame
        noisy = layout.noisy_video_indices[chunk * chunk_rows : (chunk + 1) * chunk_rows]
        times = torch.full((1 + chunk_rows,), float(timestep), device=self.device)
        times[0] = 1.0
        if fixed and not commit:
            times[1 : 1 + fixed * rows_per_frame] = KEYFRAME_TIME
        prediction = self._rollout_forward(
            layout=layout,
            rows=patchify_video(latents),
            index=torch.cat((layout.text_indices[:1], noisy)),
            video=torch.arange(1, 1 + chunk_rows, device=self.device),
            audio_rows=inputs.audio[:, :0],
            audio=layout.audio_indices[:0],
            prompt=inputs.prompt[:, :1],
            text=torch.zeros(1, device=self.device, dtype=torch.long),
            times=times,
            control=H3Ref2VARolloutAttention("commit" if commit else "step", cache, chunk=chunk),
        )
        if commit:
            return None
        velocity = unpatchify_video(
            prediction, CHUNK_LATENTS, self.target_height, self.target_width, channels=24
        ).float()
        x0 = predict_clean_sample(latents, velocity, timestep)
        if fixed:
            x0[:, :, :fixed] = inputs.given_value
        return x0

    def rollout(
        self,
        inputs: H3Ref2VASGFInputs,
        *,
        generator: Any = None,
        exit_index: int | None = None,
        progress: Any = None,
    ) -> H3SGFRollout:
        import torch

        schedule, _ = h3_sgf_schedule(video_shift=self.video_shift, device=self.device)
        if exit_index is None:
            selected = torch.randint(0, 4, (), device=self.device, generator=generator)
            self._broadcast(selected)
            exit_index = int(selected)
        if not 0 <= exit_index < 4:
            raise ValueError("H3 SGF exit index must lie in [0,4)")
        layout, _ = self.full_window(inputs)
        shape = (1, 24, self.rollout_frames, self.target_height, self.target_width)
        noise = self.random_like(torch.empty(shape, device=self.device), generator=generator)
        cache = H3Ref2VAKVCache()
        committed, noisy_exits = [], []
        try:
            with torch.no_grad():
                self._prefill(inputs, layout, cache)
                for chunk in range(self.rollout_chunks):
                    fixed = inputs.given if chunk == 0 else 0
                    current = noise[:, :, chunk * CHUNK_LATENTS : (chunk + 1) * CHUNK_LATENTS]
                    if fixed:
                        current = current.clone()
                        current[:, :, :fixed] = inputs.given_value
                    exit_x0 = None
                    # Four forwards on every rank: FSDP collectives need identical counts.
                    for step, timestep in enumerate(schedule.timesteps):
                        if step == exit_index:
                            noisy_exits.append(current.clone())
                        x0 = self._chunk_forward(
                            inputs,
                            layout,
                            cache,
                            chunk=chunk,
                            latents=current,
                            timestep=timestep,
                            fixed=fixed,
                            commit=False,
                        )
                        if not bool(torch.isfinite(x0).all()):
                            raise FloatingPointError(
                                f"non-finite H3 Ref2VA SGF chunk={chunk}, denoise={step}"
                            )
                        if step == exit_index:
                            exit_x0 = x0.clone()
                        if step < 3:
                            current = scale_noise(
                                x0,
                                self.random_like(x0, generator=generator),
                                schedule.timesteps[step + 1],
                            )
                            if fixed:
                                current[:, :, :fixed] = inputs.given_value
                    committed.append(exit_x0)
                    self._chunk_forward(
                        inputs,
                        layout,
                        cache,
                        chunk=chunk,
                        latents=exit_x0,
                        timestep=1.0,
                        fixed=fixed,
                        commit=True,
                    )
                    if progress is not None:
                        progress(chunk + 1)
            return H3SGFRollout(
                torch.cat(committed, dim=2),
                torch.cat(noisy_exits, dim=2),
                exit_index,
                schedule.timesteps[exit_index],
            )
        finally:
            cache.clear()

    def replay(self, inputs: H3Ref2VASGFInputs, rollout: H3SGFRollout) -> Any:
        """Exit-step clean predictions of every chunk, teacher-forced on the rollout."""

        import torch

        layout, mask = self.full_window(inputs)
        timestep = rollout.native_exit_timestep
        times = torch.ones(layout.sequence_length, device=self.device, dtype=torch.float32)
        times[layout.condition_video_indices] = KEYFRAME_TIME
        times[layout.audio_indices] = inputs.audio_timestep
        times[layout.noisy_video_indices] = timestep
        if inputs.given:
            fixed = inputs.given * int(layout.rows_per_video_frame)
            times[layout.noisy_video_indices[:fixed]] = KEYFRAME_TIME
        values, inverse = torch.unique(times, sorted=True, return_inverse=True)
        rows = torch.cat(
            (
                _condition_rows(inputs.anchor, inputs.proxy),
                patchify_video(rollout.cache_target.detach()),
                patchify_video(rollout.noisy_at_exit),
            ),
            dim=1,
        )
        prediction, _ = self.model(
            hidden_states=rows,
            audio_hidden_states=inputs.audio,
            encoder_hidden_states=inputs.prompt,
            timestep=values,
            timestep_indices=inverse,
            attention_mask=mask,
            fused_prope=False,
            packed_sequence_parallel=is_sequence_parallel_enabled(),
            return_dict=False,
            **layout.transformer_kwargs(),
        )
        velocity = unpatchify_video(
            prediction[:, layout.noisy_video_output_slice],
            self.rollout_frames,
            self.target_height,
            self.target_width,
            channels=24,
        )
        return predict_clean_sample(rollout.noisy_at_exit, velocity, timestep)

    def supervised(self, video: Any) -> Any:
        return video[:, :, : self.target_frames]

    def kl_mask(self, inputs: H3Ref2VASGFInputs) -> Any:
        """Score every cached latent except the given ones, which are held fixed."""

        import torch

        mask = torch.ones((1, 1, self.target_frames, 1, 1), device=self.device, dtype=torch.bool)
        mask[:, :, : inputs.given] = False
        return mask

    def score_inputs(self, clean: Any, inputs: H3Ref2VASGFInputs) -> tuple[Any, ...]:
        video_time, audio_time = h3_sgf_score_timesteps(
            video_shift=self.video_shift,
            audio_shift=self.audio_shift,
            min_sigma=float(self.train["score_min_sigma"]),
            max_sigma=float(self.train["score_max_sigma"]),
            device=self.device,
        )
        self._broadcast(video_time, audio_time)
        noise = self.random_like(clean)
        noisy = scale_noise(clean, noise, video_time)
        if inputs.given:
            noisy[:, :, : inputs.given] = inputs.given_value
        audio = scale_noise(inputs.audio_clean, self.random_like(inputs.audio_clean), audio_time)
        return noisy, noise, video_time, audio, audio_time

    def score_forward(
        self,
        model: Any,
        inputs: H3Ref2VASGFInputs,
        noisy: Any,
        video_time: Any,
        audio: Any,
        audio_time: Any,
    ) -> Any:
        """The bidirectional Stage0.5 Ref2VA velocity of all cached target latents."""

        import torch

        layout = self.layout(inputs.tags, inputs.anchor, inputs.proxy)
        given_rows = inputs.given * (self.target_height // 2) * (self.target_width // 2)
        video = self._forward(
            video_rows=torch.cat(
                (_condition_rows(inputs.anchor, inputs.proxy), patchify_video(noisy)), dim=1
            ),
            audio_rows=audio,
            prompt=inputs.prompt,
            layout=layout,
            video_t=video_time,
            audio_t=audio_time,
            fixed_rows=given_rows,
            model=model,
        )
        return unpatchify_video(
            video, self.target_frames, self.target_height, self.target_width, channels=24
        )

    def critic_loss(
        self, velocity: Any, *, noise: Any, clean: Any, inputs: H3Ref2VASGFInputs
    ) -> Any:
        import torch.nn.functional as F

        given = inputs.given
        target = clean[:, :, given:].float() - noise[:, :, given:].float()
        return F.mse_loss(velocity[:, :, given:].float(), target)

    def generate(
        self,
        batch: H3ProxyArtifactBatch,
        *,
        noise_seed: int,
        num_inference_steps: int,
    ) -> Any:
        import torch

        if int(num_inference_steps) != 4:
            raise ValueError("H3 Ref2VA SGF rollout uses four denoiser evaluations per chunk")
        generator = torch.Generator(device=self.device).manual_seed(int(noise_seed))
        with torch.no_grad():
            inputs = self.prepare_inputs(batch, generator=generator)
            rollout = self.rollout(inputs, generator=generator, exit_index=3)
        return self.supervised(rollout.cache_target).contiguous()


__all__ = [
    "H3Ref2VAKVCache",
    "H3Ref2VARolloutAttention",
    "H3Ref2VASGFCore",
    "H3Ref2VASGFInputs",
    "prefill_spans",
]
