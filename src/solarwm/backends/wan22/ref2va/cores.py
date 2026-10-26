"""Stage cores of Wan Ref2VA: bidirectional flow matching, W-chunk AnyFlow, and SGF (DMD).

Each core turns one cache sample into model documents and back, mirroring the H3 Ref2VA cores
in Wan's noise-ward convention (see :mod:`.objectives`):

* ``Ref2VAWanBDCore`` (Stage0.5): ``[pictures | proxy | target]`` with full attention.
* ``Ref2VAWanARCore`` (Stage1): teacher-forced ``[pictures | proxy | clean | noisy]`` under the
  W-chunk visibility, AnyFlow v1.5, and a per-chunk window-recompute NFE4 rollout.
* ``Ref2VAWanSGFCore`` (Stage2): a key/value-cached four-step rollout, its teacher-forced
  replay with gradients (clean history windowed so replay and rollout read the same keys), and
  Stage0.5 scoring by a frozen teacher and a trained critic.

References and given latents are keyframes: ``aug * clean + (1 - aug) * noise`` at timestep
``1000 * KEYFRAME_SIGMA``. Given latents are held fixed and carry no loss.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from .attention import (
    RolloutAttention,
    RolloutCache,
    WindowMask,
    build_window_mask,
    proxy_chunk_spans,
)
from .data import Ref2VAWanSample, Ref2VAWanSpec
from .layout import (
    CLEAN_ROLE,
    NOISY_ROLE,
    PICTURE_ROLE,
    PROXY_ROLE,
    Ref2VAWanLayout,
    Segment,
    build_ref2va_layout,
)
from .objectives import (
    KEYFRAME_SIGMA,
    NUM_TRAIN_TIMESTEPS,
    add_noise,
    keyframe,
    predict_x0,
    sample_training_sigma,
    sgf_score_sigma,
    shifted_sigmas,
    wan_anyflow_v15_loss,
)

KEYFRAME_TIME = KEYFRAME_SIGMA * NUM_TRAIN_TIMESTEPS


def unpatchify(tokens: Any, segment: Segment, channels: int) -> Any:
    """``[B, F*h*w, 4*C]`` head tokens of ``segment`` to ``[B, C, F, 2h, 2w]`` latents."""

    b = tokens.shape[0]
    x = tokens.view(b, segment.frames, segment.height, segment.width, 1, 2, 2, channels)
    x = torch.einsum("bfhwpqrc->bcfphqwr", x)
    return x.reshape(b, channels, segment.frames, segment.height * 2, segment.width * 2)


def _pad_even(latent: Any) -> Any:
    pad_h, pad_w = (-int(latent.shape[-2])) % 2, (-int(latent.shape[-1])) % 2
    return F.pad(latent, (0, pad_w, 0, pad_h)) if pad_h or pad_w else latent


def dp_rank_world() -> tuple[int, int]:
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def dp_group() -> Any:
    import torch.distributed as dist

    return dist.group.WORLD if dist.is_available() and dist.is_initialized() else None


@dataclass
class Ref2VAWanInputs:
    """One prepared document: keyframed references, the clean target and its conditions."""

    pictures: tuple[Any, ...]  # [1, C, 1, h, w] keyframes, the anchor first
    proxy: Any  # [1, C, T, h, w] keyframes
    prompt: Any  # [1, n, 4096] bf16
    clean: Any  # [1, C, T, H, W]
    i2v_y: Any | None  # [1, 20, R, H, W] A14B mask + first-frame latent, padded to the rollout
    given: int
    given_value: Any | None  # [1, C, given, H, W] keyframes


@dataclass
class Ref2VAWanRollout:
    cache_target: Any  # committed clean chunks [1, C, R, H, W]
    noisy_at_exit: Any  # each chunk's input at its exit step [1, C, R, H, W]
    exit_index: int
    exit_sigma: Any


class Ref2VAWanBDCore:
    """Stage0.5: bidirectional flow matching on ``[pictures | proxy | target]``."""

    windowed_clean = False

    def __init__(self, model: Any, device: Any, config: Mapping[str, Any]) -> None:
        self.model = model
        self.device = torch.device(device)
        self.config = config
        self.spec = Ref2VAWanSpec.from_config(config)
        self.train = config["train"]
        self.shift = float(self.train.get("timestep_shift", 5.0))
        self.reference_y = str(config["model"].get("reference_i2v_y", "self"))
        if self.reference_y not in ("self", "zeros"):
            raise ValueError("model.reference_i2v_y must be 'self' or 'zeros'")
        self._layouts: OrderedDict[tuple[Any, ...], tuple[Ref2VAWanLayout, Any]] = OrderedDict()
        self._masks: OrderedDict[tuple[Any, ...], WindowMask] = OrderedDict()

    # ----------------------------------------------------------------------------- documents

    def generator(self, seed: int | None) -> Any:
        return None if seed is None else torch.Generator(device=self.device).manual_seed(int(seed))

    def noise_like(self, value: Any, generator: Any) -> Any:
        return torch.randn(
            value.shape, generator=generator, device=self.device, dtype=torch.float32
        )

    def prepare(self, sample: Ref2VAWanSample, generator: Any) -> Ref2VAWanInputs:
        spec = self.spec
        to = {"device": self.device, "dtype": torch.float32, "non_blocking": True}
        pictures = []
        for picture in sample.pictures:
            latent = _pad_even(picture.to(**to).unsqueeze(0))
            pictures.append(keyframe(latent, self.noise_like(latent, generator)))
        proxy = _pad_even(sample.proxy.to(**to).unsqueeze(0))
        proxy = keyframe(proxy, self.noise_like(proxy, generator))
        clean = sample.target.to(**to).unsqueeze(0)
        given = spec.given_latents
        given_value = None
        if given:
            head = clean[:, :, :given]
            given_value = keyframe(head, self.noise_like(head, generator))
        i2v_y = None
        if spec.i2v_y:
            frames = max(spec.rollout_latents, spec.latent_frames)
            grid = (frames, spec.latent_height, spec.latent_width)
            latent = torch.zeros((1, 16, *grid), device=self.device, dtype=torch.float32)
            latent[:, :, : spec.latent_frames] = sample.i2v_latent.to(**to).unsqueeze(0)
            mask = torch.zeros((1, 4, *grid), device=self.device, dtype=torch.float32)
            mask[:, :, 0] = 1.0
            i2v_y = torch.cat((mask, latent), dim=1)
        prompt = sample.prompt.to(self.device, dtype=torch.bfloat16, non_blocking=True).unsqueeze(0)
        return Ref2VAWanInputs(tuple(pictures), proxy, prompt, clean, i2v_y, given, given_value)

    def layout(
        self,
        inputs: Ref2VAWanInputs,
        *,
        frame_range: tuple[int, int],
        teacher_forcing: bool,
        noisy_first: int | None = None,
    ) -> tuple[Ref2VAWanLayout, Any]:
        """The cached layout and its device ``[L, 3]`` RoPE coordinates."""

        sizes = tuple((int(p.shape[-2]), int(p.shape[-1])) for p in inputs.pictures)
        proxy = (int(inputs.proxy.shape[-2]), int(inputs.proxy.shape[-1]))
        key = (
            sizes,
            proxy,
            int(inputs.proxy.shape[2]),
            tuple(frame_range),
            teacher_forcing,
            noisy_first,
        )
        cached = self._layouts.get(key)
        if cached is not None:
            self._layouts.move_to_end(key)
            return cached
        layout = build_ref2va_layout(
            picture_latent_sizes=sizes,
            proxy_latent_size=proxy,
            proxy_frames=int(inputs.proxy.shape[2]),
            target_latent_size=(self.spec.latent_height, self.spec.latent_width),
            frame_range=frame_range,
            teacher_forcing=teacher_forcing,
            chunk_latents=self.spec.chunk_latents,
            align_proxy_reference_time=self.spec.align_proxy_reference_time,
            noisy_first=noisy_first,
        )
        cached = (layout, torch.from_numpy(layout.coords).to(self.device))
        self._layouts[key] = cached
        while len(self._layouts) > 4 * self.spec.window_chunks:
            self._layouts.popitem(last=False)
        return cached

    def mask(self, layout: Ref2VAWanLayout, *, windowed_clean: bool) -> WindowMask:
        key = (layout.length, layout.roles.tobytes(), layout.chunks.tobytes(), windowed_clean)
        cached = self._masks.get(key)
        if cached is None:
            cached = build_window_mask(
                layout,
                window_chunks=self.spec.window_chunks,
                windowed_clean=windowed_clean,
                device=self.device,
            )
            self._masks[key] = cached
            while len(self._masks) > 2 * self.spec.window_chunks + 2:
                self._masks.popitem(last=False)
        else:
            self._masks.move_to_end(key)
        return cached

    def release_caches(self) -> None:
        self._layouts.clear()
        self._masks.clear()

    def _reference_input(self, latent: Any) -> Any:
        if not self.spec.i2v_y:
            return latent
        y = torch.cat((torch.ones_like(latent[:, :4]), latent), dim=1)
        if self.reference_y == "zeros":
            y = torch.zeros_like(y)
        return torch.cat((latent, y), dim=1)

    def _target_input(self, latent: Any, inputs: Ref2VAWanInputs, first_frame: int) -> Any:
        if not self.spec.i2v_y:
            return latent
        frames = int(latent.shape[2])
        return torch.cat((latent, inputs.i2v_y[:, :, first_frame : first_frame + frames]), dim=1)

    def segment_tensors(
        self,
        layout: Ref2VAWanLayout,
        inputs: Ref2VAWanInputs,
        *,
        clean: Any = None,
        noisy: Any = None,
        roles: Sequence[int] | None = None,
    ) -> list[Any]:
        """Model inputs of the layout's segments; ``clean``/``noisy`` hold exactly their frames."""

        tensors = []
        picture = 0
        for segment in layout.segments:
            if roles is not None and segment.role not in roles:
                continue
            if segment.role == PICTURE_ROLE:
                tensors.append(self._reference_input(inputs.pictures[picture]))
                picture += 1
                continue
            if segment.role == PROXY_ROLE:
                frames = inputs.proxy[
                    :, :, segment.first_frame : segment.first_frame + segment.frames
                ]
                tensors.append(self._reference_input(frames))
                continue
            value = clean if segment.role == CLEAN_ROLE else noisy
            if value is None or int(value.shape[2]) != segment.frames:
                got = None if value is None else int(value.shape[2])
                raise ValueError(
                    f"segment role {segment.role} needs {segment.frames} latents, got {got}"
                )
            tensors.append(self._target_input(value, inputs, segment.first_frame))
        return tensors

    def token_times(
        self,
        layout: Ref2VAWanLayout,
        *,
        condition: Any,
        noisy: Any,
        clean: Any = 0.0,
        given: int = 0,
        given_time: Any = KEYFRAME_TIME,
    ) -> Any:
        """Raw ``[L]`` timesteps; noisy rows of the first ``given`` latents get ``given_time``."""

        times = torch.empty(layout.length, device=self.device, dtype=torch.float32)
        values = {
            PICTURE_ROLE: condition,
            PROXY_ROLE: condition,
            CLEAN_ROLE: clean,
            NOISY_ROLE: noisy,
        }
        for segment in layout.segments:
            times[segment.slice] = torch.as_tensor(
                values[segment.role], device=self.device
            ).reshape(())
            if segment.role == NOISY_ROLE and given > segment.first_frame:
                held = min(given - segment.first_frame, segment.frames) * segment.tokens_per_frame
                times[segment.start : segment.start + held] = torch.as_tensor(
                    given_time, device=self.device
                ).reshape(())
        return times

    def run(
        self,
        model: Any,
        layout: Ref2VAWanLayout,
        coords: Any,
        tensors: Sequence[Any],
        inputs: Ref2VAWanInputs,
        timesteps: Any,
        *,
        r_timesteps: Any = None,
        plan: Any = None,
        role: int = NOISY_ROLE,
    ) -> Any:
        """Velocity latents ``[1, C, F, H, W]`` of the layout's ``role`` segment."""

        segment = layout.only(role)
        tokens = model(
            list(tensors),
            coords=coords,
            timesteps=timesteps,
            context=inputs.prompt,
            r_timesteps=r_timesteps,
            plan=plan,
            output=(segment.start, segment.stop),
        )
        return unpatchify(tokens, segment, self.spec.latent_channels).float()

    # ------------------------------------------------------------------------------- Stage0.5

    def bd_velocity(self, model: Any, inputs: Ref2VAWanInputs, noisy: Any, sigma: Any) -> Any:
        """Bidirectional velocity of all ``T`` target latents at noise level ``sigma``."""

        layout, coords = self.layout(
            inputs, frame_range=(0, self.spec.latent_frames), teacher_forcing=False
        )
        sigma = torch.as_tensor(sigma, device=self.device, dtype=torch.float32)
        condition = torch.clamp(sigma, max=KEYFRAME_SIGMA) * NUM_TRAIN_TIMESTEPS
        times = self.token_times(
            layout,
            condition=condition,
            noisy=sigma * NUM_TRAIN_TIMESTEPS,
            given=inputs.given,
            given_time=condition,
        )
        return self.run(
            model, layout, coords, self.segment_tensors(layout, inputs, noisy=noisy), inputs, times
        )

    def forward_loss(self, sample: Ref2VAWanSample, *, noise_seed: int | None) -> Any:
        generator = self.generator(noise_seed)
        inputs = self.prepare(sample, generator)
        clean = inputs.clean
        noise = self.noise_like(clean, generator)
        sigma = sample_training_sigma(generator, shift=self.shift, device=self.device)
        noisy = add_noise(clean, noise, sigma)
        target = noise - clean
        given = inputs.given
        if given:
            noisy[:, :, :given] = inputs.given_value
            target[:, :, :given] = 0.0
        prediction = self.bd_velocity(self.model, inputs, noisy, sigma)
        if given:
            prediction = torch.cat(
                (torch.zeros_like(prediction[:, :, :given]), prediction[:, :, given:]), dim=2
            )
        return F.mse_loss(prediction, target)

    @torch.no_grad()
    def generate(
        self, sample: Ref2VAWanSample, *, noise_seed: int, num_inference_steps: int
    ) -> Any:
        """Shifted Euler; ``num_inference_steps`` counts grid points as H3 does (``n - 1`` NFE)."""

        generator = self.generator(noise_seed)
        inputs = self.prepare(sample, generator)
        current = self.noise_like(inputs.clean, generator)
        given = inputs.given
        if given:
            current[:, :, :given] = inputs.given_value
        sigmas = shifted_sigmas(int(num_inference_steps), self.shift, device=self.device)
        for index in range(int(sigmas.numel()) - 1):
            sigma, sigma_next = sigmas[index], sigmas[index + 1]
            velocity = self.bd_velocity(self.model, inputs, current, sigma)
            if not bool(torch.isfinite(velocity).all()):
                raise FloatingPointError("Wan Ref2VA BD inference predicted non-finite velocity")
            current = current - (sigma - sigma_next) * velocity
            if given:
                current[:, :, :given] = inputs.given_value
        return current.contiguous()


class Ref2VAWanARCore(Ref2VAWanBDCore):
    """Stage1: teacher-forced W-chunk AnyFlow and its chunked NFE4 rollout."""

    def __init__(self, model: Any, device: Any, config: Mapping[str, Any]) -> None:
        super().__init__(model, device, config)
        self._check_train_frames()

    def _check_train_frames(self) -> None:
        if self.spec.train_target_latents % self.spec.chunk_latents:
            raise ValueError(
                "Wan Ref2VA Stage1 trains whole chunks: train_target_latents % chunk_latents != 0"
            )

    def forward_loss(self, sample: Ref2VAWanSample, *, noise_seed: int | None) -> Any:
        from solarwm.training.anyflow import apply_timestep_shift, sample_anyflow_time_pairs

        generator = self.generator(noise_seed)
        inputs = self.prepare(sample, generator)
        frames = self.spec.train_target_latents
        clean = inputs.clean[:, :, :frames].contiguous()
        layout, coords = self.layout(inputs, frame_range=(0, frames), teacher_forcing=True)
        plan = self.mask(layout, windowed_clean=self.windowed_clean)
        noise = self.noise_like(clean, generator)
        rank, world = dp_rank_world()
        pairs = sample_anyflow_time_pairs(
            1,
            logical_dp_rank=rank,
            logical_dp_world_size=world,
            diffusion_ratio=float(self.train["diffusion_ratio"]),
            consistency_ratio=float(self.train["consistency_ratio"]),
            generator=generator,
            device=self.device,
        )
        sigma_t = apply_timestep_shift(pairs.t, self.shift)
        sigma_r = apply_timestep_shift(pairs.r, self.shift)
        given = inputs.given
        given_velocity = (noise[:, :, :given] - clean[:, :, :given]) if given else None

        def velocity(noisy: Any, raw_t: Any, raw_r: Any) -> Any:
            if given:
                noisy = torch.cat((inputs.given_value.to(noisy.dtype), noisy[:, :, given:]), dim=2)
            t = self.token_times(layout, condition=KEYFRAME_TIME, noisy=raw_t, given=given)
            r = self.token_times(layout, condition=KEYFRAME_TIME, noisy=raw_r, given=given)
            tensors = self.segment_tensors(layout, inputs, clean=clean, noisy=noisy)
            prediction = self.run(
                self.model, layout, coords, tensors, inputs, t, r_timesteps=r, plan=plan
            )
            if given:
                prediction = torch.cat((given_velocity, prediction[:, :, given:]), dim=2)
            return prediction

        return wan_anyflow_v15_loss(
            velocity,
            clean,
            noise,
            sigma_t,
            sigma_r,
            pairs.is_diffusion,
            epsilon=float(self.train["finite_difference_epsilon"]),
            shift=self.shift,
            process_group=dp_group(),
        )

    @torch.no_grad()
    def generate(
        self, sample: Ref2VAWanSample, *, noise_seed: int, num_inference_steps: int
    ) -> Any:
        """Roll out chunk by chunk; each chunk recomputes its W-chunk window with linear sigmas."""

        spec = self.spec
        steps = int(num_inference_steps)
        generator = self.generator(noise_seed)
        inputs = self.prepare(sample, generator)
        c, window = spec.chunk_latents, spec.window_chunks
        sigmas = torch.linspace(1.0, 0.0, steps + 1, device=self.device, dtype=torch.float32)
        shape = (1, spec.latent_channels, c, spec.latent_height, spec.latent_width)
        generated: list[Any] = []
        for chunk in range(spec.rollout_chunks):
            low = chunk * c
            fixed = min(max(inputs.given - low, 0), c)
            held = inputs.given_value[:, :, low : low + fixed] if fixed else None
            if fixed == c:
                generated.append(held.contiguous())
                continue
            first = max(0, chunk - (window - 1))
            layout, coords = self.layout(
                inputs, frame_range=(first * c, low + c), teacher_forcing=True, noisy_first=low
            )
            plan = self.mask(layout, windowed_clean=False)
            history = torch.cat(generated[first:chunk], dim=2) if chunk > first else None
            current = self.noise_like(torch.empty(shape, device=self.device), generator)
            if fixed:
                current[:, :, :fixed] = held
            for step in range(steps):
                t = self.token_times(
                    layout, condition=KEYFRAME_TIME, noisy=sigmas[step] * 1000, given=inputs.given
                )
                r = self.token_times(
                    layout,
                    condition=KEYFRAME_TIME,
                    noisy=sigmas[step + 1] * 1000,
                    given=inputs.given,
                )
                tensors = self.segment_tensors(layout, inputs, clean=history, noisy=current)
                velocity = self.run(
                    self.model, layout, coords, tensors, inputs, t, r_timesteps=r, plan=plan
                )
                current = current - (sigmas[step] - sigmas[step + 1]) * velocity
                if fixed:
                    current[:, :, :fixed] = held
            if not bool(torch.isfinite(current).all()):
                raise FloatingPointError(f"Wan Ref2VA AR rollout chunk {chunk} is non-finite")
            generated.append(current.contiguous())
        return torch.cat(generated, dim=2)[:, :, : spec.latent_frames].contiguous()


class Ref2VAWanSGFCore(Ref2VAWanARCore):
    """Stage2: cached rollout, windowed teacher-forced replay, and Stage0.5 scoring."""

    windowed_clean = True

    def __init__(self, model: Any, device: Any, config: Mapping[str, Any]) -> None:
        super().__init__(model, device, config)
        self.num_steps = int(self.train.get("num_denoising_steps", 4))
        self.schedule = shifted_sigmas(self.num_steps + 1, self.shift, device=self.device)[
            : self.num_steps
        ]
        if int(self.schedule.numel()) != self.num_steps:
            raise ValueError(
                "the SGF shifted schedule collapsed; lower timestep_shift or the step count"
            )

    def _check_train_frames(self) -> None:
        if self.spec.train_target_latents != self.spec.latent_frames:
            raise ValueError(
                "Wan Ref2VA SGF scores all cached latents: "
                "train_target_latents must equal encoded_latents"
            )

    def full_layout(self, inputs: Ref2VAWanInputs) -> tuple[Ref2VAWanLayout, Any]:
        return self.layout(inputs, frame_range=(0, self.spec.rollout_latents), teacher_forcing=True)

    def _prefill(
        self, inputs: Ref2VAWanInputs, layout: Ref2VAWanLayout, coords: Any, cache: RolloutCache
    ) -> None:
        condition = layout.condition_length
        pictures = sum(segment.length for segment in layout.role_segments(PICTURE_ROLE))
        tensors = self.segment_tensors(layout, inputs, roles=(PICTURE_ROLE, PROXY_ROLE))
        self.model(
            tensors,
            coords=coords[:condition],
            timesteps=torch.full((condition,), KEYFRAME_TIME, device=self.device),
            context=inputs.prompt,
            plan=RolloutAttention(
                "prefill",
                cache,
                picture_tokens=pictures,
                proxy_chunk_spans=proxy_chunk_spans(layout),
            ),
            output=(0, 0),
        )

    def _chunk_forward(
        self,
        inputs: Ref2VAWanInputs,
        layout: Ref2VAWanLayout,
        coords: Any,
        cache: RolloutCache,
        *,
        chunk: int,
        latents: Any,
        sigma: Any,
        held: Any,
        commit: bool,
    ) -> Any:
        c = self.spec.chunk_latents
        noisy = layout.only(NOISY_ROLE)
        per_frame = noisy.tokens_per_frame
        start = noisy.start + chunk * c * per_frame
        stop = start + c * per_frame
        fixed = 0 if held is None else int(held.shape[2])
        raw = (
            torch.zeros((), device=self.device)
            if commit
            else torch.as_tensor(sigma, device=self.device) * 1000
        )
        times = raw.expand(c * per_frame).clone()
        if fixed and not commit:
            times[: fixed * per_frame] = KEYFRAME_TIME
        tokens = self.model(
            [self._target_input(latents, inputs, chunk * c)],
            coords=coords[start:stop],
            timesteps=times,
            context=inputs.prompt,
            plan=RolloutAttention("commit" if commit else "step", cache, chunk=chunk),
            output=(0, 0) if commit else None,
        )
        if commit:
            return None
        segment = Segment(NOISY_ROLE, c, noisy.height, noisy.width, 0, chunk * c)
        x0 = predict_x0(latents, unpatchify(tokens, segment, self.spec.latent_channels), sigma)
        if fixed:
            x0[:, :, :fixed] = held
        return x0

    @torch.no_grad()
    def rollout(
        self,
        inputs: Ref2VAWanInputs,
        *,
        generator: Any = None,
        exit_index: int | None = None,
        progress: Any = None,
    ) -> Ref2VAWanRollout:
        spec, steps = self.spec, self.num_steps
        if exit_index is None:
            selected = torch.randint(0, steps, (), device=self.device, generator=generator)
            exit_index = int(selected)
        if not 0 <= exit_index < steps:
            raise ValueError(f"SGF exit index must lie in [0, {steps})")
        layout, coords = self.full_layout(inputs)
        c = spec.chunk_latents
        shape = (
            1,
            spec.latent_channels,
            spec.rollout_latents,
            spec.latent_height,
            spec.latent_width,
        )
        noise = self.noise_like(torch.empty(shape, device=self.device), generator)
        cache = RolloutCache(spec.window_chunks)
        committed, exits = [], []
        try:
            self._prefill(inputs, layout, coords, cache)
            for chunk in range(spec.rollout_chunks):
                low = chunk * c
                fixed = min(max(inputs.given - low, 0), c)
                held = inputs.given_value[:, :, low : low + fixed] if fixed else None
                if fixed == c:
                    exits.append(held.clone())
                    committed.append(held.clone())
                    self._chunk_forward(
                        inputs,
                        layout,
                        coords,
                        cache,
                        chunk=chunk,
                        latents=held,
                        sigma=0.0,
                        held=held,
                        commit=True,
                    )
                    continue
                current = noise[:, :, low : low + c].clone()
                if fixed:
                    current[:, :, :fixed] = held
                exit_x0 = None
                # Every rank runs all steps: FSDP collectives need identical forward counts.
                for step in range(steps):
                    sigma = self.schedule[step]
                    if step == exit_index:
                        exits.append(current.clone())
                    x0 = self._chunk_forward(
                        inputs,
                        layout,
                        coords,
                        cache,
                        chunk=chunk,
                        latents=current,
                        sigma=sigma,
                        held=held,
                        commit=False,
                    )
                    if not bool(torch.isfinite(x0).all()):
                        raise FloatingPointError(
                            f"non-finite Wan Ref2VA SGF chunk={chunk} step={step}"
                        )
                    if step == exit_index:
                        exit_x0 = x0.clone()
                    if step < steps - 1:
                        current = add_noise(
                            x0, self.noise_like(x0, generator), self.schedule[step + 1]
                        )
                        if fixed:
                            current[:, :, :fixed] = held
                committed.append(exit_x0)
                self._chunk_forward(
                    inputs,
                    layout,
                    coords,
                    cache,
                    chunk=chunk,
                    latents=exit_x0,
                    sigma=0.0,
                    held=held,
                    commit=True,
                )
                if progress is not None:
                    progress(chunk + 1)
            return Ref2VAWanRollout(
                torch.cat(committed, dim=2),
                torch.cat(exits, dim=2),
                exit_index,
                self.schedule[exit_index],
            )
        finally:
            cache.clear()

    def replay(self, inputs: Ref2VAWanInputs, rollout: Ref2VAWanRollout) -> Any:
        """Exit-step clean predictions of every chunk, teacher-forced on the rollout (with grad)."""

        layout, coords = self.full_layout(inputs)
        plan = self.mask(layout, windowed_clean=True)
        sigma = rollout.exit_sigma
        times = self.token_times(
            layout, condition=KEYFRAME_TIME, noisy=sigma * 1000, given=inputs.given
        )
        tensors = self.segment_tensors(
            layout, inputs, clean=rollout.cache_target.detach(), noisy=rollout.noisy_at_exit
        )
        velocity = self.run(self.model, layout, coords, tensors, inputs, times, plan=plan)
        return predict_x0(rollout.noisy_at_exit, velocity, sigma)

    def supervised(self, video: Any) -> Any:
        return video[:, :, : self.spec.latent_frames]

    def kl_mask(self, inputs: Ref2VAWanInputs) -> Any:
        mask = torch.ones(
            (1, 1, self.spec.latent_frames, 1, 1), device=self.device, dtype=torch.bool
        )
        mask[:, :, : inputs.given] = False
        return mask

    def score_inputs(
        self, clean: Any, inputs: Ref2VAWanInputs, *, generator: Any = None
    ) -> tuple[Any, Any, Any]:
        sigma = sgf_score_sigma(
            shift=self.shift,
            min_sigma=float(self.train["score_min_sigma"]),
            max_sigma=float(self.train["score_max_sigma"]),
            generator=generator,
            device=self.device,
        )
        noise = self.noise_like(clean, generator)
        noisy = add_noise(clean, noise, sigma)
        if inputs.given:
            noisy[:, :, : inputs.given] = inputs.given_value
        return noisy, noise, sigma

    def score_forward(self, model: Any, inputs: Ref2VAWanInputs, noisy: Any, sigma: Any) -> Any:
        return self.bd_velocity(model, inputs, noisy, sigma)

    def critic_loss(self, velocity: Any, *, noise: Any, clean: Any, inputs: Ref2VAWanInputs) -> Any:
        given = inputs.given
        target = noise[:, :, given:].float() - clean[:, :, given:].float()
        return F.mse_loss(velocity[:, :, given:].float(), target)

    @torch.no_grad()
    def generate(
        self, sample: Ref2VAWanSample, *, noise_seed: int, num_inference_steps: int
    ) -> Any:
        if int(num_inference_steps) != self.num_steps:
            raise ValueError(
                f"Wan Ref2VA SGF rollout uses {self.num_steps} denoiser evaluations per chunk"
            )
        generator = self.generator(noise_seed)
        inputs = self.prepare(sample, generator)
        rollout = self.rollout(inputs, generator=generator, exit_index=self.num_steps - 1)
        return self.supervised(rollout.cache_target).contiguous()


CORES = {"stage0p5": Ref2VAWanBDCore, "stage1": Ref2VAWanARCore, "stage2": Ref2VAWanSGFCore}

__all__ = [
    "CORES",
    "KEYFRAME_TIME",
    "Ref2VAWanARCore",
    "Ref2VAWanBDCore",
    "Ref2VAWanInputs",
    "Ref2VAWanRollout",
    "Ref2VAWanSGFCore",
    "dp_group",
    "dp_rank_world",
    "unpatchify",
]
