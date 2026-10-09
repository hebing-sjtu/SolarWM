"""Native standalone inference for cached 124-frame H3 Ref2VA proxy samples."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from solarwm.errors import BackendContractError
from solarwm.inference import InferenceCase, InferenceEngine
from solarwm.runtime import Topology, rng_identity
from solarwm.runtime.distributed import gather_and_assert_sp_identity
from solarwm.runtime.randomness import seed_process
from solarwm.training.wandb_sink import WandbEventSink

from .distributed import get_sp_group, get_sp_rank, get_sp_size
from .inference import package_proxy_generated
from .optional import load_conditioners, load_transformer, require_h3_runtime
from .proxy_ablation import apply_proxy_ablation
from .proxy_artifacts import H3ProxyPtStream
from .proxy_weights import load_proxy_checkpoint
from .stage0p5_ref2va import H3Ref2VAStage0p5Core


def _next_distinct_donor(
    stream: H3ProxyPtStream,
    sample_id: str,
    proxy_references: tuple[str, ...],
) -> Any:
    for _ in range(len(stream.paths)):
        donor = stream.next()
        if donor.sample_id != sample_id and donor.proxy_references == proxy_references:
            return donor
    raise BackendContractError(
        "shuffled proxy ablation requires at least two distinct cached samples "
        f"with modality {list(proxy_references)}"
    )


def run_proxy_inference(config: Mapping[str, Any]) -> int:
    """Evaluate cached proxy documents on one torchrun worker."""

    torch, _diffusers, _transformers = require_h3_runtime()
    import torch.distributed as dist

    from .fsdp import initialize_distributed, wrap_h3_fsdp
    from .lora import inject_h3_lora
    from .runtime import (
        _base_model_load_receipt,
        _base_weights_label,
        _collective_call,
        _collective_failures,
        _topology,
        _validation_schedule,
    )

    distributed = config["distributed"]
    configured_sp = int(distributed["sequence_parallel_size"])
    topology = _topology(sp_size=configured_sp, require_torchrun=True)
    configured_world = int(distributed["world_size"])
    if topology.raw_world_size != configured_world:
        raise BackendContractError(
            f"torchrun WORLD_SIZE={topology.raw_world_size} differs from configured "
            f"H3 proxy inference world_size={configured_world}"
        )
    initialize_distributed(sp_size=topology.sp_size, local_rank=topology.local_rank)
    topology = (
        Topology.from_environ(topology.sp_size)
        if {"WORLD_SIZE", "RANK", "LOCAL_WORLD_SIZE", "LOCAL_RANK"} <= set(os.environ)
        else topology
    )
    device = torch.device("cuda", topology.local_rank)
    identity = rng_identity("minimax_h3", int(config["data"].get("seed", 42)), topology)
    seed_process(identity.model_init_seed)

    modules = load_transformer(config["model"], device=device)
    modules.transformer.eval().requires_grad_(False)
    base_model = _base_model_load_receipt(config["model"], modules.transformer)
    model, lora = inject_h3_lora(
        modules.transformer,
        config["model"]["adapter"],
        base_identity=base_model,
    )
    model = wrap_h3_fsdp(
        model,
        local_rank=topology.local_rank,
        transformer_block_cls=modules.transformer_block_cls,
        fp32_units=modules.fp32_fsdp_units,
        ignored_parameters=lora.parameters,
        activation_checkpointing=False,
    )

    checkpoint = config.get("checkpoint", {})
    checkpoint_path = str(checkpoint.get("resume_from") or "").strip()
    weight_source = str(checkpoint.get("weight_source", "ema")) if checkpoint_path else "base"
    if checkpoint_path:
        weights_id = _collective_call(
            lambda: load_proxy_checkpoint(
                checkpoint_path,
                lora,
                weight_source=weight_source,
                torch=torch,
            ),
            dist=dist,
            topology=topology,
            label="proxy inference checkpoint restore",
        )
    else:
        weights_id = _base_weights_label(config["model"])
    checkpoint_step = int(weights_id.rsplit("step=", 1)[1]) if checkpoint_path else 0

    sample_count, num_waves = _validation_schedule(config["validation"], topology)
    stream = H3ProxyPtStream(
        config,
        topology,
        selection_seed=int(config["validation"]["selection_seed"]),
    )
    ablation_mode = str(config["validation"].get("proxy_ablation", "correct")).strip().lower()
    donor_stream = (
        H3ProxyPtStream(
            config,
            topology,
            selection_seed=int(config["validation"]["selection_seed"]) + 1,
        )
        if ablation_mode == "shuffled"
        else None
    )
    core = H3Ref2VAStage0p5Core(model, device, config)
    output_root = Path(str(config["runtime"]["output_dir"])).resolve() / "proxy-inference"
    noise_seed = int(config["validation"]["noise_seed"])
    inference_steps = int(config["validation"]["num_inference_steps"])
    conditioners = None
    media_sink = None
    tracking = config["runtime"].get("tracking")
    tracking_error = ""
    if (
        topology.raw_rank == 0
        and isinstance(tracking, Mapping)
        and bool(tracking.get("enabled"))
        and bool(tracking.get("log_media"))
    ):
        try:
            media_sink = WandbEventSink(
                tracking,
                output_dir=config["runtime"]["output_dir"],
                resolved_config=config,
            )
        except Exception as exc:
            tracking_error = f"rank 0: {type(exc).__name__}: {exc}"
    failures = _collective_failures(dist, topology, tracking_error)
    if failures:
        raise BackendContractError(
            "H3 proxy inference W&B initialization failed: " + " | ".join(failures)
        )

    completed: list[dict[str, Any]] = []
    try:
        for wave_index in range(num_waves):
            batch = _collective_call(
                stream.next,
                dist=dist,
                topology=topology,
                label=f"proxy inference data wave {wave_index}",
            )
            donor = (
                _collective_call(
                    lambda batch=batch: _next_distinct_donor(
                        donor_stream,
                        batch.sample_id,
                        batch.proxy_references,
                    ),
                    dist=dist,
                    topology=topology,
                    label=f"proxy inference donor wave {wave_index}",
                )
                if donor_stream is not None
                else None
            )
            batch = _collective_call(
                lambda batch=batch, donor=donor: apply_proxy_ablation(
                    batch,
                    mode=ablation_mode,
                    donor=donor,
                ),
                dist=dist,
                topology=topology,
                label=f"proxy inference ablation wave {wave_index}",
            )
            donor_sample_id = donor.sample_id if donor is not None else None
            slot = wave_index * int(topology.dp_world_size) + int(topology.dp_rank)
            seed = noise_seed + slot
            gather_and_assert_sp_identity(
                {
                    "sample_id": batch.sample_id,
                    "start_frame": batch.start_frame,
                    "noise_seed": seed,
                    "plan_fingerprint": batch.plan_fingerprint,
                },
                sp_size=get_sp_size(),
                group=get_sp_group(),
            )
            case = InferenceCase(
                slot=slot,
                sample_id=batch.sample_id,
                prompt="",
                start_frame=0,
                noise_seed=seed,
                camera_fingerprint=f"proxy:{batch.plan_fingerprint}",
                metadata={
                    "key": batch.sample_id,
                    "dataset": batch.dataset_source,
                    "plan_fingerprint": batch.plan_fingerprint,
                    "source_pixel_frames": 124,
                    "output_pixel_frames": 124,
                    "train_latent_frames": 37,
                    "rollout_latent_frames": 37,
                    "generation_mode": "bidirectional-ref2va-proxy",
                    "sample_solver": "shifted-euler-data-ward",
                    "weights_source": weight_source,
                    "proxy_ablation": ablation_mode,
                    "proxy_donor_sample_id": donor_sample_id,
                    "artifact_valid": True,
                },
            )
            generated_latents = _collective_call(
                lambda batch=batch, seed=seed: core.generate(
                    batch,
                    noise_seed=seed,
                    num_inference_steps=inference_steps,
                ),
                dist=dist,
                topology=topology,
                label=f"proxy inference wave {wave_index} generation",
            )
            local_error = ""
            if get_sp_rank() == 0:
                try:
                    if conditioners is None:
                        conditioners = load_conditioners(
                            config["model"],
                            device=device,
                            qwen=False,
                            video_vae=True,
                            audio_vae=False,
                            schedulers=False,
                        )
                    packaged = package_proxy_generated(
                        generated_latents,
                        video_vae=conditioners.video_vae,
                        device=device,
                        weights_id=weights_id,
                        num_inference_steps=inference_steps,
                        proxy_latents=batch.proxy_latents,
                        reference_latents=batch.target_latents,
                        proxy_references=batch.proxy_references,
                    )
                    panel_caption = " | ".join(
                        f"proxy_{name}" for name in batch.proxy_references
                    )

                    class CachedAdapter:
                        family = "minimax_h3"

                        def __init__(self, sample: Any) -> None:
                            self.sample = sample

                        def generate(self, _case: Any, *, weights_id: str) -> Any:
                            del _case, weights_id
                            return self.sample

                    destination = (
                        output_root / f"wave-{wave_index:03d}" / f"dp-rank-{topology.dp_rank:05d}"
                    )
                    summary = InferenceEngine(CachedAdapter(packaged)).run(
                        [case],
                        weights_id=weights_id,
                        output_dir=destination,
                    )
                    completed.append(
                        {
                            "slot": slot,
                            "sample_id": batch.sample_id,
                            "proxy_ablation": ablation_mode,
                            "proxy_donor_sample_id": donor_sample_id,
                            "noise_seed": seed,
                            "output_dir": str(summary.output_dir.relative_to(output_root)),
                        }
                    )
                    if media_sink is not None:
                        compare = summary.output_dir / f"slot-{case.slot:06d}" / "compare.mp4"
                        media_sink.run.log(
                            {
                                f"eval/compare/slot-{slot:06d}": media_sink.wandb.Video(
                                    str(compare),
                                    fps=24,
                                    format="mp4",
                                    caption=(
                                        f"{panel_caption} | prediction | target; "
                                        f"sample={batch.sample_id}; "
                                        f"ablation={ablation_mode}; donor={donor_sample_id}"
                                    ),
                                )
                            },
                            step=checkpoint_step,
                            commit=False,
                        )
                    print(
                        f"[h3-proxy-infer] slot={slot} sample={batch.sample_id} "
                        f"output={summary.output_dir}",
                        flush=True,
                    )
                except Exception as exc:
                    local_error = f"rank {topology.raw_rank}: {type(exc).__name__}: {exc}"
            failures = _collective_failures(dist, topology, local_error)
            if failures:
                raise BackendContractError(
                    "H3 proxy inference output failed: " + " | ".join(failures)
                )
    finally:
        stream.close()
        if donor_stream is not None:
            donor_stream.close()

    tracking_error = ""
    if topology.raw_rank == 0 and media_sink is not None:
        try:
            media_sink.run.log(
                {
                    "eval/completed": 1,
                    "eval/sample_count": len(completed),
                    "eval/num_inference_steps": inference_steps,
                },
                step=checkpoint_step,
            )
            media_sink.finish()
        except Exception as exc:
            tracking_error = f"rank 0: {type(exc).__name__}: {exc}"
    failures = _collective_failures(dist, topology, tracking_error)
    if failures:
        raise BackendContractError(
            "H3 proxy inference W&B publication failed: " + " | ".join(failures)
        )

    completion_error = ""
    if topology.raw_rank == 0:
        try:
            output_root.mkdir(parents=True, exist_ok=True)
            payload = {
                "schema": "solarwm.minimax-h3-proxy-inference.v1",
                "weights_id": weights_id,
                "checkpoint": checkpoint_path or None,
                "weight_source": weight_source,
                "proxy_ablation": ablation_mode,
                "sample_count": sample_count,
                "num_inference_steps": inference_steps,
                "samples": completed,
            }
            (output_root / "COMPLETE.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        except Exception as exc:
            completion_error = f"rank 0: {type(exc).__name__}: {exc}"
    failures = _collective_failures(dist, topology, completion_error)
    if failures:
        raise BackendContractError("H3 proxy inference completion failed: " + " | ".join(failures))
    return 0


__all__ = ["run_proxy_inference"]
