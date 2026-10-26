"""Training and inference runtime of Wan Ref2VA: torchrun, FSDP1, full fine-tuning or LoRA.

``model.training_mode: full`` trains every weight (fp32 masters, bf16 compute, FSDP-sharded
optimizer state); ``lora`` freezes the bf16 base under FSDP and trains replicated fp32 adapters
(plus AnyFlow's delta embedding), whose gradients are averaged by hand.

A checkpoint directory ``checkpoint-NNNNNN`` holds, per trained role (``student``, and
``critic`` in Stage2): ``ROLE.safetensors`` (trainable weights: the whole model for full
fine-tuning, adapters for LoRA), optional ``ROLE-ema.safetensors``, the optimizer state, the
reader positions, and ``COMPLETE.json`` written last. Later stages initialize from these files.
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn

from .cores import CORES, Ref2VAWanSGFCore
from .data import Ref2VAWanSpec, Ref2VAWanStream, read_sample, sample_paths
from .lora import adapter_state, clean_name, inject_lora, merge_lora
from .lora import sync_gradients as sync_lora_gradients
from .model import Ref2VAWanBlock, build_model
from .objectives import predict_x0

ANYFLOW_STAGES = ("stage1", "stage2")


# ------------------------------------------------------------------------------- distributed


@dataclass(frozen=True)
class Topology:
    rank: int
    world: int
    local_rank: int
    device: Any

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def initialize() -> Topology:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.cuda.is_available():
        raise RuntimeError("Wan Ref2VA needs CUDA")
    torch.cuda.set_device(local_rank)
    if world > 1 and not dist.is_initialized():
        method = os.environ.get("SOLARWM_DISTRIBUTED_INIT_METHOD", "env://")
        extra = {} if method == "env://" else {"rank": rank, "world_size": world}
        dist.init_process_group("nccl", init_method=method, timeout=timedelta(hours=1), **extra)
    return Topology(rank, world, local_rank, torch.device("cuda", local_rank))


def barrier() -> None:
    if dist.is_initialized():
        dist.barrier()


def cleanup() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    if dist.is_initialized():
        dist.destroy_process_group()


# ------------------------------------------------------------------------------------ roles


class ParameterEMA:
    """FP32 shadow of the local trainable tensors (FSDP shards or replicated adapters)."""

    def __init__(self, named: list[tuple[str, nn.Parameter]], decay: float) -> None:
        self.decay = float(decay)
        self.shadow = {name: parameter.detach().float().clone() for name, parameter in named}

    @torch.no_grad()
    def update(self, named: list[tuple[str, nn.Parameter]]) -> None:
        for name, parameter in named:
            self.shadow[name].lerp_(parameter.detach().float(), 1.0 - self.decay)

    @torch.no_grad()
    def swap(self, named: list[tuple[str, nn.Parameter]]) -> None:
        """Exchange live weights and the shadow in place (twice restores both)."""

        for name, parameter in named:
            live = parameter.detach().float().clone()
            parameter.data.copy_(self.shadow[name].to(parameter.dtype))
            self.shadow[name].copy_(live)


@dataclass
class Role:
    name: str
    model: Any
    full: bool
    trainable: bool
    identity: dict[str, Any]
    optimizer: Any = None
    scheduler: Any = None
    ema: ParameterEMA | None = None
    clip: float = 1.0
    named: list[tuple[str, nn.Parameter]] = field(default_factory=list)

    @property
    def parameters(self) -> list[nn.Parameter]:
        return [parameter for _, parameter in self.named]


def _weights_file(entry: Mapping[str, Any], role: str) -> Path:
    path = Path(str(entry["path"]))
    if path.is_file():
        return path
    source = str(entry.get("weight_source", "live"))
    if source not in ("live", "ema"):
        raise ValueError(f"weight_source must be live or ema, got {source!r}")
    name = str(entry.get("role", role))
    file = path / (f"{name}-ema.safetensors" if source == "ema" else f"{name}.safetensors")
    if not file.is_file():
        raise FileNotFoundError(f"Wan Ref2VA weights {file} do not exist")
    return file


def load_role_weights(model: nn.Module, file: Path, *, full: bool) -> dict[str, Any]:
    """Load a role file into an unwrapped model; returns its identity and if it held a delta."""

    from safetensors.torch import load_file

    state = load_file(str(file), device="cpu")
    expected = dict(model.named_parameters())
    delta = {key for key in state if key.startswith("delta_embedding.")}
    usable = {key: value for key, value in state.items() if key in expected}
    unexpected = sorted(set(state) - set(expected) - delta)
    if unexpected:
        raise ValueError(f"{file} has keys the model lacks: {unexpected[:6]}")
    if full:
        missing = sorted(
            key for key in expected if key not in state and not key.startswith("delta_embedding.")
        )
    else:
        missing = sorted(key for key in expected if ".lora_" in key and key not in state)
    if missing:
        raise ValueError(f"{file} misses {len(missing)} weights, e.g. {missing[:6]}")
    with torch.no_grad():
        for key, value in usable.items():
            if tuple(expected[key].shape) != tuple(value.shape):
                raise ValueError(
                    f"{file}:{key} has shape {tuple(value.shape)}, "
                    f"model {tuple(expected[key].shape)}"
                )
            expected[key].copy_(value.to(expected[key].dtype))
    digest = hashlib.sha256(f"{file.resolve()}\0{file.stat().st_size}".encode()).hexdigest()[:16]
    return {
        "file": str(file),
        "keys": len(usable),
        "id": digest,
        "delta": bool(delta & set(usable)),
    }


def _wrap(
    model: nn.Module,
    *,
    role_trainable: bool,
    full: bool,
    topology: Topology,
    fsdp: Mapping[str, Any],
) -> nn.Module:
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        CheckpointImpl,
        apply_activation_checkpointing,
        checkpoint_wrapper,
    )

    replicated = [p for p in model.parameters() if p.requires_grad] if not full else []
    if topology.world > 1:
        from torch.distributed.fsdp import BackwardPrefetch, MixedPrecision, ShardingStrategy
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp.wrap import ModuleWrapPolicy

        key = "sharding_strategy" if role_trainable else "frozen_sharding_strategy"
        strategy = getattr(
            ShardingStrategy, str(fsdp.get(key, fsdp.get("sharding_strategy", "FULL_SHARD")))
        )
        mixed = MixedPrecision(
            param_dtype=torch.bfloat16 if full and role_trainable else None,
            reduce_dtype=torch.float32,
            buffer_dtype=None,
        )
        model = FSDP(
            model,
            auto_wrap_policy=ModuleWrapPolicy({Ref2VAWanBlock}),
            sharding_strategy=strategy,
            mixed_precision=mixed,
            device_id=topology.device,
            use_orig_params=True,
            forward_prefetch=True,
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            limit_all_gathers=True,
            sync_module_states=False,
            ignored_states=replicated or None,
        )
        for parameter in replicated:
            parameter.data = parameter.data.to(topology.device)
    else:
        model = model.to(topology.device)
    if role_trainable and bool(fsdp.get("activation_checkpointing", True)):
        apply_activation_checkpointing(
            model,
            checkpoint_wrapper_fn=functools.partial(
                checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT
            ),
            check_fn=lambda module: isinstance(module, Ref2VAWanBlock),
        )
    return model


def build_role(
    config: Mapping[str, Any],
    name: str,
    *,
    anyflow: bool,
    trainable: bool,
    init: Mapping[str, Any] | None,
    topology: Topology,
    distributed: bool = True,
) -> Role:
    """Base weights, adapters, initialization, freezing and wrapping of one model role."""

    model_cfg, train = config["model"], config["train"]
    full = str(model_cfg["training_mode"]) == "full"
    if str(model_cfg["training_mode"]) not in ("full", "lora"):
        raise ValueError("model.training_mode must be full or lora")
    gate = float(train.get("anyflow_gate", 0.25)) if anyflow else None
    dtype = torch.float32 if full and trainable else torch.bfloat16
    weights = model_cfg.get("weights")
    model, receipt = build_model(
        str(model_cfg["family"]),
        weights=None if weights is None else str(weights),
        anyflow_gate=gate,
        dtype=dtype,
        overrides=model_cfg.get("architecture_overrides"),
    )
    identity: dict[str, Any] = {"base": receipt["directory"]}
    if not full:
        adapter = model_cfg["adapter"]
        targets = inject_lora(
            model,
            target=str(adapter["target"]),
            rank=int(adapter["rank"]),
            alpha=float(adapter["alpha"]),
        )
        expected = adapter.get("expected_target_linear_modules")
        if expected is not None and len(targets) != int(expected):
            raise ValueError(f"LoRA wrapped {len(targets)} linears, expected {expected}")
    if init is not None:
        loaded = load_role_weights(model, _weights_file(init, "student"), full=full)
        identity["init"] = loaded
        if anyflow and not loaded["delta"]:
            model.initialize_anyflow_delta_from_time()
    model.requires_grad_(False)
    if not full and not trainable:
        merge_lora(model)
    if trainable:
        for parameter_name, parameter in model.named_parameters():
            if full or ".lora_" in parameter_name or parameter_name.startswith("delta_embedding."):
                parameter.requires_grad_(True)
        if not full and anyflow:
            model.delta_embedding.float()
    if distributed:
        model = _wrap(
            model,
            role_trainable=trainable,
            full=full,
            topology=topology,
            fsdp=train.get("fsdp", {}),
        )
    else:
        model = model.to(topology.device)
    model.train(trainable)
    role = Role(name, model, full, trainable, identity)
    role.named = [(clean_name(n), p) for n, p in model.named_parameters() if p.requires_grad]
    return role


def attach_optimizer(role: Role, optimizer_cfg: Mapping[str, Any], *, max_steps: int) -> None:
    from solarwm.training.schedule import make_warmup_cosine

    role.optimizer = torch.optim.AdamW(
        role.parameters,
        lr=float(optimizer_cfg["learning_rate"]),
        betas=tuple(float(value) for value in optimizer_cfg.get("betas", (0.9, 0.999))),
        eps=float(optimizer_cfg.get("epsilon", 1e-8)),
        weight_decay=float(optimizer_cfg.get("weight_decay", 0.0)),
        foreach=True,
    )
    role.scheduler = make_warmup_cosine(
        role.optimizer,
        warmup_steps=int(optimizer_cfg.get("warmup_steps", 0)),
        total_steps=int(max_steps),
        min_lr_ratio=float(optimizer_cfg.get("min_lr_ratio", 1.0)),
    )
    role.clip = float(optimizer_cfg.get("gradient_clip", 1.0))


def finish_update(role: Role, topology: Topology) -> float:
    """Reduce (LoRA), clip, step and clear one role's gradients; returns the pre-clip norm."""

    if role.full:
        if topology.world > 1:
            norm = role.model.clip_grad_norm_(role.clip)
        else:
            norm = torch.nn.utils.clip_grad_norm_(role.parameters, role.clip)
        norm = float(norm)
        finite = bool(torch.isfinite(torch.tensor(norm)))
    else:
        from solarwm.backends.minimax_h3.fsdp import finite_clip_norm

        sync_lora_gradients(role.parameters)
        finite, norm = finite_clip_norm(role.parameters, role.clip)
    if not finite:
        raise FloatingPointError(f"non-finite Wan Ref2VA {role.name} gradient norm {norm}")
    role.optimizer.step()
    role.scheduler.step()
    role.optimizer.zero_grad(set_to_none=True)
    return norm


# ------------------------------------------------------------------------------ checkpoints


def _full_state(role: Role, topology: Topology) -> dict[str, Any] | None:
    """Rank 0's CPU copy of the role's trainable weights; ``None`` elsewhere."""

    if not role.full:
        state = adapter_state(role.model, extra_prefixes=("delta_embedding.",))
        return state if topology.is_main else None
    if topology.world == 1:
        return {
            clean_name(k): v.detach().float().cpu().contiguous()
            for k, v in role.model.state_dict().items()
        }
    from torch.distributed.fsdp import FullStateDictConfig, StateDictType
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(role.model, StateDictType.FULL_STATE_DICT, config):
        state = role.model.state_dict()
    if not topology.is_main:
        return None
    return {clean_name(k): v.detach().float().cpu().contiguous() for k, v in state.items()}


def save_role(role: Role, directory: Path, topology: Topology) -> None:
    from safetensors.torch import save_file

    state = _full_state(role, topology)
    if state is not None:
        save_file(state, str(directory / f"{role.name}.safetensors"))
    del state
    if role.ema is not None:
        role.ema.swap(role.named)
        try:
            state = _full_state(role, topology)
        finally:
            role.ema.swap(role.named)
        if state is not None:
            save_file(state, str(directory / f"{role.name}-ema.safetensors"))
        del state
    if role.optimizer is not None:
        shard = role.full or topology.is_main
        if shard:
            suffix = f"-rank{topology.rank:03d}" if role.full else ""
            payload = {
                "optimizer": role.optimizer.state_dict(),
                "scheduler": role.scheduler.state_dict(),
                "ema": None if role.ema is None else role.ema.shadow,
            }
            torch.save(payload, directory / f"{role.name}-optimizer{suffix}.pt")


def load_role_state(role: Role, directory: Path, topology: Topology) -> None:
    suffix = f"-rank{topology.rank:03d}" if role.full else ""
    path = directory / f"{role.name}-optimizer{suffix}.pt"
    payload = torch.load(path, map_location=topology.device, weights_only=False)
    role.optimizer.load_state_dict(payload["optimizer"])
    role.scheduler.load_state_dict(payload["scheduler"])
    if payload.get("ema") is not None:
        if role.ema is None:
            role.ema = ParameterEMA(role.named, decay=0.0)
        role.ema.shadow = {
            name: value.to(topology.device) for name, value in payload["ema"].items()
        }


# --------------------------------------------------------------------------------- training


class Ref2VAWanTrainer:
    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = config
        self.topology = initialize()
        self.train = config["train"]
        self.stage = str(self.train["stage"])
        if self.stage not in CORES:
            raise ValueError(f"train.stage must be one of {sorted(CORES)}")
        self.spec = Ref2VAWanSpec.from_config(config)
        self.max_steps = int(self.train["max_steps"])
        self.accumulation = int(self.train.get("gradient_accumulation_steps", 1))
        expected = self.train.get("global_batch_size")
        if expected is not None and int(expected) != self.topology.world * self.accumulation:
            raise ValueError(
                f"global_batch_size {expected} != world {self.topology.world} "
                f"x accumulation {self.accumulation}"
            )
        self.output_dir = Path(str(config["runtime"]["output_dir"]))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_cfg = config["checkpoint"]
        self.seed = int(config["data"].get("seed", 42))
        torch.manual_seed(self.seed + self.topology.rank)
        resume = self.checkpoint_cfg.get("resume_from")
        self.resume_dir = Path(str(resume)) if resume else None
        self.global_step = 0
        self.student_step = 0
        initialization = self.checkpoint_cfg.get("initialization") or {}

        def init_for(role: str, trained: bool) -> Mapping[str, Any] | None:
            if trained and self.resume_dir is not None:
                return {"path": str(self.resume_dir), "role": role, "weight_source": "live"}
            return initialization.get(role)

        anyflow = self.stage in ANYFLOW_STAGES
        self.student = build_role(
            config,
            "student",
            anyflow=anyflow,
            trainable=True,
            init=init_for("student", True),
            topology=self.topology,
        )
        attach_optimizer(self.student, self.train["optimizer"], max_steps=self.max_steps)
        self.roles = [self.student]
        self.teacher = self.critic = None
        if self.stage == "stage2":
            self.teacher = build_role(
                config,
                "teacher",
                anyflow=False,
                trainable=False,
                init=initialization["teacher"],
                topology=self.topology,
            )
            critic_init = init_for("critic", True)
            if critic_init is None:
                raise ValueError("Wan Ref2VA SGF needs checkpoint.initialization.critic")
            self.critic = build_role(
                config,
                "critic",
                anyflow=False,
                trainable=True,
                init=critic_init,
                topology=self.topology,
            )
            attach_optimizer(self.critic, self.train["critic_optimizer"], max_steps=self.max_steps)
            self.roles.append(self.critic)
        self.core = CORES[self.stage](self.student.model, self.topology.device, config)
        self.reader = Ref2VAWanStream(
            config, rank=self.topology.rank, world_size=self.topology.world
        )
        if self.resume_dir is not None:
            self._resume(self.resume_dir)
        self._ema_cfg = self.checkpoint_cfg.get("ema") or {}

    # -------------------------------------------------------------------------- lifecycle

    def _resume(self, directory: Path) -> None:
        if not (directory / "COMPLETE.json").is_file():
            raise FileNotFoundError(f"{directory} is not a complete Wan Ref2VA checkpoint")
        state = json.loads((directory / "state.json").read_text())
        if int(state["world_size"]) != self.topology.world or state["stage"] != self.stage:
            raise ValueError("Wan Ref2VA resume needs the same stage and world size")
        self.global_step = int(state["global_step"])
        self.student_step = int(state.get("student_step", self.global_step))
        for role in self.roles:
            load_role_state(role, directory, self.topology)
        reader = json.loads((directory / f"reader-rank{self.topology.rank:03d}.json").read_text())
        self.reader.load_state_dict(reader)
        if self.topology.is_main:
            print(f"[wan-ref2va] resumed {directory} global_step={self.global_step}", flush=True)

    def _update_ema(self, role: Role, step: int) -> None:
        cfg = self._ema_cfg
        if not bool(cfg.get("enabled", False)) or step < int(cfg.get("start_step", 0)):
            return
        if role.ema is None:
            role.ema = ParameterEMA(role.named, decay=float(cfg["decay"]))
        role.ema.decay = float(cfg["decay"])
        role.ema.update(role.named)

    def save_checkpoint(self, step: int) -> Path:
        torch.cuda.synchronize(self.topology.device)
        torch.cuda.empty_cache()
        directory = self.output_dir / f"checkpoint-{step:06d}"
        if self.topology.is_main:
            directory.mkdir(parents=True, exist_ok=True)
        barrier()
        for role in self.roles:
            save_role(role, directory, self.topology)
        (directory / f"reader-rank{self.topology.rank:03d}.json").write_text(
            json.dumps(self.reader.state_dict())
        )
        barrier()
        if self.topology.is_main:
            state = {
                "schema": "solarwm.wan22-ref2va-checkpoint.v1",
                "stage": self.stage,
                "family": self.spec.family,
                "training_mode": str(self.config["model"]["training_mode"]),
                "global_step": step,
                "student_step": self.student_step,
                "world_size": self.topology.world,
                "roles": {role.name: role.identity for role in self.roles},
            }
            (directory / "state.json").write_text(json.dumps(state, indent=2, sort_keys=True))
            (directory / "COMPLETE.json").write_text(
                json.dumps({"step": step, "time": time.time()})
            )
            print(f"[wan-ref2va] saved {directory}", flush=True)
        barrier()
        return directory

    def _noise_seed(self, micro: int) -> int:
        return (
            self.seed * 1_000_003
            + self.global_step * self.topology.world * self.accumulation
            + micro * self.topology.world
            + self.topology.rank
        ) % (2**63 - 1)

    # ------------------------------------------------------------------------------- steps

    def train_step(self) -> dict[str, float]:
        loss_total = 0.0
        for micro in range(self.accumulation):
            loss = self.core.forward_loss(self.reader.next(), noise_seed=self._noise_seed(micro))
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(
                    f"non-finite Wan Ref2VA {self.stage} loss at step {self.global_step + 1}"
                )
            (loss / self.accumulation).backward()
            loss_total += float(loss.detach()) / self.accumulation
        norm = finish_update(self.student, self.topology)
        self.global_step += 1
        self.student_step = self.global_step
        self._update_ema(self.student, self.global_step)
        return {
            "loss": loss_total,
            "grad_norm": norm,
            "lr": float(self.student.optimizer.param_groups[0]["lr"]),
        }

    def sgf_step(self) -> dict[str, float]:
        from solarwm.training.sgf import (
            compute_sgf_kl_gradient,
            sgf_student_loss,
            should_update_student,
        )

        core: Ref2VAWanSGFCore = self.core
        ratio = int(self.train.get("critic_updates_per_student", 5))
        update_student = should_update_student(self.global_step, ratio)
        stats = {
            "student_updated": float(update_student),
            "loss_student": 0.0,
            "student_grad_norm": 0.0,
        }
        generator = core.generator(self._noise_seed(0))
        if update_student:
            inputs = core.prepare(self.reader.next(), generator)
            rollout = core.rollout(inputs, generator=generator)
            output = core.supervised(core.replay(inputs, rollout))
            mask = core.kl_mask(inputs)
            with torch.no_grad():
                noisy, _noise, sigma = core.score_inputs(
                    output.detach(), inputs, generator=generator
                )
                fake = predict_x0(
                    noisy, core.score_forward(self.critic.model, inputs, noisy, sigma), sigma
                )
                real = predict_x0(
                    noisy, core.score_forward(self.teacher.model, inputs, noisy, sigma), sigma
                )
                if not bool(torch.isfinite(fake).all() and torch.isfinite(real).all()):
                    raise FloatingPointError("non-finite Wan Ref2VA SGF score output")
                gradient = compute_sgf_kl_gradient(
                    fake_x0=fake, real_x0=real, student_output=output.detach(), mask=mask
                )
            loss = sgf_student_loss(output, gradient, mask=mask)
            loss.backward()
            stats["loss_student"] = float(loss.detach())
            stats["student_grad_norm"] = finish_update(self.student, self.topology)
            self.student_step += 1
            self._update_ema(self.student, self.student_step)
            del inputs, rollout, output, noisy, fake, real, gradient, loss
        inputs = core.prepare(self.reader.next(), generator)
        rollout = core.rollout(inputs, generator=generator)
        with torch.no_grad():
            clean = core.supervised(core.replay(inputs, rollout))
            noisy, noise, sigma = core.score_inputs(clean, inputs, generator=generator)
        velocity = core.score_forward(self.critic.model, inputs, noisy, sigma)
        loss = core.critic_loss(velocity, noise=noise, clean=clean, inputs=inputs)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("non-finite Wan Ref2VA SGF critic loss")
        loss.backward()
        stats["critic_grad_norm"] = finish_update(self.critic, self.topology)
        self.global_step += 1
        stats.update(
            loss_critic=float(loss.detach()),
            student_step=float(self.student_step),
            loss=stats["loss_student"] if update_student else float(loss.detach()),
            lr=float(self.student.optimizer.param_groups[0]["lr"]),
            critic_lr=float(self.critic.optimizer.param_groups[0]["lr"]),
            grad_norm=stats["critic_grad_norm"],
        )
        return stats

    def run(self) -> int:
        from solarwm.training import JsonlEventSink, StepPolicy

        policy = StepPolicy(
            max_steps=self.max_steps,
            save_every=int(self.checkpoint_cfg.get("save_every_steps", 0)),
            save_steps=tuple(int(step) for step in self.checkpoint_cfg.get("save_steps", ())),
        )
        sink = (
            JsonlEventSink(self.output_dir / "training-events.jsonl")
            if self.topology.is_main
            else None
        )
        tracking = self.config["runtime"].get("tracking") or {}
        wandb = None
        if self.topology.is_main and bool(tracking.get("enabled", False)):
            from solarwm.training.wandb_sink import WandbEventSink

            wandb = WandbEventSink(
                tracking, output_dir=str(self.output_dir), resolved_config=dict(self.config)
            )
        if self.topology.is_main:
            identities = {
                role.name: role.identity for role in [*self.roles, self.teacher] if role is not None
            }
            print(
                f"[wan-ref2va] stage={self.stage} family={self.spec.family} roles={identities}",
                flush=True,
            )
        try:
            while self.global_step < self.max_steps:
                started = time.monotonic()
                torch.cuda.reset_peak_memory_stats(self.topology.device)
                stats = self.sgf_step() if self.stage == "stage2" else self.train_step()
                stats["compute_time_s"] = time.monotonic() - started
                stats["peak_allocated_gib"] = (
                    torch.cuda.max_memory_allocated(self.topology.device) / 2**30
                )
                step = self.global_step
                if self.topology.is_main:
                    body = " ".join(f"{key}={value:.6g}" for key, value in sorted(stats.items()))
                    print(f"[wan-ref2va] step={step} {body}", flush=True)
                if sink is not None:
                    sink({"event": "train_step", "step": step, **stats})
                if wandb is not None:
                    losses = (
                        {
                            "sgf_critic": stats["loss_critic"],
                            **(
                                {"sgf_student": stats["loss_student"]}
                                if stats["student_updated"]
                                else {}
                            ),
                        }
                        if self.stage == "stage2"
                        else {self.stage: stats["loss"]}
                    )
                    wandb(
                        {
                            "event": "optimizer_step",
                            "step": step,
                            "losses": losses,
                            "lr": stats["lr"],
                            "gradient_norm": stats["grad_norm"],
                            "compute_time_s": stats["compute_time_s"],
                            "peak_allocated_gib": stats["peak_allocated_gib"],
                        }
                    )
                if policy.should_save(step) or step == self.max_steps:
                    directory = self.save_checkpoint(step)
                    if sink is not None:
                        sink({"event": "checkpoint", "step": step, "path": str(directory)})
        finally:
            if wandb is not None:
                wandb.finish()
        return 0


def run_ref2va_training(config: Mapping[str, Any]) -> int:
    try:
        return Ref2VAWanTrainer(config).run()
    finally:
        cleanup()


# -------------------------------------------------------------------------------- inference


def _write_mp4(pixels_tchw: Any, path: Path, *, fps: int) -> None:
    import imageio.v2 as imageio
    import numpy as np

    frames = pixels_tchw.float().add(1).mul(127.5).round().clamp(0, 255).to(torch.uint8)
    frames = frames.permute(0, 2, 3, 1).cpu().numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(
        str(path),
        [np.asarray(frame) for frame in frames],
        fps=int(fps),
        quality=8,
        macro_block_size=1,
        ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
    )


def load_vae(config: Mapping[str, Any], device: Any) -> Any:
    from ..runtime.components import Wan5BVAE, WanA14BVAE

    family = str(config["model"]["family"])
    weights = str(config["model"]["vae_weights"])
    vae = Wan5BVAE(weights) if family == "wan22_ti2v_5b" else WanA14BVAE(weights)
    return vae.to(device, dtype=torch.bfloat16)


def run_ref2va_inference(config: Mapping[str, Any]) -> int:
    """Data-parallel generation of a sample list: every rank holds a whole bf16 model."""

    topology = initialize()
    try:
        inference = config["inference"]
        stage = str(inference["stage"])
        spec = Ref2VAWanSpec.from_config(config)
        role = build_role(
            config,
            "student",
            anyflow=stage in ANYFLOW_STAGES,
            trainable=False,
            init=inference["checkpoint"],
            topology=topology,
            distributed=False,
        )
        core = CORES[stage](role.model, topology.device, config)
        vae = load_vae(config, topology.device)
        output = Path(str(config["runtime"]["output_dir"]))
        paths = sample_paths(str(inference.get("data_path", config["data"]["data_path"])))
        limit = inference.get("max_samples")
        paths = paths[: int(limit)] if limit else paths
        steps = int(inference["num_inference_steps"])
        seed = int(inference.get("seed", 0))
        fps = int(inference.get("fps", 24 if spec.family == "wan22_ti2v_5b" else 16))
        use_oc = bool(config["data"].get("oc_reference", False))
        records = []
        for index in range(topology.rank, len(paths), topology.world):
            path = paths[index]
            sample = read_sample(path, spec, use_oc=use_oc)
            started = time.monotonic()
            latents = core.generate(sample, noise_seed=seed + index, num_inference_steps=steps)
            with torch.no_grad():
                pixels = vae.decode(
                    latents.permute(0, 2, 1, 3, 4).to(torch.bfloat16), use_cache=False
                )[0]
            video = output / "videos" / f"{sample.sample_id}.mp4"
            _write_mp4(pixels, video, fps=fps)
            if bool(inference.get("save_latents", True)):
                torch.save(
                    latents.to(torch.bfloat16).cpu(),
                    output / "videos" / f"{sample.sample_id}.latent.pt",
                )
            record = {
                "sample_id": sample.sample_id,
                "video": str(video),
                "source": str(path),
                "frames": int(pixels.shape[0]),
                "seconds": time.monotonic() - started,
                "oc": len(sample.pictures) > 1,
                "proxy_modality": sample.info.get("proxy_modality"),
            }
            records.append(record)
            print(f"[wan-ref2va-infer] rank={topology.rank} {json.dumps(record)}", flush=True)
            core.release_caches()
        manifest = output / f"manifest-rank{topology.rank:03d}.jsonl"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text("".join(json.dumps(record) + "\n" for record in records))
        barrier()
        if topology.is_main:
            (output / "COMPLETE.json").write_text(
                json.dumps(
                    {
                        "stage": stage,
                        "checkpoint": dict(inference["checkpoint"]),
                        "samples": len(paths),
                    }
                )
            )
        return 0
    finally:
        cleanup()


__all__ = [
    "ParameterEMA",
    "Ref2VAWanTrainer",
    "build_role",
    "load_role_weights",
    "run_ref2va_inference",
    "run_ref2va_training",
]
