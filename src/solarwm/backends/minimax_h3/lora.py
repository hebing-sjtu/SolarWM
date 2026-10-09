"""Exact PEFT LoRA-384 target and checkpoint contract for MiniMax-H3."""

from __future__ import annotations

import json
import re
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from solarwm.errors import BackendContractError

H3_ROLE_SPLIT_SHARED = "shared"
H3_ROLE_SPLIT_SGF_PLUS = "sgf_plus"
H3_CONTEXT_ADAPTER = "context"
H3_LORA_TARGET_COUNT = 312
H3_LORA_TRAINABLE_PARAMETERS = 2_075_394_048
H3_LORA_SUFFIXES = (
    "attn.to_q",
    "attn.to_k",
    "attn.to_v",
    "attn.to_out.0",
    "ff.net.0.proj",
    "ff.net.2",
)
H3_PROXY_LORA_TARGET_COUNT = 200
H3_PROXY_LORA_SUFFIXES = H3_LORA_SUFFIXES[:4]


def _indexed_prefixes(
    modules: Mapping[str, Any], pattern: str, expected: int, label: str
) -> tuple[str, ...]:
    expression = re.compile(pattern)
    found: dict[int, str] = {}
    for name in modules:
        match = expression.fullmatch(name)
        if match:
            index = int(match.group(1))
            if index in found:
                raise BackendContractError(f"duplicate {label} block index {index}")
            found[index] = name
    required = set(range(expected))
    if set(found) != required:
        raise BackendContractError(
            f"{label} topology differs: missing={sorted(required - set(found))} "
            f"extra={sorted(set(found) - required)}"
        )
    return tuple(found[index] for index in range(expected))


def discover_h3_lora_targets(model: Any, *, target: str = "block_qkvo_ffn") -> tuple[str, ...]:
    """Discover and audit all 50 main + 2 refiner block QKVO/FFN linears."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional local dependency
        raise BackendContractError("PyTorch is required for H3 LoRA discovery") from exc
    modules = dict(model.named_modules())
    main = _indexed_prefixes(
        modules,
        r"(?:.*\.)?transformer_blocks\.(\d+)",
        50,
        "MiniMax-H3 main transformer",
    )
    if target == "main_attention_qkvo":
        blocks = main
        suffixes = H3_PROXY_LORA_SUFFIXES
        expected_count = H3_PROXY_LORA_TARGET_COUNT
    elif target == "block_qkvo_ffn":
        refiner = _indexed_prefixes(
            modules,
            r"(?:.*\.)?token_refiner\.refiner_blocks\.(\d+)",
            2,
            "MiniMax-H3 token refiner",
        )
        blocks = main + refiner
        suffixes = H3_LORA_SUFFIXES
        expected_count = H3_LORA_TARGET_COUNT
    else:
        raise BackendContractError(f"unsupported H3 LoRA target profile {target!r}")
    for block in blocks:
        attention = modules.get(f"{block}.attn")
        if attention is None or getattr(attention, "fused_projections", None) is not False:
            raise BackendContractError(f"H3 LoRA requires split Q/K/V projections at {block!r}")
    targets = tuple(sorted(f"{block}.{suffix}" for block in blocks for suffix in suffixes))
    if len(targets) != expected_count or len(set(targets)) != len(targets):
        raise AssertionError("internal H3 LoRA target-count error")
    invalid = [
        name
        for name in targets
        if name not in modules or not isinstance(modules[name], torch.nn.Linear)
    ]
    if invalid:
        raise BackendContractError(
            f"H3 LoRA expected nn.Linear targets; first invalid={invalid[:8]}"
        )
    return targets


@dataclass
class H3LoRARuntime:
    model: Any
    targets: tuple[str, ...]
    parameter_by_key: OrderedDict[str, Any]
    peft_config: Any
    peft_module: Any
    base_identity: Mapping[str, Any]
    rank: int
    alpha: int
    # SGF+ only: context-adapter state key -> the shared adapter key it mirrors.
    context_keys: Mapping[str, str] = field(default_factory=dict)

    @property
    def role_split(self) -> str:
        return H3_ROLE_SPLIT_SGF_PLUS if self.context_keys else H3_ROLE_SPLIT_SHARED

    @property
    def parameters(self) -> tuple[Any, ...]:
        return tuple(self.parameter_by_key.values())

    def role_parameters(self) -> dict[str, tuple[Any, ...]]:
        """Trainable tensors per SGF+ role: ``denoise`` (shared adapter) and ``context``."""

        return {
            "denoise": tuple(
                value
                for key, value in self.parameter_by_key.items()
                if key not in self.context_keys
            ),
            "context": tuple(self.parameter_by_key[key] for key in self.context_keys),
        }

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters)

    def state_dict(self) -> OrderedDict[str, Any]:
        return OrderedDict(
            (key, parameter.detach()) for key, parameter in self.parameter_by_key.items()
        )

    def load_state_dict(self, values: Mapping[str, Any], *, broadcast: bool = True) -> None:
        import torch
        import torch.distributed as dist

        if set(values) != set(self.parameter_by_key):
            raise BackendContractError(
                "H3 LoRA checkpoint keys differ: "
                f"missing={sorted(set(self.parameter_by_key) - set(values))[:8]} "
                f"extra={sorted(set(values) - set(self.parameter_by_key))[:8]}"
            )
        with torch.no_grad():
            for key, parameter in self.parameter_by_key.items():
                value = values[key]
                if tuple(value.shape) != tuple(parameter.shape):
                    raise BackendContractError(f"H3 LoRA shape differs for {key!r}")
                if value.dtype != parameter.dtype:
                    raise BackendContractError(f"H3 LoRA dtype differs for {key!r}")
                parameter.copy_(value.to(device=parameter.device))
                if broadcast and dist.is_available() and dist.is_initialized():
                    dist.broadcast(parameter, src=0)

    def metadata(self) -> dict[str, Any]:
        split = (
            {"role_split": self.role_split, "context_keys": dict(self.context_keys)}
            if self.context_keys
            else {}
        )
        return {
            **split,
            "schema": "solarwm.minimax-h3-lora.v1",
            "peft_version": str(self.peft_module.__version__),
            "rank": self.rank,
            "alpha": self.alpha,
            "target_count": len(self.targets),
            "target_modules": list(self.targets),
            "state_keys": list(self.parameter_by_key),
            "state_shapes": {
                key: list(parameter.shape) for key, parameter in self.parameter_by_key.items()
            },
            "state_dtypes": {
                key: str(parameter.dtype).removeprefix("torch.")
                for key, parameter in self.parameter_by_key.items()
            },
            "trainable_parameters": self.parameter_count,
            "base_identity": json.loads(json.dumps(dict(self.base_identity), sort_keys=True)),
        }


def inject_h3_lora(
    model: Any,
    adapter_cfg: Mapping[str, Any],
    *,
    base_identity: Mapping[str, Any],
    role_split: str = H3_ROLE_SPLIT_SHARED,
) -> tuple[Any, H3LoRARuntime]:
    """Inject standard BF16 PEFT LoRA and verify its realized topology/size.

    ``role_split="sgf_plus"`` adds a second adapter on the main transformer
    blocks for SGF+ context writing; only causal students may request it.
    """

    try:
        import peft
        import torch
        import torch.distributed.tensor
    except ImportError as exc:  # pragma: no cover - optional runtime dependency
        raise BackendContractError("H3 LoRA requires peft==0.20.0") from exc

    rank = int(adapter_cfg.get("rank", 0))
    alpha = int(adapter_cfg.get("alpha", 0))
    target_profile = str(adapter_cfg.get("target", "block_qkvo_ffn"))
    expected_rank = 128 if target_profile == "main_attention_qkvo" else 384
    if rank != expected_rank or alpha != expected_rank:
        raise BackendContractError(
            f"H3 {target_profile} adapter requires rank=alpha={expected_rank}"
        )
    targets = discover_h3_lora_targets(model, target=target_profile)
    configuration = peft.LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=float(adapter_cfg.get("dropout", 0.0)),
        target_modules=list(targets),
        bias="none",
        init_lora_weights=True,
    )
    wrapped = peft.get_peft_model(
        model,
        configuration,
        adapter_name="default",
        autocast_adapter_dtype=False,
    )
    realized = tuple(sorted(str(name) for name in wrapped.base_model.targeted_module_names))
    if realized != targets:
        raise BackendContractError(
            "PEFT realized a different H3 topology: "
            f"missing={sorted(set(targets) - set(realized))[:8]} "
            f"extra={sorted(set(realized) - set(targets))[:8]}"
        )
    wrapped.peft_config["default"].target_modules = set(targets)
    for parameter in wrapped.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.to(torch.bfloat16)
    keep_vars = wrapped.state_dict(keep_vars=True)
    standard = peft.get_peft_model_state_dict(
        wrapped,
        state_dict=keep_vars,
        adapter_name="default",
        save_embedding_layers=False,
    )
    parameter_by_key: OrderedDict[str, Any] = OrderedDict()
    for key in sorted(standard):
        value = standard[key]
        if not isinstance(value, torch.nn.Parameter):
            raise BackendContractError(f"PEFT state {key!r} is not a Parameter")
        parameter_by_key[key] = value
    trainable = tuple(parameter for parameter in wrapped.parameters() if parameter.requires_grad)
    if {id(value) for value in trainable} != {id(value) for value in parameter_by_key.values()}:
        raise BackendContractError("all and only H3 LoRA parameters must be trainable")
    expected_targets = int(adapter_cfg.get("expected_target_linear_modules", len(targets)))
    if len(targets) != expected_targets:
        raise BackendContractError(
            f"H3 LoRA discovered {len(targets)} targets, expected {expected_targets}"
        )
    if len(parameter_by_key) != 2 * len(targets):
        raise BackendContractError("H3 LoRA must expose one A/B tensor pair per target")
    runtime = H3LoRARuntime(
        model=wrapped,
        targets=targets,
        parameter_by_key=parameter_by_key,
        peft_config=wrapped.peft_config["default"],
        peft_module=peft,
        base_identity=dict(base_identity),
        rank=rank,
        alpha=alpha,
    )
    expected = adapter_cfg.get("expected_trainable_parameters")
    if expected is not None and runtime.parameter_count != int(expected):
        raise BackendContractError(
            f"H3 LoRA trainable parameters={runtime.parameter_count:,}, expected={int(expected):,}"
        )
    if role_split == H3_ROLE_SPLIT_SGF_PLUS:
        if target_profile != "block_qkvo_ffn":
            raise BackendContractError("SGF+ role split requires the block_qkvo_ffn adapter")
        from .lora_roles import install_context_adapter

        main = tuple(target for target in targets if "transformer_blocks." in target)
        model.h3_lora_role_split = True
        context = install_context_adapter(
            wrapped,
            peft_module=peft,
            context_targets=main,
            blocks=tuple(model.transformer_blocks),
            rank=rank,
            alpha=alpha,
            dropout=float(adapter_cfg.get("dropout", 0.0)),
        )
        context_keys = {}
        for key, parameter in context.items():
            mirrored = key.replace(f".{H3_CONTEXT_ADAPTER}.weight", ".weight")
            if mirrored not in parameter_by_key:
                raise BackendContractError(f"SGF+ context tensor {key!r} has no shared twin")
            if tuple(parameter.shape) != tuple(parameter_by_key[mirrored].shape):
                raise BackendContractError(f"SGF+ context tensor {key!r} differs in shape")
            context_keys[key] = mirrored
        runtime.parameter_by_key = OrderedDict((*parameter_by_key.items(), *context.items()))
        runtime.context_keys = context_keys
        trainable = tuple(
            parameter for parameter in wrapped.parameters() if parameter.requires_grad
        )
        if {id(value) for value in trainable} != {id(value) for value in runtime.parameters}:
            raise BackendContractError("SGF+ must train exactly the shared and context adapters")
    elif role_split != H3_ROLE_SPLIT_SHARED:
        raise BackendContractError(f"unsupported H3 LoRA role split {role_split!r}")
    if any(parameter.dtype != torch.bfloat16 for parameter in runtime.parameters):
        raise BackendContractError("all H3 LoRA parameters must remain BF16")
    return wrapped, runtime


def h3_role_split(model_cfg: Mapping[str, Any]) -> str:
    """The configured student LoRA role split; absent means one shared adapter."""

    return str(model_cfg.get("adapter", {}).get("role_split", H3_ROLE_SPLIT_SHARED))


__all__ = [
    "H3_CONTEXT_ADAPTER",
    "H3_LORA_SUFFIXES",
    "H3_LORA_TARGET_COUNT",
    "H3_LORA_TRAINABLE_PARAMETERS",
    "H3_PROXY_LORA_SUFFIXES",
    "H3_PROXY_LORA_TARGET_COUNT",
    "H3_ROLE_SPLIT_SGF_PLUS",
    "H3_ROLE_SPLIT_SHARED",
    "H3LoRARuntime",
    "discover_h3_lora_targets",
    "h3_role_split",
    "inject_h3_lora",
]
