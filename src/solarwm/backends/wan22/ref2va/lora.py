"""LoRA on the Wan self-attention projections (``main_attention_qkvo``, as in the H3 runs).

Adapters hold fp32 ``A``/``B`` outside FSDP: the frozen base is sharded, the adapters are
replicated and their gradients averaged over data parallel by :func:`sync_gradients`.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping
from typing import Any

import torch
import torch.nn as nn

TARGETS = {
    "main_attention_qkvo": re.compile(r"^blocks\.\d+\.self_attn\.(q|k|v|o)$"),
    "attention_qkvo": re.compile(r"^blocks\.\d+\.(self_attn|cross_attn)\.(q|k|v|o)$"),
    "attention_ffn": re.compile(r"^blocks\.\d+\.(self_attn\.(q|k|v|o)|ffn\.(0|2))$"),
}


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, *, rank: int, alpha: float) -> None:
        super().__init__()
        self.base = base
        self.rank = int(rank)
        self.scale = float(alpha) / float(rank)
        self.lora_A = nn.Parameter(torch.empty(self.rank, base.in_features, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, self.rank, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: Any) -> Any:
        delta = (x @ self.lora_A.to(x.dtype).t()) @ self.lora_B.to(x.dtype).t()
        return self.base(x) + delta * self.scale


def inject_lora(model: nn.Module, *, target: str, rank: int, alpha: float) -> tuple[str, ...]:
    """Replace every matching ``nn.Linear`` with a zero-initialized LoRA wrapper."""

    pattern = TARGETS.get(target)
    if pattern is None:
        raise ValueError(f"unknown LoRA target {target!r}; expected one of {sorted(TARGETS)}")
    names = tuple(
        name
        for name, module in model.named_modules()
        if pattern.match(name) and isinstance(module, nn.Linear)
    )
    if not names:
        raise ValueError(f"LoRA target {target!r} matches no linear layer")
    for name in names:
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name)
        setattr(parent, child, LoRALinear(getattr(parent, child), rank=rank, alpha=alpha))
    return names


def adapter_state(model: nn.Module, *, extra_prefixes: Iterable[str] = ()) -> dict[str, Any]:
    """Adapter tensors (and any fully trained extra modules) keyed by clean module paths."""

    prefixes = tuple(extra_prefixes)
    state = {}
    for name, parameter in named_trainable(model):
        if ".lora_" in name or name.startswith(prefixes):
            state[name] = parameter.detach().float().cpu().contiguous()
    return state


def clean_name(name: str) -> str:
    for token in ("_fsdp_wrapped_module.", "_checkpoint_wrapped_module.", "module."):
        name = name.replace(token, "")
    return name


def named_trainable(model: nn.Module) -> list[tuple[str, nn.Parameter]]:
    return [
        (clean_name(name), parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]


def load_adapter_state(model: nn.Module, state: Mapping[str, Any], *, strict: bool = True) -> int:
    """Copy adapter tensors into the matching trainable parameters; returns how many loaded."""

    targets = dict(named_trainable(model))
    missing = sorted(set(targets) - set(state))
    unexpected = sorted(set(state) - set(targets))
    if strict and (missing or unexpected):
        raise ValueError(
            f"adapter state mismatch: missing={missing[:6]} unexpected={unexpected[:6]}"
        )
    loaded = 0
    with torch.no_grad():
        for name, value in state.items():
            if name in targets:
                if tuple(targets[name].shape) != tuple(value.shape):
                    raise ValueError(
                        f"adapter {name} has shape {tuple(value.shape)}, "
                        f"model {tuple(targets[name].shape)}"
                    )
                targets[name].copy_(value.to(targets[name].device, targets[name].dtype))
                loaded += 1
    return loaded


@torch.no_grad()
def merge_lora(model: nn.Module) -> int:
    """Fold every adapter into its base weight and unwrap it (for frozen teachers and export)."""

    merged = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, LoRALinear):
            delta = (module.lora_B.float() @ module.lora_A.float()) * module.scale
            module.base.weight.add_(
                delta.to(device=module.base.weight.device, dtype=module.base.weight.dtype)
            )
            parent_name, _, child = name.rpartition(".")
            setattr(model.get_submodule(parent_name) if parent_name else model, child, module.base)
            merged += 1
    return merged


def sync_gradients(parameters: Iterable[nn.Parameter]) -> None:
    """Average replicated adapter gradients over all ranks in one flat bucket."""

    import torch.distributed as dist

    values = [parameter for parameter in parameters if parameter.grad is not None]
    if not values or not dist.is_initialized() or dist.get_world_size() == 1:
        return
    flat = torch.cat([parameter.grad.reshape(-1) for parameter in values])
    dist.all_reduce(flat, op=dist.ReduceOp.SUM)
    flat.div_(dist.get_world_size())
    offset = 0
    for parameter in values:
        count = parameter.grad.numel()
        parameter.grad.copy_(flat[offset : offset + count].view_as(parameter.grad))
        offset += count


__all__ = [
    "TARGETS",
    "LoRALinear",
    "adapter_state",
    "clean_name",
    "inject_lora",
    "load_adapter_state",
    "merge_lora",
    "named_trainable",
    "sync_gradients",
]
