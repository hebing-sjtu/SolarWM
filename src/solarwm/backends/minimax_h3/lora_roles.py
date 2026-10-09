"""Per-token routing between two H3 LoRA parameter sets for SGF+.

SGF+ gives context writing and denoising independent parameters. The shared
PEFT adapter keeps the denoising role; a second adapter named ``context``
serves the rows whose keys and values later conditioning chunks read: clean
history rows in replay and the committed chunk in a rollout's KV-commit
forward. Both adapters run on every token and their rank-space activations are
multiplied by a 0/1 row mask, so a row receives exactly one LoRA delta.

The mask travels inside each transformer block's attention control argument.
A block pre-hook publishes it to the LoRA hooks, which keeps routing correct
when activation checkpointing recomputes a block during backward.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable, Mapping
from typing import Any

from solarwm.errors import BackendContractError

from .lora import H3_CONTEXT_ADAPTER

# Masks of the block currently executing, as (denoise, context) pairs.
_ACTIVE: list[tuple[Any, Any]] = []


def _block_route(args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> Any:
    from .model import H3AttentionControl

    for value in (*args, *kwargs.values()):
        if isinstance(value, H3AttentionControl):
            if value.lora_route is None:
                break
            return value.lora_route
    raise BackendContractError("SGF+ student forward has no LoRA role mask")


def _enter_block(module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
    context = _block_route(args, kwargs)
    _ACTIVE.append((1 - context, context))
    module._h3_route_depth = getattr(module, "_h3_route_depth", 0) + 1


def _exit_block(module: Any, _args: Any, _output: Any) -> None:
    # always_call also runs after a failed pre-hook, which pushed nothing.
    if getattr(module, "_h3_route_depth", 0):
        module._h3_route_depth -= 1
        _ACTIVE.pop()


def _role_hook(role: int):
    def hook(_module: Any, _inputs: Any, output: Any) -> Any:
        if not _ACTIVE:
            raise BackendContractError("SGF+ LoRA ran outside a routed transformer block")
        mask = _ACTIVE[-1][role]
        if mask.shape[1] != output.shape[1]:
            raise BackendContractError(
                f"SGF+ role mask covers {mask.shape[1]} rows, LoRA input has {output.shape[1]}"
            )
        return output * mask.to(dtype=output.dtype)

    return hook


def install_context_adapter(
    wrapped: Any,
    *,
    peft_module: Any,
    context_targets: Iterable[str],
    blocks: Iterable[Any],
    rank: int,
    alpha: int,
    dropout: float,
) -> OrderedDict[str, Any]:
    """Add the ``context`` adapter, activate both adapters and install the row routing."""

    import torch

    targets = tuple(sorted(context_targets))
    wrapped.add_adapter(
        H3_CONTEXT_ADAPTER,
        peft_module.LoraConfig(
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=list(targets),
            bias="none",
            init_lora_weights=True,
        ),
    )
    wrapped.base_model.set_adapter(["default", H3_CONTEXT_ADAPTER])
    modules = dict(wrapped.named_modules())
    by_target: dict[str, Any] = {}
    for name, module in modules.items():
        if (
            any(name == target or name.endswith(f".{target}") for target in targets)
            and hasattr(module, "lora_A")
            and H3_CONTEXT_ADAPTER in module.lora_A
        ):
            by_target[name] = module
    if len(by_target) != len(targets):
        raise BackendContractError(
            f"SGF+ context adapter reached {len(by_target)} of {len(targets)} targets"
        )
    for module in by_target.values():
        module.lora_A["default"].register_forward_hook(_role_hook(0))
        module.lora_A[H3_CONTEXT_ADAPTER].register_forward_hook(_role_hook(1))
    count = 0
    for block in blocks:
        block.register_forward_pre_hook(_enter_block, with_kwargs=True)
        block.register_forward_hook(_exit_block, always_call=True)
        count += 1
    if not count:
        raise BackendContractError("SGF+ routing found no transformer blocks")
    context: OrderedDict[str, Any] = OrderedDict()
    marker = f".{H3_CONTEXT_ADAPTER}."
    for name, parameter in wrapped.named_parameters():
        if marker in name and ".lora_" in name:
            parameter.data = parameter.data.to(torch.bfloat16)
            context[name] = parameter
    if len(context) != 2 * len(targets):
        raise BackendContractError("SGF+ context adapter must expose one A/B pair per target")
    return OrderedDict(sorted(context.items()))


def student_role_split(student: Any) -> bool:
    """Whether a (possibly FSDP/PEFT-wrapped) student carries the SGF+ context adapter."""

    cached = getattr(student, "_h3_role_split_cached", None)
    if cached is None:
        cached = any(getattr(module, "h3_lora_role_split", False) for module in student.modules())
        student._h3_role_split_cached = cached
    return bool(cached)


def context_rows_mask(rows: Any, *, sequence_length: int, like: Any) -> Any:
    """A ``[1, S, 1]`` 0/1 mask selecting context-writing rows of the packed document."""

    import torch

    mask = torch.zeros(sequence_length, device=like.device, dtype=like.dtype)
    if rows.numel():
        mask[rows.to(device=like.device, dtype=torch.long)] = 1
    return mask.view(1, -1, 1)


__all__ = ["context_rows_mask", "install_context_adapter", "student_role_split"]
