"""SGF+ split of context-writing and denoising LoRA parameters."""

from __future__ import annotations

import copy
import json
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from solarwm.backends.minimax_h3.config import validate_h3_config
from solarwm.errors import BackendContractError, ConfigurationError

EXAMPLES = Path(__file__).resolve().parents[3] / "configs/examples/minimax_h3"


def _yaml(name: str) -> dict:
    return yaml.safe_load((EXAMPLES / name).read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    "name",
    [
        "stage2-158f-lora384-w6-sp4.yaml",
        "infer-stage2-158f-sp4.yaml",
        "infer-stage2-source-length-sp8.yaml",
    ],
)
def test_stage2_examples_accept_both_role_splits(name: str) -> None:
    config = _yaml(name)
    assert config["model"]["adapter"]["role_split"] == "shared"
    validate_h3_config(config)
    config["model"]["adapter"]["role_split"] = "sgf_plus"
    validate_h3_config(config)
    config["model"]["adapter"]["role_split"] = "split"
    with pytest.raises(ConfigurationError, match="shared or sgf_plus"):
        validate_h3_config(config)


@pytest.mark.parametrize(
    "name",
    [
        "stage1-158f-lora384-w6-sp2.yaml",
        "stage0p5-124f-ref2va-omni-mixed-low-704p-sp1.yaml",
    ],
)
def test_sgf_plus_is_stage2_only(name: str) -> None:
    config = _yaml(name)
    config["model"]["adapter"]["role_split"] = "sgf_plus"
    with pytest.raises(ConfigurationError, match="requires Stage2"):
        validate_h3_config(config)


def test_sgf_plus_checkpoints_carry_a_distinct_parameterization() -> None:
    from solarwm.backends.minimax_h3.runtime import _checkpoint_contract
    from solarwm.checkpoint import assert_resume_compatible
    from solarwm.errors import CheckpointError

    shared = _yaml("stage2-158f-lora384-w6-sp4.yaml")
    split = copy.deepcopy(shared)
    split["model"]["adapter"]["role_split"] = "sgf_plus"
    contracts = [
        _checkpoint_contract(encoder_profile={}, silence_profile={}, base_model={}, config=config)
        for config in (shared, split)
    ]
    assert contracts[0].parameterization == "peft-lora-r384-alpha384"
    assert contracts[1].parameterization == "peft-lora-r384-alpha384-sgf-plus"
    with pytest.raises(CheckpointError):
        assert_resume_compatible(contracts[0], contracts[1])


# ----------------------------------------------------------------------
# Cross-stage weights
# ----------------------------------------------------------------------


def _split_runtime(torch):
    from solarwm.backends.minimax_h3.lora import H3LoRARuntime

    shared = torch.nn.Parameter(torch.zeros(2, 2, dtype=torch.bfloat16))
    context = torch.nn.Parameter(torch.zeros(2, 2, dtype=torch.bfloat16))
    return H3LoRARuntime(
        model=None,
        targets=("blk",),
        parameter_by_key=OrderedDict(
            [("blk.lora_A.weight", shared), ("blk.lora_A.context.weight", context)]
        ),
        peft_config=None,
        peft_module=SimpleNamespace(__version__="0.20.0"),
        base_identity={},
        rank=384,
        alpha=384,
        context_keys={"blk.lora_A.context.weight": "blk.lora_A.weight"},
    )


def _checkpoint(root: Path, torch, *, stage: str, parameterization: str, state: dict) -> dict:
    root.mkdir()
    (root / "COMPLETE.json").write_text("{}")
    extras = {
        "encoder_profile": {"pixel_frames": 158, "height": 768, "width": 1344},
        "chunk_latents": 5,
        "window_chunks": 6,
        "target_latents": 45 if stage == "stage1" else 47,
        "rollout_latents": 50,
        "student_rope_mode": "native_absolute" if stage == "stage1" else "sliding_local",
    }
    manifest = {
        "step": 3000,
        "contract": {
            "family": "minimax_h3",
            "stage": stage,
            "camera_translation_transform": "logd4",
            "parameterization": parameterization,
            "extras": extras,
        },
    }
    (root / "checkpoint-manifest.json").write_text(json.dumps(manifest))
    torch.save({"state": state}, root / "adapter.pt")
    return {"path": str(root), "stage": stage, "weight_source": "live"}


def test_shared_source_initializes_both_roles_and_split_source_keeps_them(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from solarwm.backends.minimax_h3.lora import H3LoRARuntime
    from solarwm.backends.minimax_h3.weights import load_initial_weights

    lora = _split_runtime(torch)
    stage1 = _checkpoint(
        tmp_path / "stage1",
        torch,
        stage="stage1",
        parameterization="peft-lora-r384-alpha384",
        state={"blk.lora_A.weight": torch.full((2, 2), 0.5)},
    )
    assert load_initial_weights(stage1, lora) == "stage1:live:step=3000"
    shared, context = lora.parameter_by_key.values()
    assert torch.equal(shared, context)
    assert shared.data_ptr() != context.data_ptr()

    stage2 = _checkpoint(
        tmp_path / "stage2",
        torch,
        stage="stage2",
        parameterization="peft-lora-r384-alpha384-sgf-plus",
        state={
            "blk.lora_A.weight": torch.full((2, 2), 1.0),
            "blk.lora_A.context.weight": torch.full((2, 2), 2.0),
        },
    )
    load_initial_weights(stage2, lora)
    assert float(shared[0, 0]) == 1.0 and float(context[0, 0]) == 2.0

    single = H3LoRARuntime(
        model=None,
        targets=("blk",),
        parameter_by_key=OrderedDict([("blk.lora_A.weight", torch.nn.Parameter(shared.clone()))]),
        peft_config=None,
        peft_module=SimpleNamespace(__version__="0.20.0"),
        base_identity={},
        rank=384,
        alpha=384,
    )
    with pytest.raises(BackendContractError, match="role_split=sgf_plus"):
        load_initial_weights(stage2, single)
    assert "role_split" not in single.metadata()
    assert lora.metadata()["role_split"] == "sgf_plus"


# ----------------------------------------------------------------------
# Routing on a miniature H3 topology (50 main + 2 refiner blocks)
# ----------------------------------------------------------------------


def _toy_h3(torch):
    nn = torch.nn
    width = 4

    class Projection(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Linear(width, width)

    class Attention(nn.Module):
        fused_projections = False

        def __init__(self) -> None:
            super().__init__()
            self.to_q = nn.Linear(width, width)
            self.to_k = nn.Linear(width, width)
            self.to_v = nn.Linear(width, width)
            self.to_out = nn.ModuleList([nn.Linear(width, width), nn.Identity()])

    class FeedForward(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.ModuleList([Projection(), nn.Identity(), nn.Linear(width, width)])

    class Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attn = Attention()
            self.ff = FeedForward()

        def forward(self, states, control):
            del control
            attn = self.attn
            mixed = attn.to_q(states) * torch.sigmoid(attn.to_k(states)) + attn.to_v(states)
            states = states + attn.to_out[0](mixed)
            return states + self.ff.net[2](torch.tanh(self.ff.net[0].proj(states)))

    class Refiner(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.refiner_blocks = nn.ModuleList([Block(), Block()])

    class Toy(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.token_refiner = Refiner()
            self.transformer_blocks = nn.ModuleList([Block() for _ in range(50)])

        def forward(self, states, control):
            for block in self.token_refiner.refiner_blocks:
                states = block(states, None)
            for block in self.transformer_blocks:
                states = block(states, control)
            return states

    torch.manual_seed(0)
    return Toy()


ADAPTER = {"type": "lora", "target": "block_qkvo_ffn", "rank": 384, "alpha": 384, "dropout": 0.0}


def _control(route):
    from solarwm.backends.minimax_h3.model import H3AttentionControl

    return H3AttentionControl(None, None, None, None, None, None, None, lora_route=route)


def _randomize(torch, parameters, seed):
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in parameters:
            parameter.copy_(0.05 * torch.randn(parameter.shape, generator=generator))


def test_rows_receive_exactly_their_role_adapter_and_gradients_stay_separate() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    from solarwm.backends.minimax_h3.lora import inject_h3_lora

    base = _toy_h3(torch)
    split_model, split = inject_h3_lora(
        copy.deepcopy(base), ADAPTER, base_identity={}, role_split="sgf_plus"
    )
    assert len(split.targets) == 312 and len(split.context_keys) == 600
    assert all(".transformer_blocks." in key for key in split.context_keys)
    roles = split.role_parameters()
    _randomize(torch, roles["denoise"], 1)
    _randomize(torch, roles["context"], 2)

    states = torch.randn(1, 6, 4)
    context_rows = torch.tensor([1, 4])
    route = torch.zeros(1, 6, 1)
    route[:, context_rows] = 1
    output = split_model(states, _control(route))

    # A shared-adapter model loaded with either role's weights reproduces those rows exactly.
    for role, rows in (("denoise", [0, 2, 3, 5]), ("context", [1, 4])):
        reference_model, reference = inject_h3_lora(copy.deepcopy(base), ADAPTER, base_identity={})
        values = {key: value.detach() for key, value in split.parameter_by_key.items()}
        if role == "context":
            values.update({twin: values[key] for key, twin in split.context_keys.items()})
        reference.load_state_dict(
            {key: values[key] for key in reference.parameter_by_key}, broadcast=False
        )
        expected = reference_model(states, None)
        torch.testing.assert_close(output[:, rows], expected[:, rows], rtol=0, atol=1e-6)

    main_denoise = [
        value
        for key, value in split.parameter_by_key.items()
        if key not in split.context_keys and ".transformer_blocks." in key
    ]
    for rows, silent in (([1, 4], main_denoise), ([0, 2, 3, 5], roles["context"])):
        split_model.zero_grad(set_to_none=True)
        split_model(states, _control(route))[:, rows].square().sum().backward()
        assert all(parameter.grad is None or not parameter.grad.any() for parameter in silent)
    assert any(parameter.grad is not None and parameter.grad.any() for parameter in main_denoise)


def test_routing_survives_activation_recomputation_and_requires_a_mask() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        CheckpointImpl,
        apply_activation_checkpointing,
        checkpoint_wrapper,
    )

    from solarwm.backends.minimax_h3.lora import inject_h3_lora

    model, lora = inject_h3_lora(_toy_h3(torch), ADAPTER, base_identity={}, role_split="sgf_plus")
    roles = lora.role_parameters()
    _randomize(torch, roles["denoise"], 3)
    _randomize(torch, roles["context"], 4)
    states = torch.randn(1, 5, 4)
    route = torch.tensor([0.0, 1.0, 1.0, 0.0, 0.0]).view(1, 5, 1)

    def gradients():
        model.zero_grad(set_to_none=True)
        model(states, _control(route)).square().sum().backward()
        return [parameter.grad.clone() for parameter in lora.parameters]

    eager = gradients()
    main_blocks = {id(block) for block in model.base_model.model.transformer_blocks}
    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=lambda module: checkpoint_wrapper(
            module, checkpoint_impl=CheckpointImpl.NO_REENTRANT
        ),
        check_fn=lambda module: id(module) in main_blocks,
    )
    for left, right in zip(eager, gradients(), strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    with pytest.raises(BackendContractError, match="role mask"):
        model(states, _control(None))


def test_student_forward_declares_context_rows_per_pass() -> None:
    torch = pytest.importorskip("torch")
    from solarwm.backends.minimax_h3.sgf_rollout import H3SGFInputs, h3_student_forward

    class Captured(Exception):
        pass

    class Student(torch.nn.Module):
        h3_lora_role_split = True

        def forward(self, **kwargs):
            self.kwargs = kwargs
            raise Captured

    inputs = H3SGFInputs(
        prompt=torch.zeros(1, 3, 8),
        text_tags=torch.tensor([0, 1, 1]),
        anchor_rows=torch.zeros(1, 1, 96),
        audio_rows=torch.zeros(1, 6, 32),
        audio_timestep=torch.tensor(0.5),
        camera_viewmats=torch.eye(4).repeat(1, 51, 1, 1),
        camera_K=torch.eye(3).repeat(1, 51, 1, 1),
        latent_height=2,
        latent_width=2,
    )
    student = Student()
    chunk = torch.zeros(1, 24, 5, 2, 2)
    passes = {
        "denoise": dict(noisy=chunk),
        "commit": dict(noisy=chunk, commit_cache=True),
        "replay": dict(noisy=torch.zeros(1, 24, 50, 2, 2), clean=torch.zeros(1, 24, 50, 2, 2)),
    }
    for name, kwargs in passes.items():
        noisy = kwargs.pop("noisy")
        with pytest.raises(Captured):
            h3_student_forward(student, inputs, noisy, torch.tensor(0.5), **kwargs)
        rows = student.kwargs["lora_context_rows"]
        layout = inputs.layout("replay" if name == "replay" else "rollout")
        expected = {
            "denoise": layout.noisy_video_indices[:0],
            "commit": layout.noisy_video_indices,
            "replay": layout.clean_video_indices,
        }[name]
        assert torch.equal(torch.as_tensor(rows), torch.as_tensor(expected))
    assert int(inputs.layout("replay").clean_video_indices.numel()) == 50

    class Shared(torch.nn.Module):
        def forward(self, **kwargs):
            self.kwargs = kwargs
            raise Captured

    shared = Shared()
    with pytest.raises(Captured):
        h3_student_forward(shared, inputs, chunk, torch.tensor(0.5), commit_cache=True)
    assert "lora_context_rows" not in shared.kwargs
