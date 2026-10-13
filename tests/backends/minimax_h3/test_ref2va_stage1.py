from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from solarwm.backends.minimax_h3.config import validate_h3_config
from solarwm.backends.minimax_h3.layout import (
    AUDIO_ROLE,
    CONDITION_ROLE,
    NOISY_VIDEO_ROLE,
    PROXY_VIDEO_ROLE,
)
from solarwm.backends.minimax_h3.mask import build_stage1_window_mask
from solarwm.backends.minimax_h3.proxy_loop import EvalTask, eval_config, loss_summary
from solarwm.backends.minimax_h3.ref2va_layout import (
    build_ref2va_proxy_layout,
    build_ref2va_stage1_layout,
)
from solarwm.backends.minimax_h3.stage1_ref2va import ref2va_row_time_pairs
from solarwm.backends.minimax_h3.torch_mask import build_ref2va_stage1_block_mask
from solarwm.errors import ConfigurationError

ROOT = Path(__file__).resolve().parents[3]
EXAMPLES = ROOT / "configs/examples/minimax_h3"
STAGE1 = EXAMPLES / "stage1-124f-ref2va-omni-mixed-low-704p-sp1.yaml"
STAGE1_INFER = EXAMPLES / "infer-stage1-124f-ref2va-omni-mixed-low-704p-sp8.yaml"
MIXED_INFER = EXAMPLES / "infer-stage0p5-124f-ref2va-omni-mixed-low-704p-sp8.yaml"
TAGS = np.asarray([1, 0, 0, 1], dtype=np.int64)


def _config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _small(**overrides):
    arguments = dict(
        anchor_height=4,
        anchor_width=4,
        proxy_height=2,
        proxy_width=2,
        proxy_frames=12,
        num_latent_frames=10,
        target_height=4,
        target_width=4,
        num_audio_latents=3,
    )
    arguments.update(overrides)
    return build_ref2va_stage1_layout(TAGS, **arguments)


@pytest.mark.parametrize("align", [False, True])
def test_stage1_rollout_window_keeps_stage0p5_positions(align: bool) -> None:
    geometry = dict(
        anchor_height=128,
        anchor_width=232,
        proxy_height=22,
        proxy_width=40,
        target_height=44,
        target_width=80,
        align_proxy_reference_time=align,
    )
    full = build_ref2va_proxy_layout(TAGS, proxy_frames=37, target_frames=37, **geometry)
    window = build_ref2va_stage1_layout(
        TAGS, proxy_frames=37, num_latent_frames=40, chunks=(2, 8), **geometry
    )
    prefix = 4 + 64 * 116
    np.testing.assert_array_equal(window.position_ids[:prefix], full.position_ids[:prefix])
    proxy = window.row_roles == PROXY_VIDEO_ROLE
    assert int(proxy.sum()) == (37 - 10) * 220
    np.testing.assert_array_equal(
        window.position_ids[proxy], full.position_ids[prefix + 10 * 220 : prefix + 37 * 220]
    )
    noisy = window.position_ids[window.noisy_video_indices].reshape(30, 880, 3)
    clean = window.position_ids[window.clean_video_indices].reshape(30, 880, 3)
    np.testing.assert_array_equal(noisy, clean)
    bd_target = full.position_ids[full.target_video_indices].reshape(37, 880, 3)
    np.testing.assert_array_equal(noisy[:27], bd_target[10:])
    assert np.all(np.diff(noisy[:, 0, 0]) > 0)
    np.testing.assert_array_equal(noisy[27:, :, 1:], noisy[:3, :, 1:])
    trained = build_ref2va_stage1_layout(TAGS, proxy_frames=37, num_latent_frames=35, **geometry)
    np.testing.assert_array_equal(
        trained.position_ids[trained.noisy_video_indices].reshape(35, 880, 3), bd_target[:35]
    )


def test_stage1_layout_rejects_partial_chunks_and_windows_past_the_proxy() -> None:
    with pytest.raises(ValueError, match="whole"):
        _small(num_latent_frames=12)
    with pytest.raises(ValueError, match="outside"):
        _small(chunks=(1, 3))
    with pytest.raises(ValueError, match="after the proxy"):
        _small(proxy_frames=9, num_latent_frames=15, chunks=(2, 3))


def test_target_chunks_read_only_their_own_and_earlier_proxy_chunks() -> None:
    layout = _small(proxy_frames=12, num_latent_frames=15, window_chunks=2)
    allowed = build_stage1_window_mask(layout)
    roles, chunks = layout.row_roles, layout.target_video_chunk_ids
    proxy = np.flatnonzero(roles == PROXY_VIDEO_ROLE)
    assert np.bincount(chunks[proxy]).tolist() == [5, 5, 2]
    for query in np.flatnonzero(roles != PROXY_VIDEO_ROLE):
        seen = chunks[proxy][allowed[query, proxy]]
        if roles[query] in (CONDITION_ROLE, AUDIO_ROLE):
            assert seen.size == 0
        else:
            expected = [c for c in range(3) if 0 <= chunks[query] - c < 2]
            assert sorted(set(seen.tolist())) == expected
    for query in proxy:
        keys = np.flatnonzero(allowed[query])
        assert set(roles[keys].tolist()) <= {CONDITION_ROLE, PROXY_VIDEO_ROLE}
    noisy = np.flatnonzero(roles == NOISY_VIDEO_ROLE)
    assert not allowed[np.ix_(proxy, noisy)].any()


@pytest.mark.parametrize(
    "arguments",
    [
        dict(),
        dict(num_latent_frames=15, chunks=(0, 3)),
        dict(num_latent_frames=15, chunks=(1, 3), window_chunks=2),
        dict(num_latent_frames=15, chunks=(2, 3), window_chunks=1),
        dict(num_latent_frames=15, num_proxy_references=2),
        dict(align_proxy_reference_time=True),
    ],
)
def test_ref2va_block_mask_matches_dense_visibility(arguments: dict) -> None:
    source = _small(
        anchor_height=8,
        anchor_width=8,
        proxy_height=4,
        proxy_width=4,
        target_height=8,
        target_width=16,
        num_audio_latents=20,
        **arguments,
    )
    assert source.sequence_length > 2 * 128
    layout = source.to("cpu")
    mask = build_ref2va_stage1_block_mask(layout, device="cpu")
    rows = torch.arange(layout.sequence_length)
    actual = mask.mask_mod(torch.tensor(0), torch.tensor(0), rows[:, None], rows[None, :])
    expected = torch.from_numpy(build_stage1_window_mask(source))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    blocks = -(-layout.sequence_length // 128)
    padded = torch.zeros((blocks * 128, blocks * 128), dtype=torch.bool)
    padded[: layout.sequence_length, : layout.sequence_length] = expected
    outside = torch.ones_like(padded)
    outside[: layout.sequence_length, : layout.sequence_length] = False
    any_tile = padded.reshape(blocks, 128, blocks, 128).any(dim=(1, 3))
    all_tile = (padded | outside).reshape(blocks, 128, blocks, 128).all(dim=(1, 3))

    def listed(counts, indices):
        result = torch.zeros((blocks, blocks), dtype=torch.bool)
        for query in range(blocks):
            result[query, indices[0, 0, query, : counts[0, 0, query]].long()] = True
        return result

    partial = listed(mask.kv_num_blocks, mask.kv_indices)
    full = listed(mask.full_kv_num_blocks, mask.full_kv_indices)
    assert not (partial & full).any()
    assert torch.equal(partial | full, any_tile)
    assert torch.all(all_tile[full])


def test_row_times_hold_conditions_and_given_latents_at_the_keyframe_time() -> None:
    layout = _small().to("cpu")
    times, targets, inverse = ref2va_row_time_pairs(
        layout,
        video_t=torch.tensor([0.3]),
        video_r=torch.tensor([0.6]),
        audio_t=torch.tensor(0.2),
        current_chunk=1,
        given_rows=4,
        device="cpu",
    )
    t, r = times[inverse], targets[inverse]
    roles = layout.row_roles.numpy()
    chunks = layout.target_video_chunk_ids.numpy()
    keyframe = torch.tensor(0.999)
    torch.testing.assert_close(t[layout.condition_video_indices], keyframe.expand(4 + 10))
    assert torch.all(t[layout.text_indices] == 1) and torch.all(t[layout.clean_video_indices] == 1)
    assert torch.all(t[layout.audio_indices] == torch.tensor(0.2))
    noisy = np.flatnonzero(roles == NOISY_VIDEO_ROLE)
    current = noisy[chunks[noisy] == 1]
    assert torch.all(t[current] == torch.tensor(0.3)) and torch.all(r[current] == 0.6)
    given = noisy[:4]
    torch.testing.assert_close(t[given], keyframe.expand(4))
    torch.testing.assert_close(r[given], keyframe.expand(4))
    others = np.setdiff1d(np.arange(layout.sequence_length), np.concatenate((current, given)))
    torch.testing.assert_close(t[others], r[others])


def test_stage1_proxy_examples_validate() -> None:
    contract = validate_h3_config(_config(STAGE1))
    assert (contract.stage, contract.encoded_latents) == ("stage1", 37)
    assert validate_h3_config(_config(STAGE1_INFER)).sequence_parallel_size == 8
    config = _config(STAGE1)
    config["data"]["train_target_latents"] = 37
    with pytest.raises(ConfigurationError, match="train_target_latents"):
        validate_h3_config(config)
    config = _config(STAGE1_INFER)
    config["validation"]["num_inference_steps"] = 30
    with pytest.raises(ConfigurationError, match="num_inference_steps"):
        validate_h3_config(config)
    config = _config(STAGE1)
    config["model"]["attention_backend"] = "flash"
    with pytest.raises(ConfigurationError, match="attention_backend"):
        validate_h3_config(config)


def test_stage1_eval_config_requires_a_stage1_infer_base(tmp_path: Path) -> None:
    trained = _config(STAGE1)
    common = dict(
        task=EvalTask("ar", 100, "correct"),
        checkpoint=tmp_path / "checkpoint_model_000100",
        eval_data="/data/heldout",
        weight_source="live",
        output_dir=tmp_path / "out",
        loop="ar",
    )
    validate_h3_config(eval_config(_config(STAGE1_INFER), copy.deepcopy(trained), **common))
    with pytest.raises(SystemExit, match="stage0p5"):
        eval_config(_config(MIXED_INFER), trained, **common)


def _proxy_checkpoint(path: Path, *, stage: str, metadata: dict, value: float) -> str:
    from solarwm.checkpoint import CheckpointContract, CheckpointTransaction

    route = {
        "stage0p5": ("bidirectional", "flow_matching", "data_ward_velocity"),
        "stage1": ("teacher_forcing", "anyflow_forward_map", "v1_5"),
    }[stage]
    key = "base_model.model.block.attn.to_q.lora_B.weight"
    with CheckpointTransaction(path) as transaction:
        torch.save(
            {"metadata": metadata, "state": {key: torch.full((2, 2), value, dtype=torch.bfloat16)}},
            transaction.path / "adapter.pt",
        )
        transaction.commit(
            step=300,
            contract=CheckpointContract(
                family="minimax_h3",
                stage=stage,
                causal_mode=route[0],
                objective=route[1],
                objective_variant=route[2],
                camera_translation_transform="none",
                parameterization="peft-lora-r128-alpha128",
                sp_size=1,
                data_generation="h3.ref2va-proxy.124f.v1",
            ),
            required_components=("adapter.pt",),
            metadata={},
        )
    return key


def test_stage0p5_proxy_lora_starts_a_stage1_model_with_a_larger_base(tmp_path: Path) -> None:
    from solarwm.backends.minimax_h3.proxy_weights import load_proxy_checkpoint
    from solarwm.errors import BackendContractError

    def metadata(parameters: int) -> dict:
        return {
            "rank": 128,
            "base_identity": {"profile": "h3", "parameter_inventory": {"parameters": parameters}},
        }

    key = _proxy_checkpoint(tmp_path / "bd", stage="stage0p5", metadata=metadata(10), value=2.5)
    parameter = torch.nn.Parameter(torch.zeros(2, 2, dtype=torch.bfloat16))

    class FakeLoRA:
        def __init__(self, described: dict) -> None:
            self.parameter_by_key = {key: parameter}
            self.described = described

        def metadata(self) -> dict:
            return self.described

    stage1_lora = FakeLoRA(metadata(12))
    common = dict(weight_source="live", torch=torch)
    with pytest.raises(BackendContractError, match="metadata differs"):
        load_proxy_checkpoint(str(tmp_path / "bd"), stage1_lora, **common)
    load_proxy_checkpoint(str(tmp_path / "bd"), stage1_lora, model_stage="stage1", **common)
    assert torch.equal(parameter, torch.full((2, 2), 2.5, dtype=torch.bfloat16))
    other_rank = FakeLoRA({**metadata(12), "rank": 64})
    with pytest.raises(BackendContractError, match="metadata differs"):
        load_proxy_checkpoint(str(tmp_path / "bd"), other_rank, model_stage="stage1", **common)
    _proxy_checkpoint(tmp_path / "ar", stage="stage1", metadata=metadata(12), value=1.0)
    with pytest.raises(BackendContractError, match="stage0p5 LoRA"):
        load_proxy_checkpoint(str(tmp_path / "ar"), stage1_lora, **common)


def test_loss_summary_reads_the_anyflow_objective() -> None:
    events = [
        {"event": "optimizer_step", "step": step, "losses": {"anyflow_forward_map": 1.0}}
        for step in range(1, 11)
    ]
    assert loss_summary(events)["loss_ema"] == pytest.approx(1.0)
