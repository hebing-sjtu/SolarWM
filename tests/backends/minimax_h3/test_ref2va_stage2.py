from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from solarwm.backends.minimax_h3.config import validate_h3_config
from solarwm.backends.minimax_h3.layout import NOISY_VIDEO_ROLE
from solarwm.backends.minimax_h3.mask import build_stage1_window_mask
from solarwm.backends.minimax_h3.ref2va_layout import build_ref2va_stage1_layout
from solarwm.backends.minimax_h3.stage2_ref2va import (
    H3Ref2VAKVCache,
    H3Ref2VARolloutAttention,
    prefill_spans,
)
from solarwm.backends.minimax_h3.torch_mask import build_ref2va_stage1_block_mask
from solarwm.errors import ConfigurationError

ROOT = Path(__file__).resolve().parents[3]
EXAMPLES = ROOT / "configs/examples/minimax_h3"
STAGE2 = EXAMPLES / "stage2-124f-ref2va-omni-mixed-low-704p-sp2.yaml"
STAGE2_INFER = EXAMPLES / "infer-stage2-124f-ref2va-omni-mixed-low-704p-sp8.yaml"
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
        num_latent_frames=15,
        target_height=4,
        target_width=4,
        num_audio_latents=3,
        window_chunks=2,
    )
    arguments.update(overrides)
    return build_ref2va_stage1_layout(TAGS, **arguments)


def test_windowed_clean_history_drops_only_clean_keys_past_the_window() -> None:
    layout = _small()
    full = build_stage1_window_mask(layout)
    windowed = build_stage1_window_mask(layout, windowed_clean=True)
    roles, chunks = layout.row_roles, layout.target_video_chunk_ids
    clean = layout.clean_video_indices
    dropped = full & ~windowed
    assert not (windowed & ~full).any()
    queries, keys = np.nonzero(dropped)
    assert set(queries.tolist()) <= set(clean.tolist())
    assert set(keys.tolist()) <= set(clean.tolist())
    assert np.all(chunks[queries] - chunks[keys] >= 2)
    noisy = np.flatnonzero(roles == NOISY_VIDEO_ROLE)
    np.testing.assert_array_equal(full[noisy], windowed[noisy])


@pytest.mark.parametrize("arguments", [dict(), dict(window_chunks=6), dict(num_proxy_references=2)])
def test_windowed_clean_block_mask_matches_dense_visibility(arguments: dict) -> None:
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
    layout = source.to("cpu")
    mask = build_ref2va_stage1_block_mask(layout, device="cpu", windowed_clean=True)
    rows = torch.arange(layout.sequence_length)
    actual = mask.mask_mod(torch.tensor(0), torch.tensor(0), rows[:, None], rows[None, :])
    expected = torch.from_numpy(build_stage1_window_mask(source, windowed_clean=True))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    blocks = -(-layout.sequence_length // 128)
    padded = torch.zeros((blocks * 128, blocks * 128), dtype=torch.bool)
    padded[: layout.sequence_length, : layout.sequence_length] = expected
    any_tile = padded.reshape(blocks, 128, blocks, 128).any(dim=(1, 3))
    listed = torch.zeros((blocks, blocks), dtype=torch.bool)
    for counts, indices in (
        (mask.kv_num_blocks, mask.kv_indices),
        (mask.full_kv_num_blocks, mask.full_kv_indices),
    ):
        for query in range(blocks):
            listed[query, indices[0, 0, query, : counts[0, 0, query]].long()] = True
    assert torch.equal(listed, any_tile)


def _sdpa(query, key, value):
    out = torch.nn.functional.scaled_dot_product_attention(
        query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)
    )
    return out.transpose(1, 2)


@pytest.mark.parametrize("arguments", [dict(), dict(window_chunks=6), dict(num_proxy_references=2)])
def test_cached_rollout_attention_matches_the_windowed_teacher_forced_document(
    arguments: dict,
) -> None:
    source = _small(**arguments)
    layout = source.to("cpu")
    allowed = torch.from_numpy(build_stage1_window_mask(source, windowed_clean=True))
    generator = torch.Generator().manual_seed(0)
    q, k, v = (
        torch.randn((1, layout.sequence_length, 2, 8), generator=generator, dtype=torch.float64)
        for _ in range(3)
    )
    scores = torch.einsum("bqhd,bkhd->bhqk", q, k) / 8**0.5
    reference = torch.einsum(
        "bhqk,bkhd->bqhd", scores.masked_fill(~allowed, float("-inf")).softmax(-1), v
    )

    cache = H3Ref2VAKVCache(window_chunks=int(source.window_chunks))
    condition, proxy, audio = prefill_spans(layout)
    prefix = int(layout.clean_video_indices[0])
    prefill = H3Ref2VARolloutAttention(
        "prefill", cache, condition_rows=condition, proxy_spans=proxy, audio_span=audio
    )
    torch.testing.assert_close(
        prefill.attend(q[:, :prefix], k[:, :prefix], v[:, :prefix], layer=0, attend=_sdpa),
        reference[:, :prefix],
    )
    rows = int(layout.rows_per_video_frame) * 5
    chunks = source.num_clean_video_rows // rows
    for chunk in range(chunks):
        for mode, indices in (
            ("step", layout.noisy_video_indices),
            ("commit", layout.clean_video_indices),
        ):
            index = torch.cat(
                (torch.zeros(1, dtype=torch.long), indices[chunk * rows : (chunk + 1) * rows])
            )
            control = H3Ref2VARolloutAttention(mode, cache, chunk=chunk)
            with torch.no_grad():
                out = control.attend(q[:, index], k[:, index], v[:, index], layer=0, attend=_sdpa)
            torch.testing.assert_close(out[:, 1:], reference[:, index[1:]])
    assert len(cache.clean[0]) == min(int(source.window_chunks) - 1, chunks)


def test_kv_cache_rejects_out_of_order_commits() -> None:
    cache = H3Ref2VAKVCache(window_chunks=2)
    cache.common[0] = (torch.zeros(1, 1, 1, 1), torch.zeros(1, 1, 1, 1))
    cache.proxy[0] = [(torch.zeros(1, 1, 1, 1), torch.zeros(1, 1, 1, 1))] * 3
    with torch.no_grad():
        with pytest.raises(RuntimeError, match="history"):
            cache.commit(0, 1, torch.zeros(1, 1, 1, 1), torch.zeros(1, 1, 1, 1))
        cache.commit(0, 0, torch.zeros(1, 1, 1, 1), torch.zeros(1, 1, 1, 1))
        cache.commit(0, 1, torch.ones(1, 1, 1, 1), torch.ones(1, 1, 1, 1))
    assert [entry[0] for entry in cache.clean[0]] == [1]
    assert cache.keys(0, 2)[0].shape[1] == 1 + 2 + 1
    with pytest.raises(RuntimeError, match="detached"):
        cache.commit(0, 2, torch.zeros(1, 1, 1, 1), torch.zeros(1, 1, 1, 1))


def test_stage2_proxy_examples_validate() -> None:
    contract = validate_h3_config(_config(STAGE2))
    assert (contract.stage, contract.encoded_latents, contract.sequence_parallel_size) == (
        "stage2",
        37,
        2,
    )
    assert validate_h3_config(_config(STAGE2_INFER)).sequence_parallel_size == 8
    for path, edit, match in (
        (STAGE2, lambda c: c["data"].update(train_target_latents=35), "train_target_latents"),
        (
            STAGE2,
            lambda c: c["train"].update(tail_camera_policy="repeat_last_latent"),
            "tail_camera_policy",
        ),
        (
            STAGE2,
            lambda c: c["checkpoint"]["initialization"]["student"].update(stage="stage0p5"),
            "stage",
        ),
        (STAGE2, lambda c: c["checkpoint"]["initialization"].pop("critic"), "critic"),
        (STAGE2, lambda c: c["train"]["optimizer"].update(betas=[0.9, 0.999]), "betas"),
        (STAGE2, lambda c: c["checkpoint"]["ema"].update(start_step=0), "start_step"),
        (
            STAGE2_INFER,
            lambda c: c["validation"].update(num_inference_steps=30),
            "num_inference_steps",
        ),
    ):
        config = _config(path)
        edit(config)
        with pytest.raises(ConfigurationError, match=match):
            validate_h3_config(config)
    config = _config(STAGE2)
    config["train"]["critic_optimizer"]["learning_rate"] = 1.0e-6
    validate_h3_config(config)


def test_stage2_proxy_checkpoint_route(tmp_path: Path) -> None:
    from solarwm.backends.minimax_h3.proxy_weights import load_proxy_checkpoint
    from solarwm.checkpoint import CheckpointContract, CheckpointTransaction
    from solarwm.errors import BackendContractError

    key = "base_model.model.block.attn.to_q.lora_B.weight"
    routes = {
        "stage1": ("teacher_forcing", "anyflow_forward_map", "v1_5"),
        "stage2": ("self_gradient_forcing", "flow_matching", "data_ward_velocity"),
    }

    def metadata(parameters: int) -> dict:
        return {
            "rank": 128,
            "base_identity": {"profile": "h3", "parameter_inventory": {"parameters": parameters}},
        }

    def save(path: Path, stage: str, parameters: int, value: float) -> None:
        with CheckpointTransaction(path) as transaction:
            torch.save(
                {
                    "metadata": metadata(parameters),
                    "state": {key: torch.full((2, 2), value, dtype=torch.bfloat16)},
                },
                transaction.path / "adapter.pt",
            )
            transaction.commit(
                step=50,
                contract=CheckpointContract(
                    family="minimax_h3",
                    stage=stage,
                    causal_mode=routes[stage][0],
                    objective=routes[stage][1],
                    objective_variant=routes[stage][2],
                    camera_translation_transform="none",
                    parameterization="peft-lora-r128-alpha128",
                    sp_size=2,
                    data_generation="h3.ref2va-proxy.124f.v1",
                ),
                required_components=("adapter.pt",),
                metadata={},
            )

    parameter = torch.nn.Parameter(torch.zeros(2, 2, dtype=torch.bfloat16))

    class FakeLoRA:
        def __init__(self) -> None:
            self.parameter_by_key = {key: parameter}

        def metadata(self) -> dict:
            return metadata(10)

    common = dict(weight_source="live", torch=torch)
    save(tmp_path / "ar", "stage1", 12, 1.5)
    with pytest.raises(BackendContractError, match="metadata differs"):
        load_proxy_checkpoint(str(tmp_path / "ar"), FakeLoRA(), stage="stage1", **common)
    load_proxy_checkpoint(
        str(tmp_path / "ar"), FakeLoRA(), stage="stage1", model_stage="stage2", **common
    )
    assert torch.equal(parameter, torch.full((2, 2), 1.5, dtype=torch.bfloat16))
    save(tmp_path / "dmd", "stage2", 10, 3.0)
    load_proxy_checkpoint(str(tmp_path / "dmd"), FakeLoRA(), stage="stage2", **common)
    assert torch.equal(parameter, torch.full((2, 2), 3.0, dtype=torch.bfloat16))
