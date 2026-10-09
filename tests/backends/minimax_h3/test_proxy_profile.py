from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from solarwm.backends.minimax_h3.config import validate_h3_config
from solarwm.backends.minimax_h3.geometry import validate_proxy_stage0p5_geometry
from solarwm.backends.minimax_h3.layout import build_row_timesteps
from solarwm.backends.minimax_h3.lora import discover_h3_lora_targets
from solarwm.backends.minimax_h3.proxy_artifacts import (
    H3ProxyArtifactBatch,
    H3ProxyPtStream,
)
from solarwm.backends.minimax_h3.ref2va_layout import build_ref2va_proxy_layout
from solarwm.config.routes import validate_route
from solarwm.errors import ConfigurationError, DataContractError
from solarwm.runtime import Topology

ROOT = Path(__file__).resolve().parents[3]
EXAMPLE = ROOT / "configs/examples/minimax_h3/stage0p5-124f-ref2va-proxy-sp1.yaml"
INFER_EXAMPLE = ROOT / "configs/examples/minimax_h3/infer-stage0p5-124f-ref2va-proxy-sp8.yaml"


def _proxy_config(data_path: Path | None = None) -> dict[str, object]:
    config = yaml.safe_load(EXAMPLE.read_text(encoding="utf-8"))
    if data_path is not None:
        config["data"]["data_path"] = str(data_path)
    return config


def test_proxy_example_resolves_to_isolated_sp1_contract() -> None:
    config = _proxy_config()
    assert validate_route(config).family == "minimax_h3"
    contract = validate_h3_config(config)
    assert (contract.pixel_frames, contract.encoded_latents) == (124, 37)
    assert contract.sequence_parallel_size == 1
    assert contract.adapter_rank == 128
    assert contract.camera_translation_transform == "none"


def test_proxy_training_accepts_weight_only_initialization() -> None:
    config = _proxy_config()
    config["checkpoint"]["initialization"] = {
        "student": {
            "path": "/data/run/checkpoint_model_003000",
            "weight_source": "live",
            "stage": "stage0p5",
        }
    }
    validate_h3_config(config)

    config["checkpoint"]["resume_from"] = "/data/run/checkpoint_model_003000"
    validate_h3_config(config)

    config["checkpoint"]["resume_from"] = None
    config["checkpoint"]["initialization"]["student"]["path"] = "relative/checkpoint"
    with pytest.raises(ConfigurationError, match="must be absolute"):
        validate_h3_config(config)

    config["checkpoint"]["initialization"]["student"]["path"] = "/data/run/checkpoint"
    config["checkpoint"]["initialization"]["student"]["weight_source"] = "unknown"
    with pytest.raises(ConfigurationError, match="must be live or ema"):
        validate_h3_config(config)


def test_proxy_inference_example_uses_one_sp8_worker_and_allows_base_baseline() -> None:
    config = yaml.safe_load(INFER_EXAMPLE.read_text(encoding="utf-8"))
    contract = validate_h3_config(config)
    assert contract.action == "infer"
    assert contract.sequence_parallel_size == 8
    assert (contract.pixel_frames, contract.encoded_latents) == (124, 37)

    config["checkpoint"]["resume_from"] = None
    validate_h3_config(config)

    config["validation"]["proxy_ablation"] = "unknown"
    with pytest.raises(ConfigurationError, match="proxy_ablation"):
        validate_h3_config(config)
    config["validation"]["proxy_ablation"] = "correct"

    config["distributed"]["sequence_parallel_size"] = 1
    with pytest.raises(ConfigurationError, match="SP8"):
        validate_h3_config(config)


def test_proxy_tracking_is_scalar_only_and_evaluation_is_manual() -> None:
    config = _proxy_config()
    config["runtime"]["tracking"]["log_media"] = True
    with pytest.raises(ConfigurationError, match="log_media"):
        validate_h3_config(config)

    config = _proxy_config()
    config["runtime"]["tracking"]["loss_ema_beta"] = 1.0
    with pytest.raises(ConfigurationError, match="loss_ema_beta"):
        validate_h3_config(config)

    config = _proxy_config()
    config["validation"]["manual_steps"] = [96, 48]
    with pytest.raises(ConfigurationError, match="sorted and unique"):
        validate_h3_config(config)


def test_proxy_geometry_is_frozen() -> None:
    geometry = validate_proxy_stage0p5_geometry(
        pixel_frames=124,
        encoded_latents=37,
        height=768,
        width=1344,
        latent_channels=24,
        latent_height=48,
        latent_width=84,
        proxy_latent_height=12,
        proxy_latent_width=21,
    )
    assert geometry.target_rows == 37 * 1008
    assert geometry.proxy_rows == 37 * 66
    assert geometry.audio_latents == 207


def test_ref2va_layout_pads_proxy_and_marks_fixed_target_rows() -> None:
    tags = np.asarray([0, 1, 1], dtype=np.int64)
    layout = build_ref2va_proxy_layout(
        tags,
        anchor_height=128,
        anchor_width=224,
        proxy_frames=37,
        proxy_height=12,
        proxy_width=22,
        target_frames=37,
        target_height=48,
        target_width=84,
        num_audio_latents=207,
        align_proxy_reference_time=False,
    )
    anchor_rows = (128 // 2) * (224 // 2)
    proxy_rows = 37 * 66
    target_rows = 37 * 1008
    assert layout.num_condition_video_rows == anchor_rows + proxy_rows
    assert layout.num_noisy_video_rows == target_rows
    assert layout.video_indices.size == anchor_rows + proxy_rows + target_rows
    assert layout.audio_indices.size == 414
    assert layout.sequence_length == 3 + anchor_rows + proxy_rows + 414 + target_rows

    times, inverse = build_row_timesteps(
        layout,
        0.4,
        0.7,
        condition_video_timestep=0.999,
        num_fixed_video_rows=1008,
    )
    expanded = times[inverse]
    assert np.all(expanded[layout.condition_video_indices] == np.float32(0.999))
    assert np.all(expanded[layout.noisy_video_indices[:1008]] == np.float32(0.999))
    assert np.all(expanded[layout.noisy_video_indices[1008:]] == np.float32(0.4))


def _write_sample(path: Path, *, role: str = "w0") -> None:
    torch = pytest.importorskip("torch")
    torch.save(
        {
            "vae_latent": torch.zeros(24, 37, 48, 84),
            "proxy_latent": torch.zeros(24, 37, 12, 21),
            "anchor_latent": torch.zeros(24, 1, 127, 223),
            "text_embedding": torch.zeros(4, 5120),
            "text_token_tags": torch.tensor([0, 0, 1, 1], dtype=torch.int64),
            "info": {"cwm_system": role, "qwen_video_fps": 2.0},
        },
        path,
    )


def test_proxy_reader_preserves_resume_and_rejects_role_mismatch(
    tmp_path: Path,
) -> None:
    pytest.importorskip("torch")
    _write_sample(tmp_path / "a.pt")
    stream = H3ProxyPtStream(_proxy_config(tmp_path), Topology(1, 0, 1, 0))
    batch = stream.next()
    assert batch.target_latents.shape == (24, 37, 48, 84)
    assert batch.num_given_latent_frames == 1
    state = stream.state_dict()

    restored = H3ProxyPtStream(_proxy_config(tmp_path), Topology(1, 0, 1, 0))
    restored.load_state_dict(state)
    assert restored.state_dict()["cursor"] == 1

    mismatch = tmp_path / "mismatch"
    mismatch.mkdir()
    _write_sample(mismatch / "a.pt", role="wn")
    bad = H3ProxyPtStream(_proxy_config(mismatch), Topology(1, 0, 1, 0))
    with pytest.raises(DataContractError, match="cwm_system"):
        bad.next()


def test_proxy_lora_discovers_only_main_attention_qkvo() -> None:
    torch = pytest.importorskip("torch")

    class Attention(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fused_projections = False
            self.to_q = torch.nn.Linear(2, 2)
            self.to_k = torch.nn.Linear(2, 2)
            self.to_v = torch.nn.Linear(2, 2)
            self.to_out = torch.nn.Sequential(torch.nn.Linear(2, 2))

    class Block(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attn = Attention()

    class MainOnly(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.transformer_blocks = torch.nn.ModuleList(Block() for _ in range(50))

    targets = discover_h3_lora_targets(MainOnly(), target="main_attention_qkvo")
    assert len(targets) == 200
    assert all(".attn." in name for name in targets)
    assert not any("ff." in name or "refiner" in name for name in targets)


def test_proxy_ablation_changes_full_rate_and_qwen_proxy_only() -> None:
    torch = pytest.importorskip("torch")

    from solarwm.backends.minimax_h3.proxy_ablation import apply_proxy_ablation

    # FastVideo's convention: Qwen vision placeholders are VIDEO_TAG=0, text is TEXT_TAG=1.
    tags = torch.tensor([1, 0, 0, 1, 0, 0, 1, 0, 0, 1], dtype=torch.int64)

    def batch(sample_id: str, offset: float) -> H3ProxyArtifactBatch:
        return H3ProxyArtifactBatch(
            sample_id=sample_id,
            start_frame=0,
            plan_fingerprint=f"fingerprint-{sample_id}",
            target_latents=torch.zeros(1),
            proxy_latents=(torch.arange(3, dtype=torch.float32).reshape(1, 3, 1, 1) + offset),
            anchor_latents=torch.zeros(1),
            prompt_embeds=(torch.arange(20, dtype=torch.float32).reshape(10, 2) + offset),
            text_token_tags=tags,
            cwm_system="w0",
            num_given_latent_frames=1,
        )

    primary = batch("primary", 0.0)
    donor = batch("donor", 100.0)
    shuffled = apply_proxy_ablation(primary, mode="shuffled", donor=donor)
    assert torch.equal(shuffled.proxy_latents, donor.proxy_latents)
    assert torch.equal(shuffled.prompt_embeds[1:3], primary.prompt_embeds[1:3])
    assert torch.equal(shuffled.prompt_embeds[4:6], donor.prompt_embeds[4:6])
    assert torch.equal(shuffled.prompt_embeds[7:9], donor.prompt_embeds[7:9])
    assert torch.equal(shuffled.prompt_embeds[[0, 3, 6, 9]], primary.prompt_embeds[[0, 3, 6, 9]])

    static = apply_proxy_ablation(primary, mode="static")
    assert torch.equal(
        static.proxy_latents,
        primary.proxy_latents[:, :1].expand_as(primary.proxy_latents),
    )
    assert torch.equal(static.prompt_embeds[1:3], primary.prompt_embeds[1:3])
    assert torch.equal(static.prompt_embeds[7:9], primary.prompt_embeds[4:6])


def test_proxy_ablation_fits_different_qwen_video_token_counts() -> None:
    torch = pytest.importorskip("torch")

    from solarwm.backends.minimax_h3.proxy_ablation import apply_proxy_ablation

    def batch(
        sample_id: str,
        tags: list[int],
        offset: float,
    ) -> H3ProxyArtifactBatch:
        prompt = torch.arange(len(tags), dtype=torch.float32).reshape(-1, 1) + offset
        return H3ProxyArtifactBatch(
            sample_id=sample_id,
            start_frame=0,
            plan_fingerprint=f"fingerprint-{sample_id}",
            target_latents=torch.zeros(1),
            proxy_latents=torch.full((1, 3, 1, 1), offset),
            anchor_latents=torch.zeros(1),
            prompt_embeds=prompt,
            text_token_tags=torch.tensor(tags, dtype=torch.int64),
            cwm_system="w0",
            num_given_latent_frames=1,
        )

    primary = batch(
        "primary",
        [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1],
        0.0,
    )
    donor = batch(
        "donor",
        [1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1],
        100.0,
    )

    shuffled = apply_proxy_ablation(primary, mode="shuffled", donor=donor)
    assert torch.equal(
        shuffled.prompt_embeds[4:7, 0],
        torch.tensor([103.0, 104.0, 104.0]),
    )
    assert torch.equal(
        shuffled.prompt_embeds[8:10, 0],
        torch.tensor([106.0, 109.0]),
    )
    assert torch.equal(shuffled.prompt_embeds[1:3], primary.prompt_embeds[1:3])
    assert torch.equal(
        shuffled.prompt_embeds[[0, 3, 7, 10]],
        primary.prompt_embeds[[0, 3, 7, 10]],
    )

    static = apply_proxy_ablation(primary, mode="static")
    assert torch.equal(static.prompt_embeds[4:7], primary.prompt_embeds[4:7])
    assert torch.equal(
        static.prompt_embeds[8:10, 0],
        torch.tensor([4.0, 6.0]),
    )


def test_proxy_ema_loader_canonicalizes_fsdp_and_adapter_prefixes(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")

    from solarwm.backends.minimax_h3.proxy_weights import load_proxy_checkpoint
    from solarwm.checkpoint import CheckpointContract, CheckpointTransaction

    target = tmp_path / "checkpoint_model_000500"
    canonical_key = "base_model.model.block.attn.to_q.lora_B.weight"
    saved_key = "base_model.model._fsdp_wrapped_module.block.attn.to_q.lora_B.default.weight"
    with CheckpointTransaction(target) as transaction:
        torch.save(
            {
                "schema": "solarwm.minimax-h3-ema.v1",
                "decay": 0.9999,
                "num_updates": 500,
                "trainable_only": True,
                "shadow": {saved_key: torch.full((2, 2), 1.25, dtype=torch.float32)},
            },
            transaction.path / "ema.pt",
        )
        transaction.commit(
            step=500,
            contract=CheckpointContract(
                family="minimax_h3",
                stage="stage0p5",
                causal_mode="bidirectional",
                objective="flow_matching",
                objective_variant="data_ward_velocity",
                camera_translation_transform="none",
                parameterization="peft-lora-r128-alpha128",
                sp_size=1,
                data_generation="h3.ref2va-proxy.124f.v1",
            ),
            required_components=("ema.pt",),
            metadata={},
        )

    parameter = torch.nn.Parameter(torch.zeros(2, 2, dtype=torch.bfloat16))

    class FakeLoRA:
        def __init__(self) -> None:
            self.parameter_by_key = {canonical_key: parameter}

    weights_id = load_proxy_checkpoint(
        str(target),
        FakeLoRA(),
        weight_source="ema",
        torch=torch,
    )
    assert weights_id.endswith(":ema:step=500")
    assert torch.equal(parameter, torch.full((2, 2), 1.25, dtype=torch.bfloat16))


def test_proxy_live_loader_supports_training_warm_start(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")

    from solarwm.backends.minimax_h3.proxy_weights import load_proxy_checkpoint
    from solarwm.checkpoint import CheckpointContract, CheckpointTransaction

    target = tmp_path / "checkpoint_model_003000"
    key = "base_model.model.block.attn.to_q.lora_B.weight"
    metadata = {"target_count": 1, "trainable_parameters": 4}
    with CheckpointTransaction(target) as transaction:
        torch.save(
            {
                "metadata": metadata,
                "state": {key: torch.full((2, 2), 2.5, dtype=torch.bfloat16)},
            },
            transaction.path / "adapter.pt",
        )
        transaction.commit(
            step=3000,
            contract=CheckpointContract(
                family="minimax_h3",
                stage="stage0p5",
                causal_mode="bidirectional",
                objective="flow_matching",
                objective_variant="data_ward_velocity",
                camera_translation_transform="none",
                parameterization="peft-lora-r128-alpha128",
                sp_size=1,
                data_generation="h3.ref2va-proxy.124f.v1",
            ),
            required_components=("adapter.pt",),
            metadata={},
        )

    parameter = torch.nn.Parameter(torch.zeros(2, 2, dtype=torch.bfloat16))

    class FakeLoRA:
        def __init__(self) -> None:
            self.parameter_by_key = {key: parameter}

        @staticmethod
        def metadata() -> dict[str, int]:
            return metadata

    weights_id = load_proxy_checkpoint(
        str(target),
        FakeLoRA(),
        weight_source="live",
        torch=torch,
    )
    assert weights_id.endswith(":live:step=3000")
    assert torch.equal(parameter, torch.full((2, 2), 2.5, dtype=torch.bfloat16))
