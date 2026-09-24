from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from solarwm.backends.minimax_h3.config import validate_h3_config
from solarwm.backends.minimax_h3.geometry import validate_proxy_stage0p5_geometry
from solarwm.backends.minimax_h3.layout import build_row_timesteps
from solarwm.backends.minimax_h3.lora import discover_h3_lora_targets
from solarwm.backends.minimax_h3.proxy_artifacts import H3ProxyPtStream
from solarwm.backends.minimax_h3.ref2va_layout import build_ref2va_proxy_layout
from solarwm.config.routes import validate_route
from solarwm.errors import DataContractError
from solarwm.runtime import Topology

ROOT = Path(__file__).resolve().parents[3]
EXAMPLE = ROOT / "configs/examples/minimax_h3/stage0p5-124f-ref2va-proxy-sp1.yaml"


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
