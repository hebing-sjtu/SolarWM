"""Config-driven proxy geometry and separate depth / semantic proxy references."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from solarwm.backends.minimax_h3.config import validate_h3_config
from solarwm.backends.minimax_h3.geometry import validate_proxy_stage0p5_geometry
from solarwm.backends.minimax_h3.proxy_artifacts import (
    H3ProxyArtifactBatch,
    H3ProxyPtStream,
    h3_proxy_encoder_contract,
)
from solarwm.backends.minimax_h3.ref2va_layout import (
    _temporal_span,
    build_ref2va_proxy_layout,
)
from solarwm.errors import ConfigurationError, DataContractError
from solarwm.runtime import Topology

ROOT = Path(__file__).resolve().parents[3]
EXAMPLES = ROOT / "configs/examples/minimax_h3"
OMNI = EXAMPLES / "stage0p5-124f-ref2va-omni-704p-sp2.yaml"
OMNI_INFER = EXAMPLES / "infer-stage0p5-124f-ref2va-omni-704p-sp8.yaml"
LEGACY = EXAMPLES / "stage0p5-124f-ref2va-proxy-sp1.yaml"


def _config(path: Path, data_path: Path | None = None) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data_path is not None:
        config["data"]["data_path"] = str(data_path)
    return config


def test_omni_examples_validate_at_704p_with_sp2_training_and_sp8_inference() -> None:
    contract = validate_h3_config(_config(OMNI))
    assert contract.sequence_parallel_size == 2
    assert (contract.pixel_frames, contract.encoded_latents) == (124, 37)
    assert validate_h3_config(_config(OMNI_INFER)).sequence_parallel_size == 8


def test_proxy_training_accepts_sp_sizes_that_divide_the_world() -> None:
    for sp, batch in ((1, 16), (4, 4), (8, 2)):
        config = _config(OMNI)
        config["distributed"]["sequence_parallel_size"] = sp
        config["train"]["global_batch_size"] = batch
        validate_h3_config(config)
    config = _config(OMNI)
    config["distributed"]["sequence_parallel_size"] = 3
    with pytest.raises(ConfigurationError, match="sequence_parallel_size in"):
        validate_h3_config(config)
    config = _config(OMNI)
    config["train"]["global_batch_size"] = 16
    with pytest.raises(ConfigurationError, match="global batch mismatch"):
        validate_h3_config(config)


def test_the_model_canvas_has_to_follow_the_data_canvas() -> None:
    config = _config(OMNI)
    config["model"]["rows_per_latent"] = 1008
    with pytest.raises(ConfigurationError, match=r"model\.rows_per_latent must be 880"):
        validate_h3_config(config)
    config = _config(OMNI)
    config["data"]["latent_height"] = 48
    with pytest.raises(ConfigurationError, match="VisualVAE compression"):
        validate_h3_config(config)
    config = _config(OMNI)
    config["data"]["height"], config["data"]["latent_height"] = 720, 45
    config["model"]["latent_height"] = 45
    with pytest.raises(ConfigurationError, match="multiples of 32"):
        validate_h3_config(config)


def test_a_chat_role_has_to_describe_the_cached_references() -> None:
    config = _config(OMNI)
    config["data"]["cwm_system"] = "w0"
    with pytest.raises(ConfigurationError, match="describes one proxy video"):
        validate_h3_config(config)
    config = _config(OMNI)
    config["data"]["proxy_references"] = ["semantic", "depth"]
    with pytest.raises(ConfigurationError, match="describes the video references"):
        validate_h3_config(config)
    config = _config(OMNI)
    config["data"]["proxy_references"] = ["depth", "depth"]
    with pytest.raises(ConfigurationError, match="distinct"):
        validate_h3_config(config)
    config = _config(LEGACY)
    config["data"]["cwm_system"] = "w0_depth_semantic"
    with pytest.raises(ConfigurationError, match="describes the video references"):
        validate_h3_config(config)


def test_704p_geometry_derives_its_rows() -> None:
    geometry = validate_proxy_stage0p5_geometry(
        pixel_frames=124,
        encoded_latents=37,
        height=704,
        width=1280,
        latent_channels=24,
        latent_height=44,
        latent_width=80,
        proxy_latent_height=44,
        proxy_latent_width=80,
    )
    assert geometry.rows_per_latent == 880
    assert geometry.target_rows == 37 * 880
    assert geometry.proxy_rows == 37 * 880
    with pytest.raises(ValueError, match="no larger than the target"):
        validate_proxy_stage0p5_geometry(
            pixel_frames=124,
            encoded_latents=37,
            height=704,
            width=1280,
            latent_channels=24,
            latent_height=44,
            latent_width=80,
            proxy_latent_height=48,
            proxy_latent_width=80,
        )


def test_two_proxy_videos_follow_the_anchor_on_one_sequential_clock() -> None:
    tags = np.asarray([1, 0, 0, 1], dtype=np.int64)
    layout = build_ref2va_proxy_layout(
        tags,
        anchor_height=128,
        anchor_width=232,
        proxy_frames=37,
        proxy_height=44,
        proxy_width=80,
        target_frames=37,
        target_height=44,
        target_width=80,
        num_proxy_references=2,
    )
    anchor_rows = 64 * 116
    proxy_rows = 37 * 880
    assert layout.num_condition_video_rows == anchor_rows + 2 * proxy_rows
    assert layout.rows_per_video_frame == 880
    assert layout.sequence_length == 4 + anchor_rows + 2 * proxy_rows + 414 + 37 * 880
    first = 4 + anchor_rows
    second = first + proxy_rows
    span = _temporal_span(37)
    assert layout.position_ids[first, 0] == pytest.approx(4 + 1.0)
    assert layout.position_ids[second, 0] == pytest.approx(4 + 1.0 + span)
    target = int(layout.target_video_indices[0])
    assert layout.position_ids[target, 0] == pytest.approx(4 + 1.0 + 2 * span)
    single = build_ref2va_proxy_layout(
        tags,
        anchor_height=128,
        anchor_width=232,
        proxy_height=44,
        proxy_width=80,
        target_height=44,
        target_width=80,
    )
    assert single.sequence_length == layout.sequence_length - proxy_rows


def test_the_legacy_encoder_contract_is_unchanged() -> None:
    contract = h3_proxy_encoder_contract(
        qwen_video_fps=2.0,
        cwm_system="w0",
        align_proxy_reference_time=False,
        anchor_short_edge=2048,
    ).as_dict()
    shapes = {tensor["name"]: tuple(tensor["shape"]) for tensor in contract["tensors"]}
    assert shapes["target_latents"] == (24, 37, 48, 84)
    assert shapes["proxy_latents"] == (24, 37, 12, 21)
    assert contract["extras"]["reference_order"] == ["picture_anchor", "video_proxy"]
    assert "proxy_references" not in contract["extras"]


def _write_omni_sample(path: Path, *, references: list[str] | None = None) -> None:
    torch = pytest.importorskip("torch")
    torch.save(
        {
            "vae_latent": torch.zeros(24, 37, 44, 80),
            "proxy_latents": torch.zeros(2, 24, 37, 44, 80),
            "anchor_latent": torch.zeros(24, 1, 128, 232),
            "text_embedding": torch.zeros(4, 5120),
            "text_token_tags": torch.tensor([1, 0, 0, 1], dtype=torch.int64),
            "info": {
                "cwm_system": "w0_depth_semantic",
                "qwen_video_fps": 2.0,
                "proxy_references": references or ["depth", "semantic"],
                "fit": "center-crop",
            },
        },
        path,
    )


def test_the_reader_loads_an_omni_cache_and_refuses_other_references(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    _write_omni_sample(tmp_path / "a.pt")
    stream = H3ProxyPtStream(_config(OMNI, tmp_path), Topology(1, 0, 1, 0))
    batch = stream.next()
    assert batch.target_latents.shape == (24, 37, 44, 80)
    assert batch.proxy_latents.shape == (2, 24, 37, 44, 80)
    assert batch.num_given_latent_frames == 1
    assert stream.references == ("depth", "semantic")
    extras = stream.encoder_profile["extras"]
    assert extras["reference_order"] == ["picture_anchor", "video_depth", "video_semantic"]
    assert extras["proxy_references"] == ["depth", "semantic"]

    legacy_dir = tmp_path / "legacy"
    legacy_dir.mkdir()
    _write_omni_sample(legacy_dir / "a.pt")
    with pytest.raises(DataContractError, match="misses"):
        H3ProxyPtStream(_config(LEGACY, legacy_dir), Topology(1, 0, 1, 0)).next()

    other = tmp_path / "other"
    other.mkdir()
    _write_omni_sample(other / "a.pt", references=["semantic", "depth"])
    with pytest.raises(DataContractError, match="proxy_references"):
        H3ProxyPtStream(_config(OMNI, other), Topology(1, 0, 1, 0)).next()


def test_static_ablation_freezes_each_proxy_video_on_its_own_first_block() -> None:
    torch = pytest.importorskip("torch")

    from solarwm.backends.minimax_h3.proxy_ablation import apply_proxy_ablation

    # text | picture | text | depth blocks x2 | text | semantic blocks x2 | text
    tags = torch.tensor([1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1], dtype=torch.int64)
    proxies = torch.arange(2 * 3, dtype=torch.float32).reshape(2, 1, 3, 1, 1)
    batch = H3ProxyArtifactBatch(
        sample_id="omni",
        start_frame=0,
        plan_fingerprint="fingerprint",
        target_latents=torch.zeros(1),
        proxy_latents=proxies,
        anchor_latents=torch.zeros(1),
        prompt_embeds=torch.arange(15, dtype=torch.float32).reshape(15, 1),
        text_token_tags=tags,
        cwm_system="w0_depth_semantic",
        num_given_latent_frames=1,
    )
    static = apply_proxy_ablation(batch, mode="static")
    assert torch.equal(static.proxy_latents[0, 0, :, 0, 0], torch.tensor([0.0, 0.0, 0.0]))
    assert torch.equal(static.proxy_latents[1, 0, :, 0, 0], torch.tensor([3.0, 3.0, 3.0]))
    assert torch.equal(static.prompt_embeds[6:8, 0], static.prompt_embeds[3:5, 0])
    assert torch.equal(static.prompt_embeds[12:14, 0], torch.tensor([9.0, 10.0]))
    text = [0, 2, 5, 8, 11, 14]
    assert torch.equal(static.prompt_embeds[text], batch.prompt_embeds[text])
    assert torch.equal(static.prompt_embeds[1:2], batch.prompt_embeds[1:2])


def test_condition_rows_keep_reference_order() -> None:
    torch = pytest.importorskip("torch")

    from solarwm.backends.minimax_h3.layout import patchify_video
    from solarwm.backends.minimax_h3.stage0p5_ref2va import _condition_rows, _proxy_videos

    anchor = torch.zeros(1, 24, 1, 2, 2)
    proxies = torch.stack((torch.ones(24, 3, 2, 2), torch.full((24, 3, 2, 2), 2.0)))
    rows = _condition_rows(anchor, proxies)
    per_video = patchify_video(proxies[:1]).shape[1]
    assert rows.shape[1] == patchify_video(anchor).shape[1] + 2 * per_video
    assert torch.all(rows[:, 1 : 1 + per_video] == 1.0)
    assert torch.all(rows[:, 1 + per_video :] == 2.0)
    assert _proxy_videos(torch.zeros(24, 3, 2, 2)).shape == (1, 24, 3, 2, 2)
