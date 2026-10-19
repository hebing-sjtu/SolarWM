"""An optional ``<Picture 2>`` OC identity sheet beside the Ref2VA anchor."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from solarwm.backends.minimax_h3.config import validate_h3_config
from solarwm.backends.minimax_h3.layout import CONDITION_ROLE, PROXY_VIDEO_ROLE
from solarwm.backends.minimax_h3.proxy_artifacts import H3ProxyArtifactBatch, H3ProxyPtStream
from solarwm.backends.minimax_h3.ref2va_layout import (
    _temporal_span,
    build_ref2va_proxy_layout,
    build_ref2va_stage1_layout,
)
from solarwm.errors import ConfigurationError, DataContractError
from solarwm.runtime import Topology

ROOT = Path(__file__).resolve().parents[3]
MIXED = ROOT / "configs/examples/minimax_h3/stage0p5-124f-ref2va-omni-mixed-low-704p-sp1.yaml"
TAGS = np.asarray([1, 0, 0, 1], dtype=np.int64)
GEOMETRY = dict(
    anchor_height=128,
    anchor_width=232,
    proxy_height=22,
    proxy_width=40,
    target_height=44,
    target_width=80,
)


def _config(data_path: Path | None = None, **data: object) -> dict:
    config = yaml.safe_load(MIXED.read_text(encoding="utf-8"))
    if data_path is not None:
        config["data"]["data_path"] = str(data_path)
    config["data"].update(data)
    return config


def test_the_oc_sheet_is_a_second_picture_between_the_anchor_and_the_proxy() -> None:
    plain = build_ref2va_proxy_layout(TAGS, **GEOMETRY)
    assert np.array_equal(
        plain.position_ids,
        build_ref2va_proxy_layout(TAGS, extra_image_sizes=(), **GEOMETRY).position_ids,
    )
    sheet = build_ref2va_proxy_layout(TAGS, extra_image_sizes=((40, 59),), **GEOMETRY)
    anchor_rows, sheet_rows, proxy_rows = 64 * 116, 20 * 30, 37 * 220
    assert sheet.num_condition_video_rows == anchor_rows + sheet_rows + proxy_rows
    assert sheet.sequence_length == plain.sequence_length + sheet_rows
    first_sheet = 4 + anchor_rows
    assert sheet.position_ids[4, 0] == pytest.approx(4.0)
    assert sheet.position_ids[first_sheet, 0] == pytest.approx(5.0)
    assert sheet.position_ids[first_sheet + sheet_rows, 0] == pytest.approx(6.0)
    target = int(sheet.target_video_indices[0])
    assert sheet.position_ids[target, 0] == pytest.approx(6.0 + _temporal_span(37))
    assert sheet.position_ids[target, 0] == pytest.approx(
        plain.position_ids[int(plain.target_video_indices[0]), 0] + 1.0
    )


def test_stage1_keeps_the_oc_rows_in_the_condition_prefix() -> None:
    full = build_ref2va_proxy_layout(
        TAGS, extra_image_sizes=((40, 60),), proxy_frames=37, target_frames=37, **GEOMETRY
    )
    window = build_ref2va_stage1_layout(
        TAGS,
        extra_image_sizes=((40, 60),),
        proxy_frames=37,
        num_latent_frames=40,
        chunks=(2, 8),
        **GEOMETRY,
    )
    prefix = 4 + 64 * 116 + 20 * 30
    np.testing.assert_array_equal(window.position_ids[:prefix], full.position_ids[:prefix])
    assert np.all(window.row_roles[:prefix] == CONDITION_ROLE)
    proxy = window.row_roles == PROXY_VIDEO_ROLE
    assert int(np.flatnonzero(proxy)[0]) == prefix
    np.testing.assert_array_equal(
        window.position_ids[proxy], full.position_ids[prefix + 10 * 220 : prefix + 37 * 220]
    )


def test_condition_rows_put_the_sheet_after_the_anchor() -> None:
    torch = pytest.importorskip("torch")

    from solarwm.backends.minimax_h3.layout import patchify_video
    from solarwm.backends.minimax_h3.stage0p5_ref2va import _condition_rows

    anchor = torch.zeros(1, 24, 1, 2, 2)
    sheet = torch.ones(1, 24, 1, 2, 4)
    proxies = torch.full((1, 24, 3, 2, 2), 2.0)
    rows = _condition_rows((anchor, sheet), proxies)
    assert rows.shape[1] == 1 + 2 + patchify_video(proxies).shape[1]
    assert torch.all(rows[:, :1] == 0.0)
    assert torch.all(rows[:, 1:3] == 1.0)
    assert torch.all(rows[:, 3:] == 2.0)
    assert torch.equal(_condition_rows(anchor, proxies), _condition_rows((anchor,), proxies))


def _write_sample(path: Path, *, sheet: bool = True) -> None:
    torch = pytest.importorskip("torch")
    payload = {
        "vae_latent": torch.zeros(24, 37, 44, 80),
        "proxy_latents": torch.zeros(1, 24, 37, 11, 20),
        "anchor_latent": torch.zeros(24, 1, 128, 232),
        "text_embedding": torch.ones(6, 5120),
        "text_token_tags": torch.tensor([1, 0, 1, 0, 0, 1], dtype=torch.int64),
        "info": {
            "cwm_system": "w0_omni",
            "qwen_video_fps": 2.0,
            "proxy_modality": "semantic",
            "proxy_references": ["semantic"],
            "proxy_variants": ["duv", "depth", "semantic", "style"],
            "fit": "center-crop",
            "code_resize": "nearest",
        },
    }
    if sheet:
        payload.update(
            {
                "oc_latent": torch.full((24, 1, 40, 60), 3.0),
                "text_embedding_no_oc": torch.zeros(4, 5120),
                "text_token_tags_no_oc": torch.tensor([1, 0, 0, 1], dtype=torch.int64),
            }
        )
    torch.save(payload, path)


def test_the_reader_presents_the_sheet_only_to_oc_runs(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    _write_sample(tmp_path / "a.pt")
    topology = Topology(1, 0, 1, 0)

    plain = H3ProxyPtStream(_config(tmp_path), topology).next()
    assert plain.oc_latents is None
    assert tuple(plain.prompt_embeds.shape) == (4, 5120)
    assert plain.text_token_tags.tolist() == [1, 0, 0, 1]
    assert len(plain.image_latents()) == 1

    stream = H3ProxyPtStream(_config(tmp_path, oc_reference=True), topology)
    batch = stream.next()
    assert tuple(batch.oc_latents.shape) == (24, 1, 40, 60)
    assert tuple(batch.prompt_embeds.shape) == (6, 5120)
    assert tuple(batch.prompt_embeds_no_oc.shape) == (4, 5120)
    assert len(batch.image_latents()) == 2
    assert stream.encoder_profile["extras"]["oc_reference"] is True
    assert stream.encoder_profile["extras"]["reference_order"] == [
        "picture_anchor",
        "picture_oc_optional",
        "video_proxy",
    ]

    color = tmp_path / "color"
    color.mkdir()
    _write_sample(color / "a.pt", sheet=False)
    unsheeted = H3ProxyPtStream(_config(color, oc_reference=True), topology).next()
    assert unsheeted.oc_latents is None
    assert tuple(unsheeted.prompt_embeds.shape) == (6, 5120)


def test_oc_dropout_is_a_deterministic_training_only_coin(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    for index in range(8):
        _write_sample(tmp_path / f"{index}.pt")
    config = _config(tmp_path, oc_reference=True, oc_drop_prob=0.5)
    topology = Topology(1, 0, 1, 0)

    def visits(stream: H3ProxyPtStream) -> list[bool]:
        return [stream.next().oc_latents is None for _ in range(32)]

    dropped = visits(H3ProxyPtStream(config, topology, oc_dropout=True))
    assert dropped == visits(H3ProxyPtStream(config, topology, oc_dropout=True))
    assert 4 <= sum(dropped) <= 28
    assert not any(visits(H3ProxyPtStream(config, topology)))
    batch = next(
        batch
        for batch in iter(H3ProxyPtStream(config, topology, oc_dropout=True).next, None)
        if batch.oc_latents is None
    )
    assert tuple(batch.prompt_embeds.shape) == (4, 5120)


def test_oc_config_is_validated() -> None:
    validate_h3_config(_config(oc_reference=True, oc_drop_prob=0.2))
    with pytest.raises(ConfigurationError, match=r"requires data\.oc_reference"):
        validate_h3_config(_config(oc_drop_prob=0.2))
    with pytest.raises(ConfigurationError, match=r"\[0, 1\)"):
        validate_h3_config(_config(oc_reference=True, oc_drop_prob=1.0))


def test_a_sheet_without_its_no_oc_rows_is_refused(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    _write_sample(tmp_path / "a.pt")
    payload = torch.load(tmp_path / "a.pt", weights_only=True)
    del payload["text_embedding_no_oc"]
    torch.save(payload, tmp_path / "a.pt")
    with pytest.raises(DataContractError, match="text_embedding_no_oc"):
        H3ProxyPtStream(_config(tmp_path, oc_reference=True), Topology(1, 0, 1, 0)).next()


def test_no_oc_ablation_restores_the_unsheeted_document() -> None:
    torch = pytest.importorskip("torch")

    from solarwm.backends.minimax_h3.proxy_ablation import apply_proxy_ablation

    # text | anchor | text | sheet | text | video block | text | video block | text
    tags = torch.tensor([1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.int64)
    batch = H3ProxyArtifactBatch(
        sample_id="oc",
        start_frame=0,
        plan_fingerprint="fingerprint",
        target_latents=torch.zeros(1),
        proxy_latents=torch.arange(3, dtype=torch.float32).reshape(1, 1, 3, 1, 1),
        anchor_latents=torch.zeros(24, 1, 2, 2),
        prompt_embeds=torch.arange(9, dtype=torch.float32).reshape(9, 1),
        text_token_tags=tags,
        cwm_system="w0_omni",
        num_given_latent_frames=1,
        oc_latents=torch.ones(24, 1, 2, 2),
        prompt_embeds_no_oc=torch.zeros(7, 1),
        text_token_tags_no_oc=torch.tensor([1, 0, 1, 0, 1, 0, 1], dtype=torch.int64),
    )
    removed = apply_proxy_ablation(batch, mode="no_oc")
    assert removed.oc_latents is None
    assert removed.text_token_tags.tolist() == [1, 0, 1, 0, 1, 0, 1]
    assert removed.plan_fingerprint != batch.plan_fingerprint
    assert apply_proxy_ablation(removed, mode="no_oc") is removed

    static = apply_proxy_ablation(batch, mode="static")
    assert torch.equal(static.prompt_embeds[3], batch.prompt_embeds[3])
    assert torch.equal(static.prompt_embeds[7], batch.prompt_embeds[5])
    assert torch.equal(static.prompt_embeds[5], batch.prompt_embeds[5])
