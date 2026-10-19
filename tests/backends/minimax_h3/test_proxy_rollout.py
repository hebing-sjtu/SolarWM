from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from solarwm.backends.minimax_h3 import stage1_ref2va
from solarwm.backends.minimax_h3.config import validate_h3_config
from solarwm.backends.minimax_h3.proxy_artifacts import H3ProxyArtifactBatch, batch_given_frames
from solarwm.backends.minimax_h3.proxy_rollout import (
    GIVEN_FRAMES,
    STRIDE_FRAMES,
    TAKE_SCHEMA,
    read_take,
    stitch_spans,
    take_frames,
    with_given_latents,
)
from solarwm.backends.minimax_h3.stage1_ref2va import H3Ref2VAStage1Core, given_in_chunk
from solarwm.backends.minimax_h3.stage2_ref2va import H3Ref2VASGFCore, H3Ref2VASGFInputs
from solarwm.errors import ConfigurationError, DataContractError

ROOT = Path(__file__).resolve().parents[3]
INFER = ROOT / "configs/examples/minimax_h3/infer-stage0p5-124f-ref2va-omni-mixed-low-704p-sp8.yaml"


def _batch(role: str, *, frames: int = 37) -> H3ProxyArtifactBatch:
    given = {"w0_omni": 1, "wn_omni": 10}[role]
    return H3ProxyArtifactBatch(
        sample_id="s",
        start_frame=0,
        plan_fingerprint="f",
        target_latents=torch.zeros(24, frames, 2, 2),
        proxy_latents=torch.zeros(1, 24, frames, 2, 2),
        anchor_latents=torch.zeros(24, 1, 2, 2),
        prompt_embeds=torch.zeros(3, 5120),
        text_token_tags=torch.zeros(3, dtype=torch.long),
        cwm_system=role,
        num_given_latent_frames=given,
        proxy_references=("depth",),
    )


def _take(tmp_path: Path, segments: int, **changes) -> Path:
    rows = []
    for index in range(segments):
        name = f"take-s{index:02d}"
        (tmp_path / f"{name}.pt").write_bytes(b"")
        role = "w0_omni" if index == 0 else "wn_omni"
        rows.append({"name": name, "start": index * STRIDE_FRAMES, "cwm_system": role})
    payload = {
        "schema": TAKE_SCHEMA,
        "name": "take",
        "fps": 24,
        "segment_frames": 124,
        "overlap_frames": 39,
        "given_frames": 34,
        "stride_frames": 85,
        "frames": take_frames(segments),
        "target": "placeholder",
        "segments": rows,
        **changes,
    }
    path = tmp_path / "take.json"
    path.write_text(json.dumps(payload))
    return path


def test_stitched_segments_cover_every_take_frame_once() -> None:
    for segments in (1, 2, 5, 9):
        covered = [
            index * STRIDE_FRAMES + frame
            for index, (first, stop) in enumerate(stitch_spans(segments))
            for frame in range(first, stop)
        ]
        assert covered == list(range(take_frames(segments)))
    # Later segments only add frames after the ones their given latents reproduce.
    assert stitch_spans(3) == ((0, STRIDE_FRAMES + GIVEN_FRAMES), (34, 119), (34, 124))
    assert take_frames(9) == 804  # 33.5 s at 24 fps


def test_take_manifest_matches_roles_starts_and_length(tmp_path: Path) -> None:
    take = read_take(_take(tmp_path, 3), base_role="w0_omni")
    assert [s.cwm_system for s in take.segments] == ["w0_omni", "wn_omni", "wn_omni"]
    assert [s.start for s in take.segments] == [0, 85, 170]
    assert take.frames == 294 and take.segments[2].path == tmp_path / "take-s02.pt"
    with pytest.raises(DataContractError, match="no continuation role"):
        read_take(_take(tmp_path, 3), base_role="w0_depth_semantic")
    with pytest.raises(DataContractError, match="stitch to 294"):
        read_take(_take(tmp_path, 3, frames=300), base_role="w0_omni")
    with pytest.raises(DataContractError, match="overlap_frames"):
        read_take(_take(tmp_path, 3, overlap_frames=40), base_role="w0_omni")
    path = _take(tmp_path, 2)
    payload = json.loads(path.read_text())
    payload["segments"][1]["cwm_system"] = "w0_omni"
    path.write_text(json.dumps(payload))
    with pytest.raises(DataContractError, match="must use CWM role 'wn_omni'"):
        read_take(path, base_role="w0_omni")


def test_continuation_documents_are_accepted_only_by_a_rollout() -> None:
    data = {"cwm_system": "w0_omni"}
    assert batch_given_frames(data, _batch("w0_omni")) == 1
    with pytest.raises(RuntimeError, match="not one of this run's"):
        batch_given_frames(data, _batch("wn_omni"))
    assert batch_given_frames(data, _batch("wn_omni"), continuation=True) == 10


def test_given_latents_replace_only_the_head_of_the_target() -> None:
    batch = _batch("wn_omni")
    given = torch.randn(1, 24, 10, 2, 2)
    spliced = with_given_latents(batch, given)
    torch.testing.assert_close(spliced.target_latents[:, :10], given[0])
    assert torch.all(spliced.target_latents[:, 10:] == 0) and torch.all(batch.target_latents == 0)
    with pytest.raises(Exception, match="do not fit"):
        with_given_latents(batch, given[:, :, :9])


def test_given_latents_fill_whole_chunks_then_the_head_of_the_next() -> None:
    assert [given_in_chunk(1, chunk) for chunk in range(3)] == [(0, 1), (5, 0), (10, 0)]
    assert [given_in_chunk(10, chunk) for chunk in range(3)] == [(0, 5), (5, 5), (10, 0)]
    assert given_in_chunk(7, 1) == (5, 2)


def test_take_config_needs_a_role_with_a_continuation() -> None:
    config = yaml.safe_load(INFER.read_text(encoding="utf-8"))
    config["inference"] = {"proxy_take": "/takes/take.json"}
    validate_h3_config(config)
    config["inference"]["proxy_take"] = ""
    with pytest.raises(ConfigurationError, match="non-empty path"):
        validate_h3_config(config)


def _stage1_core(given: int, monkeypatch: pytest.MonkeyPatch):
    core = object.__new__(H3Ref2VAStage1Core)
    core.device = torch.device("cpu")
    core.config = {"data": {"cwm_system": "w0_omni"}}
    core.train = {"keyframe_noise_augmentation": 0.999, "audio_timestep_shift": 3.0}
    core.target_frames, core.target_height, core.target_width = 37, 2, 2
    core.rollout_chunks = 8
    clean = torch.randn(1, 24, 37, 2, 2)
    pictures = (torch.zeros(1, 24, 1, 2, 2),)
    proxy = torch.zeros(1, 24, 37, 2, 2)
    inputs = (clean, pictures, proxy, torch.zeros(1, 3, 8), torch.zeros(3, dtype=torch.long), given)
    monkeypatch.setattr(core, "_inputs", lambda batch: inputs, raising=False)
    monkeypatch.setattr(core, "_audio_rows", lambda batch: torch.zeros(1, 2, 32), raising=False)
    monkeypatch.setattr(core, "window", lambda *a, **k: (SimpleNamespace(), None), raising=False)
    calls = []

    def model(layout, mask, *, rows, audio, prompt, times):
        return torch.zeros(1, int(rows.shape[1]), 96)

    def row_times(layout, **kwargs):
        calls.append(kwargs)
        return None

    monkeypatch.setattr(core, "_model", model, raising=False)
    monkeypatch.setattr(stage1_ref2va, "ref2va_row_time_pairs", row_times)
    return core, clean, calls


@pytest.mark.parametrize("given", [1, 10])
def test_stage1_rollout_holds_given_latents_across_chunks(
    given: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    core, clean, calls = _stage1_core(given, monkeypatch)
    out = core.generate(None, noise_seed=0, num_inference_steps=4)
    assert tuple(out.shape) == (1, 24, 37, 2, 2)
    torch.testing.assert_close(out[:, :, :given], clean[:, :, :given], atol=0.02, rtol=0)
    # Wholly given chunks are not denoised; the rest run four steps each.
    skipped = given // 5
    assert len(calls) == 4 * (8 - skipped)
    first = calls[0]
    assert first["current_chunk"] == skipped
    assert first["given_rows"] == given % 5
    assert first["given_start_row"] == skipped * 5


def test_stage2_rollout_commits_wholly_given_chunks_without_denoising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    core = object.__new__(H3Ref2VASGFCore)
    core.device = torch.device("cpu")
    core.video_shift = 12.0
    core.rollout_frames, core.rollout_chunks = 40, 8
    core.target_height = core.target_width = 2
    given_value = torch.randn(1, 24, 10, 2, 2)
    inputs = H3Ref2VASGFInputs(None, None, None, None, None, None, None, 10, given_value)
    monkeypatch.setattr(core, "full_window", lambda inputs: (SimpleNamespace(), None), raising=False)
    monkeypatch.setattr(core, "_prefill", lambda *a: None, raising=False)
    steps, commits = [], []

    def chunk_forward(inputs, layout, cache, *, chunk, latents, timestep, held, commit):
        (commits if commit else steps).append(chunk)
        if commit:
            return None
        x0 = torch.zeros_like(latents)
        if held is not None:
            x0[:, :, : held.shape[2]] = held
        return x0

    monkeypatch.setattr(core, "_chunk_forward", chunk_forward, raising=False)
    rollout = core.rollout(inputs, exit_index=3)
    assert commits == list(range(8))
    assert steps == [chunk for chunk in range(2, 8) for _ in range(4)]
    torch.testing.assert_close(rollout.cache_target[:, :, :10], given_value)
    assert torch.all(rollout.cache_target[:, :, 10:] == 0)
