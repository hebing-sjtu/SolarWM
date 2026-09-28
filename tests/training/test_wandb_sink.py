from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from solarwm.errors import BackendContractError
from solarwm.training.wandb_sink import WandbEventSink


class FakeRun:
    def __init__(self) -> None:
        self.summary: dict[str, Any] = {}
        self.logged: list[tuple[int, dict[str, Any]]] = []
        self.finished = False

    def log(self, values: dict[str, Any], *, step: int) -> None:
        self.logged.append((step, dict(values)))

    def finish(self) -> None:
        self.finished = True


class FakeWandb:
    def __init__(self) -> None:
        self.init_calls: list[dict[str, Any]] = []
        self.runs: list[FakeRun] = []

    def init(self, **kwargs: Any) -> FakeRun:
        self.init_calls.append(dict(kwargs))
        run = FakeRun()
        self.runs.append(run)
        return run


def _tracking(**values: Any) -> dict[str, Any]:
    return {
        "project": "solarwm-h3-proxy",
        "run_name": "proxy-test",
        "entity": None,
        "group": "tests",
        "tags": ["h3", "proxy"],
        "run_id": None,
        "resume": "allow",
        "mode": "online",
        **values,
    }


def test_wandb_sink_maps_optimizer_and_checkpoint_events(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "solarwm.training.wandb_sink.secrets.token_hex",
        lambda _bytes: "generated-run-id",
    )
    wandb = FakeWandb()
    sink = WandbEventSink(
        _tracking(),
        output_dir=tmp_path,
        resolved_config={"schema": "solarwm.run.v1", "name": "test"},
        wandb_module=wandb,
    )
    sink(
        {
            "event": "optimizer_step",
            "step": 3,
            "losses": {"flow_matching": 0.25},
            "lr": 2e-5,
            "gradient_norm": 1.5,
            "compute_time_s": 12.0,
            "peak_allocated_gib": 80.0,
        }
    )
    sink(
        {
            "event": "checkpoint",
            "step": 48,
            "checkpoint_id": "checkpoint_model_000048",
        }
    )
    sink.finish()

    assert (tmp_path / "wandb-run-id.txt").read_text().strip() == "generated-run-id"
    assert wandb.init_calls[0]["id"] == "generated-run-id"
    assert wandb.runs[0].logged[0] == (
        3,
        {
            "train/loss/flow_matching": 0.25,
            "train/learning_rate": 2e-5,
            "train/gradient_norm": 1.5,
            "perf/compute_time_s": 12.0,
            "perf/peak_allocated_gib": 80.0,
        },
    )
    assert wandb.runs[0].summary["checkpoint/last_step"] == 48
    assert wandb.runs[0].finished


def test_wandb_sink_reuses_run_id_and_rejects_conflict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "solarwm.training.wandb_sink.secrets.token_hex",
        lambda _bytes: "generated-run-id",
    )
    wandb = FakeWandb()
    first = WandbEventSink(
        _tracking(),
        output_dir=tmp_path,
        resolved_config={},
        wandb_module=wandb,
    )
    first.finish()
    second = WandbEventSink(
        _tracking(),
        output_dir=tmp_path,
        resolved_config={},
        wandb_module=wandb,
    )
    assert second.run_id == "generated-run-id"
    second.finish()

    with pytest.raises(BackendContractError, match="differs from persisted"):
        WandbEventSink(
            _tracking(run_id="different-id"),
            output_dir=tmp_path,
            resolved_config={},
            wandb_module=wandb,
        )
