"""Single-node experiment launch and rolling-evaluation helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from solarwm.backends.minimax_h3 import proxy_loop
from solarwm.backends.minimax_h3.config import validate_h3_config
from solarwm.backends.minimax_h3.proxy_loop import (
    EvalTask,
    completed_checkpoints,
    eval_config,
    launch_environment,
    loss_summary,
    pending_tasks,
    train_overrides,
)
from solarwm.config import load_config

ROOT = Path(__file__).resolve().parents[3]
EXAMPLES = ROOT / "configs/examples/minimax_h3"
MIXED = EXAMPLES / "stage0p5-124f-ref2va-omni-mixed-low-704p-sp1.yaml"
MIXED_INFER = EXAMPLES / "infer-stage0p5-124f-ref2va-omni-mixed-low-704p-sp8.yaml"


def _run(root: Path, name: str, *, complete: list[int], torn: list[int] = ()) -> Path:
    run = root / name
    run.mkdir(parents=True)
    lines = [
        json.dumps(
            {"event": "optimizer_step", "step": step, "losses": {"flow_matching": 1.0 / step}}
        )
        for step in range(1, 11)
    ]
    for step in [*complete, *torn]:
        lines.append(json.dumps({"event": "checkpoint", "step": step}))
        directory = run / f"checkpoint_model_{step:06d}"
        directory.mkdir()
        if step in complete:
            (directory / "COMPLETE.json").write_text("{}")
    (run / "training-events.jsonl").write_text("\n".join(lines) + '\n{"event": "optimizer_st')
    return run


def test_checkpoints_require_event_and_complete_marker(tmp_path: Path) -> None:
    run = _run(tmp_path, "a", complete=[5, 10], torn=[15])
    (run / "checkpoint_model_000020").mkdir()
    assert completed_checkpoints(run) == [5, 10]


def test_single_node_overrides_rescale_batch_and_resume_latest(tmp_path: Path) -> None:
    run = _run(tmp_path, "omni-a", complete=[5, 10], torn=[15])
    base = load_config(MIXED).mutable_copy()
    assert (base["distributed"]["world_size"], base["train"]["global_batch_size"]) == (16, 16)
    overrides = train_overrides(
        base,
        name="omni-a",
        nproc=8,
        run_dir=run,
        resume="auto",
        user_overrides=["data.align_proxy_reference_time=true"],
    )
    assert overrides[-1] == "data.align_proxy_reference_time=true"
    assert f"checkpoint.resume_from={run / 'checkpoint_model_000010'}" in overrides
    resolved = load_config(MIXED, overrides).mutable_copy()
    validate_h3_config(resolved)
    assert resolved["distributed"]["world_size"] == 8
    assert resolved["train"]["global_batch_size"] == 8
    assert resolved["runtime"]["output_dir"] == str(run)
    assert resolved["runtime"]["tracking"]["run_name"] == "omni-a"
    assert resolved["data"]["align_proxy_reference_time"] is True


def test_explicit_or_disabled_resume_is_not_replaced(tmp_path: Path) -> None:
    run = _run(tmp_path, "omni-a", complete=[5])
    base = load_config(MIXED).mutable_copy()
    never = train_overrides(base, name="a", nproc=8, run_dir=run, resume="never", user_overrides=[])
    assert not any(item.startswith("checkpoint.resume_from=") for item in never)
    explicit = train_overrides(
        base,
        name="a",
        nproc=8,
        run_dir=run,
        resume="auto",
        user_overrides=["checkpoint.resume_from=/elsewhere"],
    )
    assert [item for item in explicit if item.startswith("checkpoint.resume_from=")] == [
        "checkpoint.resume_from=/elsewhere"
    ]
    with pytest.raises(SystemExit, match="not divisible"):
        train_overrides(
            {**base, "distributed": {**base["distributed"], "sequence_parallel_size": 3}},
            name="a",
            nproc=8,
            run_dir=run,
            resume="never",
            user_overrides=[],
        )


def test_pending_tasks_are_lowest_step_first_and_skip_done_or_exhausted(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    evals = tmp_path / "evals"
    _run(runs, "a", complete=[5, 10])
    _run(runs, "b", complete=[5])
    (evals / "a/step-000005/correct").mkdir(parents=True)
    (evals / "a/step-000005/correct/DONE.json").write_text("{}")
    tasks = pending_tasks(
        ["a", "b"],
        run_root=runs,
        eval_dir=evals,
        steps=[],
        ablations=["correct", "static"],
        failed={"b/step-000005/static": 2},
        max_attempts=2,
    )
    assert [task.key for task in tasks] == [
        "a/step-000005/static",
        "b/step-000005/correct",
        "a/step-000010/correct",
        "a/step-000010/static",
    ]
    only = pending_tasks(
        ["a", "b"],
        run_root=runs,
        eval_dir=evals,
        steps=[10],
        ablations=["correct"],
        failed={},
        max_attempts=2,
    )
    assert [task.key for task in only] == ["a/step-000010/correct"]


def test_eval_config_carries_trained_contract_and_validates(tmp_path: Path) -> None:
    trained = load_config(MIXED, ["data.align_proxy_reference_time=true"]).mutable_copy()
    infer = load_config(MIXED_INFER).mutable_copy()
    task = EvalTask("omni-b", 500, "shuffled")
    config = eval_config(
        infer,
        trained,
        task=task,
        checkpoint=tmp_path / "checkpoint_model_000500",
        eval_data="/data/heldout",
        weight_source="live",
        output_dir=tmp_path / "out",
        loop="round1",
    )
    validate_h3_config(config)
    assert config["data"]["align_proxy_reference_time"] is True
    assert config["data"]["data_path"] == "/data/heldout"
    assert config["checkpoint"] == {
        "resume_from": str(tmp_path / "checkpoint_model_000500"),
        "weight_source": "live",
    }
    assert config["validation"]["proxy_ablation"] == "shuffled"
    tracking = config["runtime"]["tracking"]
    assert (tracking["run_name"], tracking["group"]) == ("omni-b-s500-shuffled", "round1")
    assert {"omni-b", "shuffled", "step-500"} <= set(tracking["tags"])
    assert infer["data"]["align_proxy_reference_time"] is False


def test_loss_summary_reports_trend() -> None:
    events = [
        {
            "event": "optimizer_step",
            "step": step,
            "losses": {"flow_matching": 2.0 - step / 100},
            "lr": 1e-4,
            "compute_time_s": 80.0,
            "peak_allocated_gib": 57.0,
        }
        for step in range(1, 101)
    ]
    summary = loss_summary([{"event": "checkpoint", "step": 50}, *events])
    assert summary["steps"] == 100
    assert summary["step_seconds"] == 80.0
    assert summary["window_means"][0]["loss"] > summary["window_means"][-1]["loss"]
    assert loss_summary([]) == {"steps": 0}


def test_launch_environment_reads_key_file_without_overriding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    key = tmp_path / "key"
    key.write_text("k" * 40 + "\n")
    monkeypatch.setattr(proxy_loop, "WANDB_KEY_FILE", key)
    monkeypatch.setattr(proxy_loop, "NCCL_ENV_SCRIPT", tmp_path / "missing.sh")
    env = launch_environment({"PATH": "/bin"})
    assert env["WANDB_API_KEY"] == "k" * 40
    assert env["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"
    assert launch_environment({"WANDB_API_KEY": "mine"})["WANDB_API_KEY"] == "mine"
