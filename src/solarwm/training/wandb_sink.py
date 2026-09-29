"""Optional rank-owned Weights & Biases sink for canonical training events."""

from __future__ import annotations

import json
import os
import secrets
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from solarwm.errors import BackendContractError


def _optional_text(config: Mapping[str, Any], key: str) -> str | None:
    value = config.get(key)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _run_id(config: Mapping[str, Any], output_dir: Path) -> str:
    """Resolve one stable W&B identity across checkpoint resumes."""

    path = output_dir / "wandb-run-id.txt"
    configured = _optional_text(config, "run_id")
    if path.is_file():
        persisted = path.read_text(encoding="utf-8").strip()
        if not persisted:
            raise BackendContractError(f"W&B run-id file is empty: {path}")
        if configured is not None and configured != persisted:
            raise BackendContractError(
                f"configured W&B run_id={configured!r} differs from persisted {persisted!r}"
            ) from None
        return persisted
    selected = configured or secrets.token_hex(4)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        persisted = path.read_text(encoding="utf-8").strip()
        if not persisted:
            raise BackendContractError(f"W&B run-id file is empty: {path}") from None
        if configured is not None and configured != persisted:
            raise BackendContractError(
                f"configured W&B run_id={configured!r} differs from persisted {persisted!r}"
            ) from None
        return persisted
    try:
        os.write(descriptor, f"{selected}\n".encode())
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return selected


class WandbEventSink:
    """Translate SolarWM scalar events into one resumable W&B run."""

    def __init__(
        self,
        tracking: Mapping[str, Any],
        *,
        output_dir: str | Path,
        resolved_config: Mapping[str, Any],
        wandb_module: Any | None = None,
    ) -> None:
        if wandb_module is None:
            try:
                import wandb as wandb_module
            except ImportError as exc:  # pragma: no cover - runtime dependency
                raise BackendContractError(
                    "W&B tracking is enabled but wandb is not installed; "
                    "install SolarWM[tracking] or `python -m pip install wandb`"
                ) from exc
        self.wandb = wandb_module
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = _run_id(tracking, self.output_dir)
        self.loss_ema_beta = float(tracking.get("loss_ema_beta", 0.95))
        self._loss_ema: dict[str, float] = {}
        tags = [str(value) for value in tracking.get("tags", ())]
        kwargs: dict[str, Any] = {
            "project": str(tracking["project"]),
            "name": _optional_text(tracking, "run_name"),
            "entity": _optional_text(tracking, "entity"),
            "group": _optional_text(tracking, "group"),
            "tags": tags,
            "id": self.run_id,
            "resume": str(tracking.get("resume", "allow")),
            "mode": str(tracking.get("mode", "online")),
            "dir": str(self.output_dir),
            "config": json.loads(json.dumps(dict(resolved_config), sort_keys=True)),
        }
        self.run = self.wandb.init(**kwargs)
        if self.run is None:
            raise BackendContractError("wandb.init returned no run")

    def __call__(self, event: Mapping[str, Any]) -> None:
        event_type = str(event.get("event", ""))
        step = int(event.get("step", 0))
        if step < 1:
            raise BackendContractError(f"W&B event has invalid step {step}")
        metrics: dict[str, Any] = {}
        if event_type == "optimizer_step":
            losses = event.get("losses", {})
            if not isinstance(losses, Mapping):
                raise BackendContractError("W&B optimizer event losses must be a mapping")
            for key, value in losses.items():
                scalar = float(value)
                previous = self._loss_ema.get(str(key))
                smoothed = (
                    scalar
                    if previous is None
                    else self.loss_ema_beta * previous + (1.0 - self.loss_ema_beta) * scalar
                )
                self._loss_ema[str(key)] = smoothed
                metrics[f"train/loss/{key}"] = scalar
                metrics[f"train/loss_ema/{key}"] = smoothed
            for source, destination in (
                ("lr", "train/learning_rate"),
                ("gradient_norm", "train/gradient_norm"),
                ("compute_time_s", "perf/compute_time_s"),
                ("peak_allocated_gib", "perf/peak_allocated_gib"),
            ):
                if source in event:
                    metrics[destination] = float(event[source])
        elif event_type == "checkpoint":
            metrics["checkpoint/saved"] = 1
            self.run.summary["checkpoint/last_id"] = str(event.get("checkpoint_id", ""))
            self.run.summary["checkpoint/last_step"] = step
        elif event_type == "validation":
            report = event.get("report", {})
            if isinstance(report, Mapping):
                metrics.update(
                    {
                        f"validation/{key}": float(value)
                        for key, value in report.items()
                        if isinstance(value, (int, float)) and not isinstance(value, bool)
                    }
                )
        if metrics:
            self.run.log(metrics, step=step)

    def finish(self) -> None:
        self.run.finish()


__all__ = ["WandbEventSink"]
