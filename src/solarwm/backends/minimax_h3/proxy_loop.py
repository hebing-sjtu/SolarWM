"""Single-node H3 proxy experiments and a rolling checkpoint evaluator.

One 8-GPU node trains one experiment with ``local_file_launcher`` (no TCP rendezvous), so several
nodes can run independent experiments while another node evaluates their checkpoints as they
appear. Every command runs on a pod, normally inside a detached tmux session::

    python -m solarwm.backends.minimax_h3.proxy_loop train --name omni-a --config CFG [--set K=V]
    python -m solarwm.backends.minimax_h3.proxy_loop eval-worker --loop LOOP --runs omni-a omni-b \\
        --infer-config INFER_CFG --eval-data /data/.../heldout --steps 250 500
    python -m solarwm.backends.minimax_h3.proxy_loop status --runs omni-a omni-b --loop LOOP
    python -m solarwm.backends.minimax_h3.proxy_loop frames --loop LOOP --run omni-a --step 500

``/data`` is a GCS fuse mount: checkpoints are written there by training itself, but evaluation
outputs are produced under ``/workspace`` and copied afterwards because the inference writer
renames partial files.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

RUN_ROOT = Path("/data/binghe/h3_proxy/solarwm-runs")
EVAL_ROOT = Path("/data/binghe/h3_proxy/evals/loops")
LOCAL_ROOT = Path("/workspace/h3loop")
WANDB_KEY_FILE = Path("/data/binghe/.secrets/wandb_api_key")
NCCL_ENV_SCRIPT = Path("/usr/local/gib/scripts/set_nccl_env.sh")
NCCL_LIBRARY_DIR = "/usr/local/gib/lib64"
MODEL_SHARDS = Path("/data/models/MiniMax-H3/transformer_ref")
ABLATIONS = ("correct", "static", "shuffled")
# Eval outputs must not inherit training-only or launcher-owned sections of a run config.
_RUN_SECTIONS_FOR_EVAL = ("model", "data")


def _checkpoint_dir(run_dir: Path, step: int) -> Path:
    return run_dir / f"checkpoint_model_{int(step):06d}"


def read_events(run_dir: Path) -> list[dict[str, Any]]:
    """Training events, skipping a torn last line that the writer may still be appending."""

    path = run_dir / "training-events.jsonl"
    if not path.is_file():
        return []
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            events.append(value)
    return events


def completed_checkpoints(
    run_dir: Path, events: Sequence[Mapping[str, Any]] | None = None
) -> list[int]:
    """Steps whose checkpoint event was emitted and whose directory carries ``COMPLETE.json``."""

    rows = read_events(run_dir) if events is None else events
    steps = {int(row["step"]) for row in rows if row.get("event") == "checkpoint" and "step" in row}
    return sorted(
        step for step in steps if (_checkpoint_dir(run_dir, step) / "COMPLETE.json").is_file()
    )


def run_finished(run_dir: Path) -> bool:
    path = run_dir / "run-result.json"
    if not path.is_file():
        return False
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("status") in {"complete", "failed"}
    except (OSError, json.JSONDecodeError):
        return False


def loss_summary(events: Sequence[Mapping[str, Any]], *, beta: float = 0.98) -> dict[str, Any]:
    """Bias-corrected EMA and window means of the flow-matching loss, for trend reading."""

    steps = [row for row in events if row.get("event") == "optimizer_step"]
    if not steps:
        return {"steps": 0}
    ema = 0.0
    for index, row in enumerate(steps, 1):
        ema = beta * ema + (1.0 - beta) * float(row["losses"]["flow_matching"])
        corrected = ema / (1.0 - beta**index)
    window = max(1, min(100, len(steps) // 5))
    means = []
    for start in range(0, len(steps), window):
        chunk = steps[start : start + window]
        means.append(
            {
                "steps": f"{chunk[0]['step']}-{chunk[-1]['step']}",
                "loss": round(
                    sum(float(r["losses"]["flow_matching"]) for r in chunk) / len(chunk), 5
                ),
            }
        )
    last = steps[-1]
    recent = steps[-20:]
    return {
        "steps": int(last["step"]),
        "loss_ema": round(corrected, 5),
        "lr": float(last.get("lr", 0.0)),
        "grad_norm": round(float(last.get("gradient_norm", 0.0)), 5),
        "step_seconds": round(
            sum(float(r.get("compute_time_s", 0.0)) for r in recent) / len(recent), 1
        ),
        "peak_gib": round(max(float(r.get("peak_allocated_gib", 0.0)) for r in recent), 1),
        "window_means": means[-12:],
    }


# ----------------------------------------------------------------------
# Launch environment
# ----------------------------------------------------------------------


def launch_environment(base: Mapping[str, str] | None = None) -> dict[str, str]:
    """Process environment for 8-GPU H3 jobs: cluster NCCL setup, allocator, W&B key."""

    env = dict(os.environ if base is None else base)
    if NCCL_ENV_SCRIPT.is_file():
        probe = subprocess.run(
            ["bash", "-c", f"source {NCCL_ENV_SCRIPT} >/dev/null 2>&1; env -0"],
            check=True,
            capture_output=True,
            env=env,
        )
        for item in probe.stdout.split(b"\0"):
            if b"=" in item:
                key, value = item.split(b"=", 1)
                env[key.decode()] = value.decode()
        env["LD_LIBRARY_PATH"] = ":".join(
            part for part in (NCCL_LIBRARY_DIR, env.get("LD_LIBRARY_PATH", "")) if part
        )
    env.update(
        {
            "TORCH_NCCL_ENABLE_MONITORING": "0",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
    if not env.get("WANDB_API_KEY") and WANDB_KEY_FILE.is_file():
        key = WANDB_KEY_FILE.read_text(encoding="utf-8").strip()
        if key:
            env["WANDB_API_KEY"] = key
    return env


def warm_model_shards(root: Path = MODEL_SHARDS) -> None:
    """Read every transformer shard once so eight ranks load from page cache, not cold GCS."""

    files = sorted(root.rglob("*.safetensors"))
    buffer = bytearray(64 * 1024 * 1024)
    started = time.time()
    total = 0
    for path in files:
        with path.open("rb", buffering=0) as handle:
            while count := handle.readinto(buffer):
                total += count
    print(
        f"[h3-loop] warmed {len(files)} shards, {total / 2**30:.1f} GiB "
        f"in {time.time() - started:.0f}s",
        flush=True,
    )


def _launch(
    command: Sequence[str],
    *,
    nproc: int,
    store_dir: Path,
    env: Mapping[str, str],
    log: Path | None = None,
) -> int:
    store_dir.mkdir(parents=True, exist_ok=True)
    store = store_dir / f"{os.getpid()}-{time.time_ns()}.store"
    argv = [
        sys.executable,
        "-m",
        "solarwm.runtime.local_file_launcher",
        "--nproc-per-node",
        str(nproc),
        "--store-path",
        str(store),
        "--",
        *command,
    ]
    if log is None:
        return subprocess.run(argv, env=dict(env), check=False).returncode
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("ab") as handle:
        process = subprocess.Popen(
            argv, env=dict(env), stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        assert process.stdout is not None
        for chunk in iter(lambda: process.stdout.readline(), b""):
            handle.write(chunk)
            handle.flush()
            sys.stdout.buffer.write(chunk)
            sys.stdout.flush()
        return process.wait()


def _load(config_path: Path, overrides: Sequence[str], *, validate: bool) -> dict[str, Any]:
    from solarwm.config import load_config

    from .config import validate_h3_config

    values = load_config(config_path, overrides).mutable_copy()
    if validate:
        validate_h3_config(values)
    return values


# ----------------------------------------------------------------------
# train
# ----------------------------------------------------------------------


def train_overrides(
    config: Mapping[str, Any],
    *,
    name: str,
    nproc: int,
    run_dir: Path,
    resume: str,
    user_overrides: Sequence[str],
) -> list[str]:
    """Overrides that make a cluster config a single-node run; user overrides still win."""

    distributed = config["distributed"]
    train = config["train"]
    sp = int(distributed["sequence_parallel_size"])
    if nproc % sp:
        raise SystemExit(f"--nproc {nproc} is not divisible by sequence_parallel_size {sp}")
    global_batch = (
        nproc // sp * int(train["micro_batch_size"]) * int(train["gradient_accumulation_steps"])
    )
    overrides = [
        f"distributed.world_size={nproc}",
        f"train.global_batch_size={global_batch}",
        f"runtime.output_dir={run_dir}",
        f"runtime.tracking.run_name={name}",
    ]
    explicit_resume = any(item.startswith("checkpoint.resume_from=") for item in user_overrides)
    if resume == "auto" and not explicit_resume:
        steps = completed_checkpoints(run_dir)
        if steps:
            overrides.append(f"checkpoint.resume_from={_checkpoint_dir(run_dir, steps[-1])}")
    return [*overrides, *user_overrides]


def command_train(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_root) / args.name
    if run_finished(run_dir) and not args.allow_finished:
        print(f"[h3-loop] {run_dir} already finished; pass --allow-finished to resume past it")
        return 1
    base = _load(Path(args.config), args.set, validate=False)
    overrides = train_overrides(
        base,
        name=args.name,
        nproc=args.nproc,
        run_dir=run_dir,
        resume=args.resume,
        user_overrides=args.set,
    )
    resolved = _load(Path(args.config), overrides, validate=True)
    print(
        "[h3-loop] train "
        + json.dumps(
            {
                "name": args.name,
                "output_dir": str(run_dir),
                "world_size": resolved["distributed"]["world_size"],
                "global_batch_size": resolved["train"]["global_batch_size"],
                "max_steps": resolved["train"]["max_steps"],
                "resume_from": resolved.get("checkpoint", {}).get("resume_from"),
                "overrides": overrides,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    if args.dry_run:
        return 0
    env = launch_environment()
    if not env.get("WANDB_API_KEY") and resolved["runtime"].get("tracking", {}).get("enabled"):
        print(f"[h3-loop] warning: no W&B key in env or {WANDB_KEY_FILE}", flush=True)
    if not args.no_warm:
        warm_model_shards()
    command = ["-m", "solarwm", "train", "--config", str(Path(args.config).resolve())]
    for item in overrides:
        command += ["--set", item]
    return _launch(command, nproc=args.nproc, store_dir=LOCAL_ROOT / "stores", env=env)


# ----------------------------------------------------------------------
# eval-worker
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class EvalTask:
    run: str
    step: int
    ablation: str

    @property
    def key(self) -> str:
        return f"{self.run}/step-{self.step:06d}/{self.ablation}"


def pending_tasks(
    runs: Sequence[str],
    *,
    run_root: Path,
    eval_dir: Path,
    steps: Sequence[int],
    ablations: Sequence[str],
    failed: Mapping[str, int],
    max_attempts: int,
) -> list[EvalTask]:
    """Ready, not yet evaluated work, lowest step first so every run gets early feedback."""

    tasks = []
    for run in runs:
        ready = completed_checkpoints(run_root / run)
        wanted = [step for step in ready if not steps or step in steps]
        for step in wanted:
            for ablation in ablations:
                task = EvalTask(run, step, ablation)
                if (eval_dir / task.key / "DONE.json").is_file():
                    continue
                if failed.get(task.key, 0) >= max_attempts:
                    continue
                tasks.append(task)
    order = {run: index for index, run in enumerate(runs)}
    return sorted(
        tasks, key=lambda task: (task.step, order[task.run], ABLATIONS.index(task.ablation))
    )


def eval_config(
    infer_base: Mapping[str, Any],
    run_config: Mapping[str, Any],
    *,
    task: EvalTask,
    checkpoint: Path,
    eval_data: str | None,
    weight_source: str,
    output_dir: Path,
    loop: str,
) -> dict[str, Any]:
    """An infer config that carries exactly the trained run's model and data contract."""

    config = json.loads(json.dumps(infer_base))
    for section in _RUN_SECTIONS_FOR_EVAL:
        config[section] = json.loads(json.dumps(run_config[section]))
    if eval_data:
        config["data"]["data_path"] = eval_data
    config["name"] = f"{config.get('name', 'h3-proxy-infer')}-{task.run}"
    config.setdefault("checkpoint", {})
    config["checkpoint"]["resume_from"] = str(checkpoint)
    config["checkpoint"]["weight_source"] = weight_source
    config.setdefault("validation", {})["proxy_ablation"] = task.ablation
    config["runtime"]["output_dir"] = str(output_dir)
    tracking = config["runtime"].setdefault("tracking", {})
    tracking.update(
        {
            "run_name": f"{task.run}-s{task.step}-{task.ablation}",
            "group": loop,
            "run_id": None,
            "tags": sorted(
                {*tracking.get("tags", []), task.run, task.ablation, f"step-{task.step}"}
            ),
        }
    )
    return config


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _copy_tree(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for item in source.rglob("*"):
        if "wandb" in item.relative_to(source).parts[:1] or item.suffix == ".store":
            continue
        target = destination / item.relative_to(source)
        if item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif item.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(item, target)


def _task_config(task: EvalTask, args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    from .config import validate_h3_config

    run_dir = Path(args.run_root) / task.run
    run_config = json.loads((run_dir / "resolved-config.json").read_text(encoding="utf-8"))
    config = eval_config(
        _load(Path(args.infer_config), args.set, validate=False),
        run_config,
        task=task,
        checkpoint=_checkpoint_dir(run_dir, task.step),
        eval_data=args.eval_data,
        weight_source=args.weight_source,
        output_dir=output_dir,
        loop=args.loop,
    )
    validate_h3_config(config)
    return config


def run_eval_task(task: EvalTask, args: argparse.Namespace, env: Mapping[str, str]) -> bool:
    run_dir = Path(args.run_root) / task.run
    eval_dir = Path(args.eval_root) / args.loop
    local = LOCAL_ROOT / "evals" / args.loop / task.key
    if local.exists():
        shutil.rmtree(local)
    local.mkdir(parents=True)
    generated = local.parent / f"{task.ablation}.infer.json"
    _write_json(generated, _task_config(task, args, local))
    print(f"[h3-loop] eval {task.key}", flush=True)
    started = time.time()
    code = _launch(
        ["-m", "solarwm", "infer", "--config", str(generated)],
        nproc=args.nproc,
        store_dir=LOCAL_ROOT / "stores",
        env=env,
        log=local / "launch.log",
    )
    complete = local / "proxy-inference" / "COMPLETE.json"
    destination = eval_dir / task.key
    _copy_tree(local, destination)
    shutil.copyfile(generated, destination / "infer-config.json")
    if code != 0 or not complete.is_file():
        print(f"[h3-loop] eval {task.key} failed with exit code {code}", flush=True)
        return False
    summary = json.loads(complete.read_text(encoding="utf-8"))
    _write_json(
        destination / "DONE.json",
        {
            "task": task.key,
            "checkpoint": str(_checkpoint_dir(run_dir, task.step)),
            "weight_source": args.weight_source,
            "samples": summary.get("samples", []),
            "seconds": round(time.time() - started),
        },
    )
    return True


def command_eval_worker(args: argparse.Namespace) -> int:
    for ablation in args.ablations:
        if ablation not in ABLATIONS:
            raise SystemExit(f"--ablations must be drawn from {list(ABLATIONS)}")
    eval_dir = Path(args.eval_root) / args.loop
    failures_path = eval_dir / "failures.json"
    failed: dict[str, int] = (
        json.loads(failures_path.read_text(encoding="utf-8")) if failures_path.is_file() else {}
    )
    if args.plan:
        tasks = pending_tasks(
            args.runs,
            run_root=Path(args.run_root),
            eval_dir=eval_dir,
            steps=args.steps,
            ablations=args.ablations,
            failed=failed,
            max_attempts=args.max_attempts,
        )
        for task in tasks:
            config = _task_config(task, args, LOCAL_ROOT / "evals" / args.loop / task.key)
            print(
                f"{task.key}  data={config['data']['data_path']}  "
                f"align={config['data']['align_proxy_reference_time']}  "
                f"weights={config['checkpoint']['weight_source']}"
            )
        print(f"[h3-loop] {len(tasks)} pending; failures={failed}")
        return 0
    env = launch_environment()
    if not args.no_warm:
        warm_model_shards()
    while True:
        tasks = pending_tasks(
            args.runs,
            run_root=Path(args.run_root),
            eval_dir=eval_dir,
            steps=args.steps,
            ablations=args.ablations,
            failed=failed,
            max_attempts=args.max_attempts,
        )
        if tasks:
            task = tasks[0]
            if not run_eval_task(task, args, env):
                failed[task.key] = failed.get(task.key, 0) + 1
                _write_json(failures_path, failed)
            continue
        if args.once or all(run_finished(Path(args.run_root) / run) for run in args.runs):
            print("[h3-loop] eval worker: nothing pending", flush=True)
            return 1 if failed else 0
        time.sleep(args.poll_seconds)


# ----------------------------------------------------------------------
# status / frames
# ----------------------------------------------------------------------


def run_status(run: str, *, run_root: Path, eval_dir: Path | None) -> dict[str, Any]:
    run_dir = run_root / run
    events = read_events(run_dir)
    status: dict[str, Any] = {
        "run": run,
        "exists": run_dir.is_dir(),
        "finished": run_finished(run_dir),
        "checkpoints": completed_checkpoints(run_dir, events),
        **loss_summary(events),
    }
    if eval_dir is not None and (eval_dir / run).is_dir():
        status["evals"] = sorted(
            str(path.parent.relative_to(eval_dir / run))
            for path in (eval_dir / run).rglob("DONE.json")
        )
    return status


def command_status(args: argparse.Namespace) -> int:
    eval_dir = Path(args.eval_root) / args.loop if args.loop else None
    payload = [
        run_status(run, run_root=Path(args.run_root), eval_dir=eval_dir) for run in args.runs
    ]
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def command_frames(args: argparse.Namespace) -> int:
    """Tile a few time points of every compare.mp4 into one PNG per slot for quick review."""

    root = Path(args.eval_root) / args.loop / args.run / f"step-{args.step:06d}" / args.ablation
    out = Path(
        args.output or LOCAL_ROOT / "frames" / args.loop / args.run / f"step-{args.step:06d}"
    )
    out.mkdir(parents=True, exist_ok=True)
    videos = sorted(root.rglob("compare.mp4"))
    if not videos:
        print(f"[h3-loop] no compare.mp4 under {root}")
        return 1
    interval = max(1, args.frames_total // args.count)
    for video in videos:
        slot = next((part for part in video.parts if part.startswith("slot-")), video.parent.name)
        target = out / f"{args.ablation}-{slot}.png"
        subprocess.run(
            [
                "ffmpeg",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(video),
                "-vf",
                f"select='not(mod(n\\,{interval}))',scale={args.width}:-2,tile=1x{args.count}",
                "-frames:v",
                "1",
                str(target),
            ],
            check=True,
        )
        print(target)
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)

    def common(child: argparse.ArgumentParser) -> None:
        child.add_argument("--run-root", default=str(RUN_ROOT))
        child.add_argument("--eval-root", default=str(EVAL_ROOT))

    train = commands.add_parser("train", help="train one experiment on this node")
    common(train)
    train.add_argument("--name", required=True)
    train.add_argument("--config", required=True)
    train.add_argument("--nproc", type=int, default=8)
    train.add_argument("--resume", choices=("auto", "never"), default="auto")
    train.add_argument("--allow-finished", action="store_true")
    train.add_argument("--no-warm", action="store_true")
    train.add_argument("--dry-run", action="store_true")
    train.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    train.set_defaults(handler=command_train)

    worker = commands.add_parser(
        "eval-worker", help="evaluate checkpoints of several runs as they land"
    )
    common(worker)
    worker.add_argument(
        "--loop", required=True, help="name grouping this round's evals and W&B runs"
    )
    worker.add_argument("--runs", nargs="+", required=True)
    worker.add_argument("--infer-config", required=True)
    worker.add_argument(
        "--eval-data", help="held-out cache; defaults to the run's training data_path"
    )
    worker.add_argument(
        "--steps", nargs="*", type=int, default=[], help="empty means every checkpoint"
    )
    worker.add_argument("--ablations", nargs="+", default=["correct"])
    worker.add_argument("--weight-source", choices=("ema", "live"), default="ema")
    worker.add_argument("--nproc", type=int, default=8)
    worker.add_argument("--poll-seconds", type=int, default=300)
    worker.add_argument("--max-attempts", type=int, default=2)
    worker.add_argument("--once", action="store_true", help="drain what is ready, then exit")
    worker.add_argument(
        "--plan", action="store_true", help="validate and list pending evals without GPUs"
    )
    worker.add_argument("--no-warm", action="store_true")
    worker.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override the infer base config",
    )
    worker.set_defaults(handler=command_eval_worker)

    status = commands.add_parser("status", help="print loss trend, checkpoints and evals as JSON")
    common(status)
    status.add_argument("--runs", nargs="+", required=True)
    status.add_argument("--loop")
    status.set_defaults(handler=command_status)

    frames = commands.add_parser("frames", help="tile compare.mp4 frames for review")
    common(frames)
    frames.add_argument("--loop", required=True)
    frames.add_argument("--run", required=True)
    frames.add_argument("--step", type=int, required=True)
    frames.add_argument("--ablation", default="correct")
    frames.add_argument("--count", type=int, default=5)
    frames.add_argument("--frames-total", type=int, default=124)
    frames.add_argument("--width", type=int, default=1536)
    frames.add_argument("--output")
    frames.set_defaults(handler=command_frames)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(None if argv is None else list(argv))
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
