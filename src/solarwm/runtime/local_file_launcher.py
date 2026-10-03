"""Launch one local distributed job without a TCP rendezvous address."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="launch local SolarWM workers through a PyTorch FileStore"
    )
    parser.add_argument("--nproc-per-node", type=int, required=True)
    parser.add_argument("--store-path", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def _terminate(processes: list[subprocess.Popen[bytes]]) -> None:
    for process in processes:
        if process.poll() is None:
            process.terminate()
    deadline = time.monotonic() + 10.0
    for process in processes:
        if process.poll() is None:
            try:
                process.wait(timeout=max(0.0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                process.kill()
    for process in processes:
        if process.poll() is None:
            process.wait()


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    world = int(args.nproc_per_node)
    if world < 1:
        raise SystemExit("--nproc-per-node must be positive")
    command = list(args.command)
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        raise SystemExit("a Python command is required after --")

    store = args.store_path.expanduser()
    if not store.is_absolute():
        raise SystemExit("--store-path must be absolute")
    store.parent.mkdir(parents=True, exist_ok=True)
    if store.exists() or store.is_symlink():
        raise SystemExit(f"FileStore path already exists: {store}")

    base_environment = os.environ.copy()
    for name in (
        "MASTER_ADDR",
        "MASTER_PORT",
        "PET_MASTER_ADDR",
        "PET_MASTER_PORT",
        "PET_NODE_RANK",
        "PET_NNODES",
        "PET_NPROC_PER_NODE",
    ):
        base_environment.pop(name, None)
    base_environment["WORLD_SIZE"] = str(world)
    base_environment["LOCAL_WORLD_SIZE"] = str(world)
    base_environment["SOLARWM_DISTRIBUTED_INIT_METHOD"] = f"file://{store}"

    processes: list[subprocess.Popen[bytes]] = []
    try:
        for rank in range(world):
            environment = base_environment.copy()
            environment["RANK"] = str(rank)
            environment["LOCAL_RANK"] = str(rank)
            processes.append(
                subprocess.Popen(
                    [sys.executable, *command],
                    env=environment,
                )
            )

        remaining = set(range(world))
        failure = 0
        while remaining:
            for index in tuple(remaining):
                result = processes[index].poll()
                if result is None:
                    continue
                remaining.remove(index)
                if result != 0 and failure == 0:
                    failure = int(result)
                    print(
                        f"SolarWM local worker rank {index} failed with exit code {result}",
                        file=sys.stderr,
                        flush=True,
                    )
                    _terminate(processes)
                    remaining.clear()
                    break
            if remaining:
                time.sleep(0.2)
        return failure
    except KeyboardInterrupt:
        _terminate(processes)
        return 130
    finally:
        store.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
