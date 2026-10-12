#!/usr/bin/env python3
"""Mirror an exported eval board from a BCS pod to this host, fetching only what is missing.

Media files are content-addressed and never rewritten, so a name with the same size is already
here. A new index.json replaces the old one only once every video it names is here, so the page
never points at a missing file. Pods have no rsync, so files travel as tar streams over
`kubectl exec`; those streams can drop mid-transfer, so batches are kept small and whatever
arrived whole is kept. Nothing here is ever deleted.

Standard library only and Python 3.6 compatible.

  python3 sync.py --pod-pattern '^bingghhe.*-node-0-3-' --remote /workspace/eval-board \\
      --local /data/eval-board --interval 300
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time

INDEX = ("index.json", "index.html")
BATCH_BYTES = 256 << 20
ATTEMPTS = 4


def find_pod(pattern, namespace):
    out = subprocess.run(
        [
            "kubectl",
            "-n",
            namespace,
            "--request-timeout=60s",
            "get",
            "pods",
            "--no-headers",
            "--sort-by=.status.startTime",
            "-o",
            "custom-columns=NAME:.metadata.name",
        ],
        stdout=subprocess.PIPE,
        check=True,
    ).stdout.decode("utf-8")
    names = [name for name in out.split() if re.search(pattern, name)]
    if not names:
        raise RuntimeError(f"no pod matches {pattern!r}")
    return names[-1]


def exec_args(pod, args, *command, stdin=False):
    return [
        "kubectl",
        "-n",
        args.namespace,
        "exec",
        *(["-i"] if stdin else []),
        pod,
        "-c",
        args.container,
        "--",
        *command,
    ]


def remote_files(pod, args):
    script = f"cd {args.remote} && find . -path './.*' -prune -o -type f -printf '%P\\t%s\\n'"
    out = subprocess.run(
        exec_args(pod, args, "sh", "-c", script), stdout=subprocess.PIPE, check=True
    ).stdout.decode("utf-8")
    files = {}
    for line in out.splitlines():
        name, _, size = line.rpartition("\t")
        if name:
            files[name] = int(size)
    return files


def missing(remote, local_root):
    wanted = []
    for name, size in sorted(remote.items()):
        if name in INDEX:
            continue
        path = os.path.join(local_root, name)
        if not (os.path.isfile(path) and os.path.getsize(path) == size):
            wanted.append(name)
    return wanted


def batches(names, sizes, limit=BATCH_BYTES):
    batch, total = [], 0
    for name in names:
        if batch and total + sizes[name] > limit:
            yield batch
            batch, total = [], 0
        batch.append(name)
        total += sizes[name]
    if batch:
        yield batch


def pull(pod, names, args):
    """Stream names from the pod into a fresh .incoming directory and return its path."""
    incoming = os.path.join(args.local, ".incoming")
    shutil.rmtree(incoming, ignore_errors=True)
    os.makedirs(incoming)
    sender = subprocess.Popen(
        exec_args(pod, args, "tar", "-C", args.remote, "-cf", "-", "-T", "-", stdin=True),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )
    receiver = subprocess.Popen(
        ["tar", "-C", incoming, "-xf", "-"], stdin=sender.stdout, stderr=subprocess.DEVNULL
    )
    sender.stdout.close()
    sender.stdin.write(("\n".join(names) + "\n").encode("utf-8"))
    sender.stdin.close()
    receiver.wait()
    sender.wait()
    return incoming


def fetch_media(pod, names, sizes, args):
    """Keep every file that arrived at its full size; the rest wait for the next attempt."""
    incoming = pull(pod, names, args)
    for name in names:
        source = os.path.join(incoming, name)
        if os.path.isfile(source) and os.path.getsize(source) == sizes[name]:
            dest = os.path.join(args.local, name)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            os.replace(source, dest)
    shutil.rmtree(incoming, ignore_errors=True)


def referenced(index):
    paths = set()
    for sample in index.get("samples", {}).values():
        if sample.get("target"):
            paths.add(sample["target"])
        paths.update((sample.get("proxy") or {}).values())
    for task in index.get("tasks", []):
        paths.update(item["video"] for item in task["items"].values())
    return paths


def fetch_index(pod, args):
    """Install the pod's index.json and index.html, unless the index names a missing video."""
    incoming = pull(pod, list(INDEX), args)
    try:
        with open(os.path.join(incoming, "index.json"), encoding="utf-8") as handle:
            index = json.load(handle)
        absent = [p for p in referenced(index) if not os.path.isfile(os.path.join(args.local, p))]
        if absent:
            return f"kept the previous index: the new one names {len(absent)} video(s) not here yet"
        for name in INDEX:
            os.replace(os.path.join(incoming, name), os.path.join(args.local, name))
        return None
    except (OSError, ValueError) as exc:
        return f"kept the previous index: {exc}"
    finally:
        shutil.rmtree(incoming, ignore_errors=True)


def sync_once(args):
    pod = find_pod(args.pod_pattern, args.namespace)
    remote = remote_files(pod, args)
    if "index.json" not in remote:
        raise RuntimeError(f"{pod}:{args.remote} has no index.json yet")
    wanted = missing(remote, args.local)
    total = len(wanted)
    for _ in range(ATTEMPTS):
        if not wanted:
            break
        for batch in batches(wanted, remote):
            fetch_media(pod, batch, remote, args)
        wanted = missing(remote, args.local)
    problem = fetch_index(pod, args)
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(
        f"[eval-board-sync] {stamp} {pod}: fetched {total - len(wanted)}/{total} new file(s)"
        + (f"; {problem}" if problem else ""),
        flush=True,
    )
    if wanted:
        raise RuntimeError(f"{len(wanted)} file(s) still missing after {ATTEMPTS} attempts")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--pod-pattern", required=True, help="regex; the newest match is used")
    parser.add_argument("--remote", required=True, help="board directory on the pod")
    parser.add_argument("--local", required=True, help="board directory on this host")
    parser.add_argument("--namespace", default="ultron-ls-gcp-aw")
    parser.add_argument("--container", default="node")
    parser.add_argument(
        "--interval", type=float, default=0, help="sync every this many seconds instead of once"
    )
    args = parser.parse_args()
    os.makedirs(args.local, exist_ok=True)
    while True:
        try:
            sync_once(args)
        except (RuntimeError, OSError, subprocess.CalledProcessError) as exc:
            print(f"[eval-board-sync] {exc}", file=sys.stderr, flush=True)
            if not args.interval:
                return 1
        if not args.interval:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
