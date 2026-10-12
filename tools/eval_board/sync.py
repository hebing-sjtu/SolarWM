#!/usr/bin/env python3
"""Mirror an exported eval board from a BCS pod to this host, fetching only what is missing.

Media files are content-addressed and never rewritten, so a name with the same size is already
here. index.json and index.html change on every export; they are fetched after the media they
point to, so the page never names a video this host lacks. Pods have no rsync, so files travel
as a tar stream over `kubectl exec`. Nothing here is ever deleted.

Standard library only and Python 3.6 compatible.

  python3 sync.py --pod-pattern '^bingghhe.*-node-0-3-' --remote /workspace/eval-board \\
      --local /data/eval-board --interval 300
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import time

INDEX = ("index.json", "index.html")
BATCH = 200


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


def remote_files(pod, args):
    script = f"cd {args.remote} && find . -path './.*' -prune -o -type f -printf '%P\\t%s\\n'"
    out = subprocess.run(
        [
            "kubectl",
            "-n",
            args.namespace,
            "exec",
            pod,
            "-c",
            args.container,
            "--",
            "sh",
            "-c",
            script,
        ],
        stdout=subprocess.PIPE,
        check=True,
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


def fetch(pod, names, args):
    """Copy names from the pod into the local board; each lands whole or not at all."""
    incoming = os.path.join(args.local, ".incoming")
    shutil.rmtree(incoming, ignore_errors=True)
    os.makedirs(incoming)
    sender = subprocess.Popen(
        [
            "kubectl",
            "-n",
            args.namespace,
            "exec",
            "-i",
            pod,
            "-c",
            args.container,
            "--",
            "tar",
            "-C",
            args.remote,
            "-cf",
            "-",
            "-T",
            "-",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )
    receiver = subprocess.Popen(["tar", "-C", incoming, "-xf", "-"], stdin=sender.stdout)
    sender.stdout.close()
    sender.stdin.write(("\n".join(names) + "\n").encode("utf-8"))
    sender.stdin.close()
    if receiver.wait() or sender.wait():
        raise RuntimeError(f"tar over kubectl exec failed for {len(names)} file(s)")
    for name in names:
        source = os.path.join(incoming, name)
        dest = os.path.join(args.local, name)
        os.makedirs(os.path.dirname(dest) or args.local, exist_ok=True)
        os.replace(source, dest)
    shutil.rmtree(incoming, ignore_errors=True)


def sync_once(args):
    pod = find_pod(args.pod_pattern, args.namespace)
    remote = remote_files(pod, args)
    if "index.json" not in remote:
        raise RuntimeError(f"{pod}:{args.remote} has no index.json yet")
    wanted = missing(remote, args.local)
    for start in range(0, len(wanted), BATCH):
        fetch(pod, wanted[start : start + BATCH], args)
    fetch(pod, [name for name in INDEX if name in remote], args)
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[eval-board-sync] {stamp} {pod}: fetched {len(wanted)} file(s)", flush=True)


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
