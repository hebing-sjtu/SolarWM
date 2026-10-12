"""Write a markdown comparison of eval-board tasks: group means and paired per-sample deltas.

Tasks are named by alias (`--task E1@1000=gta0930-omni-r1/<run>/step-001000/correct`), and each
`--pair A:B` reports A minus B over the held-out samples both tasks scored, with its standard error
and how many samples improved. Groups are all samples, each target kind and each proxy modality.
Aliases missing from the index are listed rather than failing, so the report can be rewritten as
tasks finish; `--wait` repeats until every alias is present.

  python tools/eval_board/report.py --index /workspace/eval-board/index.json \\
      --task base=... --task E1@1000=... --pair E1@1000:base --out reports/r3.md
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

METRICS = (("psnr_late", 2, "PSNR, frames >= 24 (dB)"), ("ssim_late", 3, "SSIM, frames >= 24"))
KINDS = ("color", "high")
MODALITIES = ("duv", "depth", "semantic", "style")


def groups(samples: dict) -> list[tuple[str, set[str]]]:
    out = [("ALL", set(samples))]
    out += [(k, {s for s, v in samples.items() if v.get("kind") == k}) for k in KINDS]
    out += [(m, {s for s, v in samples.items() if v.get("modality") == m}) for m in MODALITIES]
    return out


def parse_pairs(entries: list[str], option: str) -> list[tuple[str, str]]:
    pairs = []
    for entry in entries:
        left, sep, right = entry.partition("=" if option == "--task" else ":")
        if not sep or not left or not right:
            raise SystemExit(f"{option} got {entry!r}")
        pairs.append((left, right))
    return pairs


def render(index: dict, tasks: dict[str, str], pairs: list[tuple[str, str]], title: str) -> str:
    by_id = {task["id"]: task for task in index["tasks"]}
    found = {alias: by_id[task_id]["items"] for alias, task_id in tasks.items() if task_id in by_id}
    missing = [alias for alias in tasks if alias not in found]
    split = groups(index["samples"])
    lines = [
        f"# {title}",
        "",
        f"Written {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC from the "
        f"board export of {index['generated_at']}.",
    ]
    if missing:
        lines += ["", "Not evaluated yet: " + ", ".join(f"`{alias}`" for alias in missing) + "."]
    header = "| | " + " | ".join(name for name, _ in split) + " |"
    rule = "|---|" + "---:|" * len(split)
    for metric, digits, label in METRICS:
        lines += ["", f"## Means: {label}", "", header, rule]
        for alias, items in found.items():
            cells = []
            for _, members in split:
                values = [items[s][metric] for s in members if s in items and metric in items[s]]
                cells.append(f"{statistics.mean(values):.{digits}f}" if values else "-")
            lines.append(f"| `{alias}` | " + " | ".join(cells) + " |")
        rows = []
        for left, right in pairs:
            if left not in found or right not in found:
                continue
            cells = []
            for _, members in split:
                shared = [
                    s
                    for s in members
                    if metric in found[left].get(s, {}) and metric in found[right].get(s, {})
                ]
                deltas = [found[left][s][metric] - found[right][s][metric] for s in shared]
                if not deltas:
                    cells.append("-")
                    continue
                error = (
                    statistics.stdev(deltas) / math.sqrt(len(deltas)) if len(deltas) > 1 else 0.0
                )
                better = sum(delta > 0 for delta in deltas)
                cells.append(
                    f"{statistics.mean(deltas):+.{digits}f} ± {error:.{digits}f} "
                    f"({better}/{len(deltas)})"
                )
            rows.append(f"| `{left}` - `{right}` | " + " | ".join(cells) + " |")
        if rows:
            lines += [
                "",
                f"## Paired deltas: {label}",
                "",
                "mean ± standard error (samples where the left side is higher / shared)",
                "",
                header,
                rule,
                *rows,
            ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--index", type=Path, required=True, help="eval-board index.json")
    parser.add_argument("--task", action="append", default=[], help="ALIAS=LOOP/RUN/STEP/ABLATION")
    parser.add_argument("--pair", action="append", default=[], help="A:B reports A minus B")
    parser.add_argument("--title", default="Eval comparison")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--copy-to",
        type=Path,
        action="append",
        default=[],
        help="also write the report here (whole-file copy)",
    )
    parser.add_argument(
        "--wait",
        type=float,
        default=0,
        help="rewrite every this many seconds until every task is present",
    )
    args = parser.parse_args(argv)
    tasks = dict(parse_pairs(args.task, "--task"))
    pairs = parse_pairs(args.pair, "--pair")
    for left, right in pairs:
        if left not in tasks or right not in tasks:
            raise SystemExit(f"--pair {left}:{right} names an alias no --task defines")
    while True:
        index = json.loads(args.index.read_text())
        text = render(index, tasks, pairs, args.title)
        for path in (args.out, *args.copy_to):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        ids = {task["id"] for task in index["tasks"]}
        done = all(task_id in ids for task_id in tasks.values())
        print(
            f"[report] {datetime.now(timezone.utc):%H:%M} wrote {args.out}; "
            f"{sum(t in ids for t in tasks.values())}/{len(tasks)} tasks present",
            flush=True,
        )
        if done or not args.wait:
            return 0
        time.sleep(args.wait)


if __name__ == "__main__":
    sys.exit(main())
