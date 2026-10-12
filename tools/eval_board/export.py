"""Gather finished proxy-loop eval tasks into one static board for side-by-side viewing.

Every eval task (loop / run / step / ablation) writes its samples under its own directory and
logs them to its own W&B run, so the same held-out sample from two runs never sits side by side.
This exports every finished task into one folder that any static file server can host:

  OUT/index.html   the board page (board.html next to this file)
  OUT/index.json   tasks, samples and per-sample scores
  OUT/media/*.mp4  browser-sized H.264, named by the source digest the eval manifest records

Media files are encoded once and never rewritten, so reruns only encode new tasks and a mirror can
copy by name. Scores are PSNR/SSIM of generated.mp4 against target.mp4 over all frames and over
frames >= --skip-frames, read from `<loop>-quality/<run>__<step>__<ablation>.json` beside the loop
when present and computed and written there otherwise (the format r1_quality.py writes).

Directory scans use listings rather than path lookups: the /data gcsfuse mount can keep stale
negative lookups for files that other nodes wrote later.

  python tools/eval_board/export.py --loops-root /data/binghe/h3_proxy/evals/loops \\
      --loop gta0930-omni-r1 --loop gta0930-omni-r2 \\
      --heldout /data/binghe/h3_proxy/manifests/gta_record_0930_omni_heldout32.jsonl \\
      --out /workspace/eval-board --label gta0930-r1-e1-unaligned="E1 unaligned" --watch 600
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEP = re.compile(r"^step-(\d+)$")
SLOT = re.compile(r"^slot-\d+$")
VIDEOS = ("generated.mp4", "proxy.mp4", "target.mp4")
FPS = 24


def listing(path: Path) -> set[str]:
    try:
        return set(os.listdir(path))
    except (FileNotFoundError, NotADirectoryError):
        return set()


def finished_tasks(loops_root: Path, loops: list[str]) -> list[tuple[str, str, str, str]]:
    tasks = []
    for loop in loops:
        root = loops_root / loop
        for run in sorted(listing(root)):
            for step in sorted(name for name in listing(root / run) if STEP.match(name)):
                for ablation in sorted(listing(root / run / step)):
                    if "DONE.json" in listing(root / run / step / ablation):
                        tasks.append((loop, run, step, ablation))
    return tasks


def task_slots(task_dir: Path) -> list[Path]:
    inference = task_dir / "proxy-inference"
    slots = []
    for wave in sorted(listing(inference)):
        for rank in sorted(listing(inference / wave)):
            for slot in sorted(listing(inference / wave / rank)):
                if SLOT.match(slot):
                    slots.append(inference / wave / rank / slot)
    return slots


def per_frame(generated: Path, target: Path, metric: str) -> list[float]:
    with tempfile.NamedTemporaryFile(suffix=".log") as stats:
        subprocess.run(
            [
                "ffmpeg",
                "-loglevel",
                "error",
                "-i",
                str(generated),
                "-i",
                str(target),
                "-lavfi",
                f"[0:v][1:v]{metric}=stats_file={stats.name}",
                "-f",
                "null",
                "-",
            ],
            check=True,
        )
        text = Path(stats.name).read_text()
    pattern = r"psnr_avg:(\S+)" if metric == "psnr" else r"All:(\S+)"
    return [100.0 if match == "inf" else float(match) for match in re.findall(pattern, text)]


def score_slot(slot: Path, skip_frames: int) -> dict:
    manifest = json.loads((slot / "manifest.json").read_text())
    psnr = per_frame(slot / "generated.mp4", slot / "target.mp4", "psnr")
    ssim = per_frame(slot / "generated.mp4", slot / "target.mp4", "ssim")
    return {
        "sample_id": manifest["case"]["sample_id"],
        "frames": len(psnr),
        "psnr_all": statistics.mean(psnr),
        "psnr_late": statistics.mean(psnr[skip_frames:]),
        "ssim_all": statistics.mean(ssim),
        "ssim_late": statistics.mean(ssim[skip_frames:]),
    }


def task_scores(
    loops_root: Path,
    task: tuple[str, str, str, str],
    slots: list[Path],
    skip_frames: int,
    pool: ThreadPoolExecutor,
) -> dict[str, dict]:
    loop, run, step, ablation = task
    quality = loops_root / f"{loop}-quality"
    cache = quality / f"{run}__{step}__{ablation}.json"
    if cache.name in listing(quality):
        samples = json.loads(cache.read_text())["samples"]
    else:
        samples = list(pool.map(lambda slot: score_slot(slot, skip_frames), slots))
        quality.mkdir(parents=True, exist_ok=True)
        payload = {"run": run, "step": step, "ablation": ablation, "samples": samples}
        cache.write_text(json.dumps(payload, indent=1))
        print(f"[eval-board] scored {loop}/{run}/{step}/{ablation}", flush=True)
    return {sample["sample_id"]: sample for sample in samples}


def encode(source: Path, dest: Path, crf: int) -> None:
    partial = dest.parent.parent / ".partial" / dest.name
    partial.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-threads",
            "4",
            str(partial),
        ],
        check=True,
    )
    partial.replace(dest)


def heldout_samples(path: Path) -> dict[str, dict]:
    samples = {}
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            samples[row["name"]] = {
                "prompt": row.get("prompt", ""),
                "modality": row.get("proxy_modality"),
                "kind": row.get("target_kind"),
                "tag": row.get("tag"),
            }
    return samples


def parse_labels(pairs: list[str]) -> dict[str, str]:
    labels = {}
    for pair in pairs:
        key, sep, text = pair.partition("=")
        if not sep or not key or not text:
            raise SystemExit(f"--label wants RUN=TEXT or RUN/step-NNNNNN=TEXT, got {pair!r}")
        labels[key] = text
    return labels


def write_atomic(path: Path, data: bytes) -> None:
    partial = path.with_name(f".{path.name}.partial")
    partial.write_bytes(data)
    partial.replace(path)


def export(args: argparse.Namespace) -> dict:
    out = args.out
    media = out / "media"
    media.mkdir(parents=True, exist_ok=True)
    labels = parse_labels(args.label)
    samples = heldout_samples(args.heldout)
    have = listing(media)
    tasks, jobs, failed, frames = [], {}, [], None
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for task in finished_tasks(args.loops_root, args.loop):
            loop, run, step, ablation = task
            task_dir = args.loops_root.joinpath(*task)
            slots = task_slots(task_dir)
            done = json.loads((task_dir / "DONE.json").read_text())
            try:
                scores = task_scores(args.loops_root, task, slots, args.skip_frames, pool)
            except (subprocess.CalledProcessError, OSError, KeyError, ValueError) as exc:
                failed.append(f"{'/'.join(task)}: scoring failed: {exc}")
                continue
            items = {}
            for slot in slots:
                manifest = json.loads((slot / "manifest.json").read_text())
                sample_id = manifest["case"]["sample_id"]
                digests = {Path(a["path"]).name: a["digest"] for a in manifest["artifacts"]}
                if any(name not in digests for name in VIDEOS):
                    failed.append(f"{slot}: manifest lacks one of {VIDEOS}")
                    continue
                names = {name: f"{digests[name][:24]}.mp4" for name in VIDEOS}
                for video, name in names.items():
                    if name not in have:
                        jobs.setdefault(name, slot / video)
                sample = samples.setdefault(sample_id, {"prompt": manifest["case"].get("prompt")})
                sample.setdefault("target", f"media/{names['target.mp4']}")
                sample.setdefault("proxy", {}).setdefault(ablation, f"media/{names['proxy.mp4']}")
                item = {"video": f"media/{names['generated.mp4']}"}
                score = scores.get(sample_id, {})
                frames = frames or score.get("frames")
                item.update(
                    {
                        k: score[k]
                        for k in ("psnr_all", "psnr_late", "ssim_all", "ssim_late")
                        if k in score
                    }
                )
                items[sample_id] = item
            number = int(STEP.match(step).group(1))
            run_label = labels.get(run, run)
            default = f"{run_label} · base" if number == 0 else f"{run_label} · step {number}"
            tasks.append(
                {
                    "id": "/".join(task),
                    "loop": loop,
                    "run": run,
                    "step": number,
                    "ablation": ablation,
                    "run_label": run_label,
                    "label": labels.get(f"{run}/{step}", default),
                    "weights": done.get("weight_source", "live"),
                    "items": items,
                }
            )

        def encode_job(entry: tuple[str, Path]) -> str | None:
            name, source = entry
            try:
                encode(source, media / name, args.crf)
            except (subprocess.CalledProcessError, OSError) as exc:
                return f"{source}: encode failed: {exc}"
            return None

        if jobs:
            print(f"[eval-board] encoding {len(jobs)} video(s)", flush=True)
        failed.extend(error for error in pool.map(encode_job, jobs.items()) if error)

    have = listing(media)
    for task in tasks:
        task["items"] = {
            sample_id: item
            for sample_id, item in task["items"].items()
            if item["video"].removeprefix("media/") in have
        }
    index = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "loops": list(args.loop),
        "fps": FPS,
        "frames": frames,
        "skip_frames": args.skip_frames,
        "samples": samples,
        "tasks": tasks,
    }
    write_atomic(out / "index.json", json.dumps(index, ensure_ascii=False).encode("utf-8"))
    write_atomic(out / "index.html", (HERE / "board.html").read_bytes())
    for error in failed:
        print(f"[eval-board] {error}", flush=True)
    print(
        f"[eval-board] {index['generated_at']}: {len(tasks)} task(s), "
        f"{len(jobs)} new video(s), {len(failed)} problem(s)",
        flush=True,
    )
    return index


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--loops-root", type=Path, required=True)
    parser.add_argument("--loop", action="append", required=True, help="loop directory name")
    parser.add_argument("--heldout", type=Path, required=True, help="held-out manifest jsonl")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--label", action="append", default=[], help="RUN=TEXT or RUN/step-NNNNNN=TEXT display name"
    )
    parser.add_argument("--skip-frames", type=int, default=24)
    parser.add_argument("--crf", type=int, default=23)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--watch", type=float, default=0, help="re-export every this many seconds instead of once"
    )
    args = parser.parse_args(argv)
    if not shutil.which("ffmpeg"):
        raise SystemExit("ffmpeg is not on PATH")
    if not args.watch:
        export(args)
        return 0
    while True:
        try:
            export(args)
        except Exception as exc:
            print(f"[eval-board] export pass failed: {exc!r}", flush=True)
        time.sleep(args.watch)


if __name__ == "__main__":
    sys.exit(main())
