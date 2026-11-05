"""Judge media for every seed group whose takes are all done: proxy, first frame, sheet, samples.

A group is one cached input generated under several seeds by the H3 proxy serve workers
(``proxy_loop serve``; job ids ``<group>-sNNN-<seg>``). Every video is cut to the take length,
cropped to the 1280x704 aspect and scaled to 640x352 so the judge sees all streams at one size. The proxy is the human-readable
form of what the model was given: the low-poly render for ``style``, the depth video, the
coloured semantic video, or depth beside semantics for ``duv``. Only stdlib and ffmpeg are used,
so it runs on the pods; the judge itself runs on the Mac.

    python tools/rl_rubric/prep.py --spool /workspace/rl1/spool --groups /workspace/rl1/groups.json \\
        --manifests A=heldout32.jsonl B=heldout32_oc.jsonl C=heldout32_ocswap.jsonl \\
        --dataset /data/binghe/datasets/gta_record_0930-v0003 --out /workspace/rl1/judge [--watch 120]
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import time
from pathlib import Path

WIDTH, HEIGHT = 640, 352
FRAMES = 124
FIT = f"crop='min(iw,ih*{WIDTH}/{HEIGHT})':'min(ih,iw*{HEIGHT}/{WIDTH})',scale={WIDTH}:{HEIGHT}:flags=bicubic,setsar=1"


def crop_filter(crop: str | list | None) -> str:
    if not crop:
        return ""
    left, top, right, bottom = (int(v) for v in (json.loads(crop) if isinstance(crop, str) else crop))
    return f"crop={right - left}:{bottom - top}:{left}:{top},"


def ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", *args], check=True)


def video(src: Path, out: Path, crop: str | list | None = None) -> None:
    ffmpeg("-i", str(src), "-vf", f"fps=24,{crop_filter(crop)}{FIT}", "-frames:v", str(FRAMES), "-an",
           "-c:v", "libx264", "-crf", "24", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out))


def side_by_side(left: Path, right: Path, out: Path) -> None:
    ffmpeg("-i", str(left), "-i", str(right), "-filter_complex",
           f"[0:v]fps=24,{FIT}[a];[1:v]fps=24,{FIT}[b];[a][b]hstack=2", "-frames:v", str(FRAMES), "-an",
           "-c:v", "libx264", "-crf", "24", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out))


def first_frame(src: Path, out: Path, crop: str | list | None = None) -> None:
    ffmpeg("-i", str(src), "-vf", f"{crop_filter(crop)}{FIT}", "-frames:v", "1", "-q:v", "2", str(out))


def sheet(src: Path, out: Path) -> None:
    ffmpeg("-i", str(src), "-vf", "scale='min(1024,iw)':-2:flags=bicubic", "-q:v", "2", str(out))


def proxy(row: dict, dataset: Path, out: Path) -> str:
    seg = dataset / row["id"] / "proxy"
    modality = row["proxy_modality"]
    if modality == "style":
        video(dataset / row["proxy"], out, row.get("proxy_crop"))
        return f"low-poly re-render ({row.get('proxy_style', 'style')})"
    if modality == "depth":
        video(seg / "depth.mp4", out)
        return "depth video"
    if modality == "semantic":
        video(seg / "sem_rgb.mp4", out)
        return "coloured semantic video"
    side_by_side(seg / "depth.mp4", seg / "sem_rgb.mp4", out)
    return "packed depth + semantics, shown as depth (left) beside coloured semantics (right)"


def prepare(group: dict, row: dict, takes: dict[int, Path], dataset: Path, out: Path) -> None:
    scratch = out.with_name(f".{out.name}.tmp")
    shutil.rmtree(scratch, ignore_errors=True)
    (scratch / "samples").mkdir(parents=True)
    proxy_kind = proxy(row, dataset, scratch / "proxy.mp4")
    first_frame(dataset / row["target"], scratch / "first.jpg", row.get("target_crop"))
    has_sheet = bool(row.get("oc"))
    if has_sheet:
        sheet(dataset / row["oc"], scratch / "sheet.jpg")
    for seed, take in sorted(takes.items()):
        video(take, scratch / "samples" / f"s{seed:03d}.mp4")
    meta = {"group": group["group"], "name": row["name"], "seg": row["id"], "tag": row.get("tag"),
            "modality": row["proxy_modality"], "proxy_kind": proxy_kind, "target_kind": row["target_kind"],
            "target_style": row.get("target_style"), "sheet": has_sheet, "sheet_donor": row.get("oc_donor"),
            "prompt": group["prompt"], "seeds": sorted(takes)}
    (scratch / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    shutil.rmtree(out, ignore_errors=True)
    scratch.rename(out)


def done_takes(spool: Path, group: str) -> tuple[dict[int, Path], list[str]]:
    takes, errors = {}, []
    for result in sorted((spool / "done").glob(f"{group}-s*.json")):
        seed = int(re.match(rf"{group}-s(\d+)-", result.name).group(1))
        if "error" in json.loads(result.read_text()):
            errors.append(result.stem)
            continue
        take = spool / "out" / result.stem / "take.mp4"
        if take.exists():
            takes[seed] = take
    return takes, errors


def sweep(args: argparse.Namespace) -> int:
    groups = json.loads(Path(args.groups).read_text())
    manifests = {}
    for spec in args.manifests:
        tag, path = spec.split("=", 1)
        manifests[tag] = {row["name"]: row for row in map(json.loads, Path(path).read_text().splitlines()) if row}
    spool, dataset, out = Path(args.spool), Path(args.dataset), Path(args.out)
    pending = 0
    for group in groups:
        target = out / group["group"]
        if (target / "meta.json").exists():
            continue
        takes, errors = done_takes(spool, group["group"])
        if errors:
            print(f"[prep] {group['group']}: failed jobs {errors}", flush=True)
        if len(takes) + len(errors) < args.seeds or len(takes) < 2:
            pending += 1
            continue
        prepare(group, manifests[group["group"][0]][group["name"]], takes, dataset, target)
        print(f"[prep] {group['group']} {group['name']}: {len(takes)} samples", flush=True)
    return pending


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--spool", required=True)
    p.add_argument("--groups", required=True)
    p.add_argument("--manifests", nargs="+", required=True, help="TAG=manifest.jsonl, TAG the group letter")
    p.add_argument("--dataset", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seeds", type=int, default=8)
    p.add_argument("--watch", type=int, default=0, help="re-sweep every N seconds until every group is ready")
    args = p.parse_args()
    while True:
        pending = sweep(args)
        if not args.watch or pending == 0:
            print(f"[prep] {pending} group(s) pending", flush=True)
            return
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
