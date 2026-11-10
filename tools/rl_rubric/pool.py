"""Turn a rubric prompt pool into encode manifests, seed groups and serve jobs.

A pool (``pools/*.json``) names one realization per group. ``ranged`` groups are complete rows over
a source tree outside the dataset (``--shoot-root``): a capsule proxy and an H3 high render whose
first frame is the anchor. ``melee`` groups reuse a training row (``base``, looked up in
``--index``) with its caption rewritten; their caches are the base cache with fresh Qwen rows and
then the sheet put back, so the latents are the base's.

``build`` writes, under ``--work``::

    groups.json                every group with its full row, root, realization and cache path
    encode/full.jsonl          rows encoded from pixels (ranged)
    encode/text.jsonl          rows re-captioned on an existing cache (melee)
    encode/src/<name>.pt       links to the base caches under the new names, for --text-source

Then encode with FastVideo's ``encode_proxy_samples.py`` (flags as for the OC caches)::

    full:  --manifest encode/full.jsonl --root SHOOT_ROOT --output cache/full
    text:  --manifest encode/text.jsonl --root DATASET --text-only --text-source encode/src --output cache/text
    oc:    --manifest encode/text.jsonl --root DATASET --oc-refresh --text-source cache/text --output cache/oc

``vehicle`` groups swap the car of a training enter/exit row for another vehicle: an edited first frame
(``take_anchor``, under ``--assets``) and a rewritten caption, encoded from pixels, then the row's
protagonist sheet put back. Their vehicle sheet (``sheet``) only goes to the judge and the board::

    full:  --manifest encode/vehicle.jsonl --root DATASET --output cache/vfull
    oc:    --manifest encode/vehicle.jsonl --root DATASET --oc-refresh --text-source cache/vfull --output cache/oc

``enqueue`` drops one ``h3-proxy-serve-job.v1`` per group and seed into a serve spool; ``prep.py``
reads ``groups.json`` directly.

    python tools/rl_rubric/pool.py build --pool tools/rl_rubric/pools/p0b_realization.json \\
        --index /data/binghe/h3_proxy/manifests/gta_record_0930_omni_train_oc.jsonl \\
        --text-source /data/binghe/h3_proxy/cache/gta_record_0930_omni_704p_ref640 \\
        --dataset /data/binghe/datasets/gta_record_0930-v0003 --shoot-root /workspace/rl2/shoot --work /workspace/rl2
    python tools/rl_rubric/pool.py enqueue --work /workspace/rl2 --spool /workspace/rl2/spool --seeds 11-18
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

JOB_SCHEMA = "h3-proxy-serve-job.v1"


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    scratch = path.with_name(f".{path.name}.tmp")
    scratch.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    os.replace(scratch, path)


def build(args: argparse.Namespace) -> None:
    pool = json.loads(Path(args.pool).read_text(encoding="utf-8"))
    work = Path(args.work)
    index = {row["name"]: row for row in map(json.loads, Path(args.index).read_text().splitlines()) if row}
    links = work / "encode" / "src"
    links.mkdir(parents=True, exist_ok=True)
    full, text, vehicle, groups = [], [], [], []
    for spec in pool["groups"]:
        kind = spec["kind"]
        if kind == "ranged":
            name = f"{spec['group']}__{spec['seg']}__{spec['style']}"
            row = {"name": name, "id": spec["seg"], "target": spec["source"],
                   "proxy": str(Path(spec["source"]).parent / "04_白模胶囊_原动作.mp4"),
                   "target_style": spec["style"], **pool["ranged_row"], "prompt": spec["prompt"]}
            root, pt, event = args.shoot_root, work / "cache" / "full" / f"{name}.pt", pool["events"][spec["seg"]]
            full.append(row)
        elif kind == "melee":
            base = index[spec["base"]]
            if not base.get("oc"):
                raise SystemExit(f"{spec['group']}: base row {spec['base']} has no sheet")
            source = Path(args.text_source) / f"{spec['base']}.pt"
            if not source.is_file():
                raise SystemExit(f"{spec['group']}: no base cache {source}")
            name = f"{spec['group']}__{spec['base']}"
            row = {**base, "name": name, "prompt": spec["prompt"], "base": spec["base"]}
            link = links / f"{name}.pt"
            link.unlink(missing_ok=True)
            link.symlink_to(source)
            root, pt, event = args.dataset, work / "cache" / "oc" / f"{name}.pt", pool["events"]["melee"]
            text.append(row)
        elif kind == "vehicle":
            base = index[spec["base"]]
            name = f"{spec['group']}__{spec['base']}"
            row = {**base, "name": name, "prompt": spec["prompt"], "base": spec["base"],
                   "take_anchor": str(Path(args.assets) / spec["anchor"])}
            root, pt, event = args.dataset, work / "cache" / "oc" / f"{name}.pt", pool["events"][base["tag"]]
            vehicle.append(row)
        else:
            raise SystemExit(f"{spec['group']}: unknown kind {kind!r}")
        groups.append({"group": spec["group"], "kind": kind, "name": name, "root": str(root), "row": row,
                       "prompt": spec["prompt"], "realization": spec["realization"],
                       "realization_set": pool["sets"][kind], "event": event, "pt": str(pt),
                       **({"vehicle_sheet": str(Path(args.assets) / spec["sheet"])} if spec.get("sheet") else {})})
    for path, rows in ((work / "encode" / "full.jsonl", full), (work / "encode" / "text.jsonl", text),
                       (work / "encode" / "vehicle.jsonl", vehicle)):
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    write_json(work / "groups.json", groups)
    print(f"{pool['pool']}: {len(groups)} groups ({len(full)} full, {len(text)} re-captioned, "
          f"{len(vehicle)} vehicle swaps) -> {work}")


def seed_range(text: str) -> list[int]:
    first, _, last = text.partition("-")
    return list(range(int(first), int(last or first) + 1))


def enqueue(args: argparse.Namespace) -> None:
    groups = json.loads((Path(args.work) / "groups.json").read_text(encoding="utf-8"))
    queue = Path(args.spool) / "queue"
    queue.mkdir(parents=True, exist_ok=True)
    missing = [g["pt"] for g in groups if not Path(g["pt"]).is_file()]
    if missing:
        raise SystemExit(f"{len(missing)} cache(s) missing, e.g. {missing[0]}")
    written = 0
    for seed in seed_range(args.seeds):
        for g in groups:
            job_id = f"{g['group']}-s{seed:03d}-{g['row']['id']}"
            if any((Path(args.spool) / d / f"{job_id}.json").exists() for d in ("queue", "running", "done")):
                continue
            write_json(queue / f"{job_id}.json", {
                "schema": JOB_SCHEMA, "seed": seed, "pool": g["group"],
                "segments": [{"name": g["name"], "pt": g["pt"], "cwm_system": g["row"]["cwm_system"]}]})
            written += 1
    print(f"queued {written} job(s) in {queue}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--pool", required=True)
    b.add_argument("--index", required=True, help="manifest holding the melee base rows")
    b.add_argument("--text-source", default="", help="cache directory holding the melee base samples")
    b.add_argument("--dataset", required=True)
    b.add_argument("--shoot-root", default="")
    b.add_argument("--assets", default="", help="directory of the vehicle groups' edited first frames and sheets")
    b.add_argument("--work", required=True)
    e = sub.add_parser("enqueue")
    e.add_argument("--work", required=True)
    e.add_argument("--spool", required=True)
    e.add_argument("--seeds", default="11-18")
    args = p.parse_args()
    {"build": build, "enqueue": enqueue}[args.command](args)


if __name__ == "__main__":
    main()
