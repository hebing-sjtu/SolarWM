"""Rubric v1.1 reward: Gemini ranks every seed of one input in a single call, per dimension, with gates.

Each call sees the first frame (<Picture 1>), the character sheet (<Picture 2>, when the input has
one), the proxy video and the G samples under shuffled letters. For every sample and dimension
it writes the evidence first and then an anchored 0-4 score; then it ranks the samples per
dimension, ties allowed. ``aggregate`` turns ranks into a reward in [0, 1]: a weighted mean of
rank scores, with any sample failing a gate (proxy adherence or collapse <= 1) put below every
sample that passes. Each group is judged under ``--orders`` letter orders so the spread between
orders measures judge noise.

Runs on the Mac through the RL judge account only:

    rl-judge python tools/rl_rubric/judge.py judge --groups ~/rl1/judge --out ~/rl1/rubric_v11.jsonl --orders 2
    rl-judge python tools/rl_rubric/judge.py summary --results ~/rl1/rubric_v11.jsonl

The group folders come from ``prep.py``; ``label.py agree`` compares the rewards with human labels.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import random
import string
import sys
import time
from pathlib import Path

import rl_judge

RUBRIC_ID = "rubric-v1.1"
GATES = ("proxy_adherence", "collapse")
WEIGHTED = {"interaction": 0.30, "artifacts": 0.20, "identity": 0.15, "first_frame_look": 0.15,
            "proxy_residue": 0.10, "aesthetic": 0.10}
DIMENSIONS = GATES + tuple(WEIGHTED)
GATE_FAIL = 1

SYSTEM = """You are the reward judge for GameWM, a game engine that hands each event to a video model as a \
coarse proxy video. The engine owns positions, camera, timing and outcomes; the video model only decides how \
the scene looks: the characters' limbs and contact, effects, the target's reaction, and the appearance of the \
whole scene. You compare several samples that the same model generated from the same inputs with different \
random seeds, and you must separate them: small differences matter, because only the ordering within the group \
is used for training. Judge only what is visible. Never reward a sample for being static, blurry or empty."""

ANCHORS = """Score every sample on every dimension from 0 to 4 with these anchors, then rank.

proxy_adherence (gate). Does the sample keep what the proxy fixes: camera path and framing, scene layout, where \
the protagonist and the other subjects are and how they move, and when the event happens?
  4 all of it matches; 3 small deviations a player would not notice; 2 one clear deviation (a subject misplaced, \
timing off by about half a second, a camera drift); 1 major deviation (wrong camera path, a subject missing or \
added, the event at the wrong time); 0 unrelated to the proxy.
collapse (gate). Is the video intact?
  4 intact; 3 one brief minor glitch; 2 a visible local breakdown shorter than half a second; 1 severe melting, \
body fusion, mosaic or a cut for a noticeable part of the clip; 0 the clip is broken.
interaction. Does the event of the caption happen at the proxy's timing, completely and believably: limbs, hand \
contact, door, weapon or vehicle use, the target's reaction? For drive-only clips judge the vehicle's motion and \
the driver.
  4 complete and convincing; 3 complete with a minor implausibility; 2 it happens but contact or reaction is \
unconvincing; 1 only suggested; 0 does not happen.
artifacts. Hands, limbs, faces, vehicle shapes, texture swimming, flicker, objects popping in or out.
  4 none seen; 3 one small transient artifact; 2 several small or one clear artifact; 1 frequent artifacts; \
0 pervasive.
identity (only when <Picture 2> is given, otherwise null). Does the protagonist match the character sheet \
(silhouette, outfit, colours, hair, accessories) in every frame? The sheet wins over the first frame if they \
disagree.
  4 same character throughout; 3 matches with a minor detail off; 2 recognisable but a clear part differs or \
drifts; 1 mostly a different character; 0 unrelated.
first_frame_look. Does the whole clip keep the world that <Picture 1> sets: art style, palette, lighting, scene \
content, without drifting over time?
  4 consistent throughout; 3 slight drift; 2 clear drift in one aspect; 1 the look changes substantially; \
0 unrelated to <Picture 1>.
proxy_residue. Does proxy appearance leak into the output: capsule or primitive bodies, grey untextured \
surfaces, depth-gradient colours, flat semantic colour patches, a character left unrendered?
  4 no leakage; 3 a trace; 2 a visible patch or a partly unrendered character; 1 large proxy-looking regions; \
0 the output looks like the proxy.
aesthetic. Sharpness, lighting, composition: does it read as a polished game frame in the requested style?
  4 polished; 3 good; 2 acceptable; 1 poor; 0 very poor.

Procedure. First watch every sample against the proxy and list, per sample, what distinguishes it from the \
others: concrete observations with timestamps (e.g. "2.4s: the victim's left arm passes through his torso", \
"from 3.0s a burning object appears behind the trash can"). Then score each dimension from that dimension's own \
evidence. The dimensions measure different things and their orders usually differ: a sample can have the best \
interaction and the worst background artifacts. Never copy one order into every dimension; evidence such as \
"stable throughout" or "good" without a time or a visible detail is not evidence.

Ranking: for every dimension list the samples from best to worst as groups of tied letters, e.g. \
[["C"],["A","F"],["B"],...]. Every letter appears exactly once per dimension. Tie only when you cannot see a \
difference; scores may tie while the ranking still separates samples. For identity without <Picture 2> return \
an empty ranking. Finally rank the samples overall as a player would prefer them, with the same tie format."""


def schema(labels: list[str]) -> dict:
    dim = {"type": "OBJECT", "properties": {"evidence": {"type": "STRING"},
                                            "score": {"type": "INTEGER", "nullable": True}},
           "required": ["evidence", "score"], "propertyOrdering": ["evidence", "score"]}
    ranking = {"type": "ARRAY", "items": {"type": "ARRAY", "items": {"type": "STRING", "enum": labels}}}
    return {
        "type": "OBJECT",
        "properties": {
            "proxy_summary": {"type": "STRING"},
            "samples": {"type": "ARRAY", "items": {
                "type": "OBJECT",
                "properties": {"label": {"type": "STRING", "enum": labels},
                               "observations": {"type": "ARRAY", "items": {"type": "STRING"}},
                               **{d: dim for d in DIMENSIONS}},
                "required": ["label", "observations", *DIMENSIONS],
                "propertyOrdering": ["label", "observations", *DIMENSIONS]}},
            "rankings": {"type": "OBJECT", "properties": {d: ranking for d in DIMENSIONS},
                         "required": list(DIMENSIONS), "propertyOrdering": list(DIMENSIONS)},
            "overall": ranking,
        },
        "required": ["proxy_summary", "samples", "rankings", "overall"],
        "propertyOrdering": ["proxy_summary", "samples", "rankings", "overall"],
    }


def request(group: Path, meta: dict, order: list[int], fps: float) -> tuple[list[dict], list[str]]:
    labels = list(string.ascii_uppercase[:len(order)])
    sheet = meta["sheet"] and (group / "sheet.jpg").exists()
    intro = (f"Inputs of the generation. Proxy modality: {meta['proxy_kind']}. Caption (the event and the look):\n"
             f"{meta['prompt']}\n"
             + ("<Picture 2> is the protagonist's character sheet (front / side / back).\n" if sheet else
                "No character sheet was given; return null scores and an empty ranking for identity.\n"))
    parts = [{"text": intro}, {"text": "<Picture 1>, the first frame:"},
             rl_judge.media(group / "first.jpg", "image/jpeg")]
    if sheet:
        parts += [{"text": "<Picture 2>, the character sheet:"}, rl_judge.media(group / "sheet.jpg", "image/jpeg")]
    parts += [{"text": "The proxy video:"}, rl_judge.media(group / "proxy.mp4", "video/mp4", fps=fps)]
    for label, seed in zip(labels, order):
        parts += [{"text": f"Sample {label}:"},
                  rl_judge.media(group / "samples" / f"s{seed:03d}.mp4", "video/mp4", fps=fps)]
    parts.append({"text": ANCHORS})
    return parts, labels


def tie_ranks(groups: list[list[str]], labels: list[str]) -> dict[str, float] | None:
    """Mean 0-based position of each label; None unless every label appears exactly once."""
    flat = [label for tied in groups for label in tied]
    if sorted(flat) != sorted(labels):
        return None
    ranks, position = {}, 0
    for tied in groups:
        for label in tied:
            ranks[label] = position + (len(tied) - 1) / 2
        position += len(tied)
    return ranks


def parse(reply: dict, labels: list[str], seeds: list[int], sheet: bool) -> dict:
    data = json.loads(rl_judge.text(reply))
    by_label = dict(zip(labels, seeds))
    scores = {by_label[s["label"]]: {d: s[d]["score"] for d in DIMENSIONS} for s in data["samples"]}
    evidence = {by_label[s["label"]]: {"observations": s.get("observations", []),
                                       **{d: s[d]["evidence"] for d in DIMENSIONS}} for s in data["samples"]}
    ranks, bad = {}, []
    for dim in DIMENSIONS + ("overall",):
        if dim == "identity" and not sheet:
            continue
        groups = data["overall"] if dim == "overall" else data["rankings"][dim]
        r = tie_ranks(groups, labels)
        if r is None:
            bad.append(dim)
        else:
            ranks[dim] = {by_label[label]: value for label, value in r.items()}
    if sorted(scores) != sorted(seeds):
        bad.append("samples")
    return {"proxy_summary": data["proxy_summary"], "scores": scores, "evidence": evidence, "ranks": ranks,
            "invalid": bad}


def aggregate(call: dict, seeds: list[int], sheet: bool) -> dict[int, dict]:
    """Per seed: gate flag, rank-based reward in [0, 1] and score-based reward in [0, 1]."""
    weights = {d: w for d, w in WEIGHTED.items() if sheet or d != "identity"}
    total = sum(weights.values())
    span = max(len(seeds) - 1, 1)
    out = {}
    for seed in seeds:
        sc = call["scores"][seed]
        gate = all(sc[g] is not None and sc[g] > GATE_FAIL for g in GATES)
        rank = sum(w * (1 - call["ranks"][d][seed] / span) for d, w in weights.items() if d in call["ranks"]) / total
        absolute = sum(w * (sc[d] or 0) / 4 for d, w in weights.items()) / total
        out[seed] = {"gate": gate, "rank_reward": rank, "abs_reward": absolute,
                     "reward": rank if gate else rank - 1.0,
                     "overall_rank": call["ranks"].get("overall", {}).get(seed)}
    return out


def judge_group(group: Path, order_index: int, args: argparse.Namespace) -> dict:
    meta = json.loads((group / "meta.json").read_text())
    seeds = list(meta["seeds"])
    order = seeds[:]
    random.Random(f"{RUBRIC_ID}:{meta['group']}:{order_index}").shuffle(order)
    sheet = bool(meta["sheet"])
    parts, labels = request(group, meta, order, args.fps)
    config = {"responseMimeType": "application/json", "responseSchema": schema(labels),
              "temperature": args.temperature, "maxOutputTokens": 16384,
              "thinkingConfig": {"thinkingLevel": args.thinking}}
    last = None
    for attempt in range(args.retries + 1):
        started = time.time()
        try:
            reply = rl_judge.generate(parts, system=SYSTEM, config=config, timeout=600,
                                      tag=f"{RUBRIC_ID} {meta['group']} o{order_index}")
            call = parse(reply, labels, order, sheet)
        except (RuntimeError, ValueError, KeyError) as error:
            if "budget" in str(error):
                raise
            last = f"{type(error).__name__}: {str(error)[:200]}"
            continue
        if call["invalid"] and attempt < args.retries:
            last = f"invalid {call['invalid']}"
            continue
        usage = reply.get("usageMetadata", {})
        return {"rubric": RUBRIC_ID, "group": meta["group"], "name": meta["name"], "order_index": order_index,
                "order": order, "labels": labels, "sheet": sheet, "seconds": round(time.time() - started, 1),
                "usage": usage, **call,
                "rewards": {str(k): v for k, v in aggregate(call, seeds, sheet).items()} if not call["invalid"]
                else None}
    return {"rubric": RUBRIC_ID, "group": meta["group"], "order_index": order_index, "error": last}


def judge(args: argparse.Namespace) -> None:
    root = Path(args.groups).expanduser()
    out = Path(args.out).expanduser()
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            row = json.loads(line)
            if "error" not in row:
                done.add((row["group"], row["order_index"]))
    groups = sorted(p for p in root.iterdir() if (p / "meta.json").exists())
    if args.only:
        groups = [g for g in groups if g.name in set(args.only)]
    todo = [(g, k) for g in groups for k in range(args.orders) if (g.name, k) not in done]
    print(f"[judge] {len(todo)} call(s) for {len(groups)} group(s); spent this month ${rl_judge.spent():.3f}",
          flush=True)
    with concurrent.futures.ThreadPoolExecutor(args.workers) as pool, out.open("a") as handle:
        futures = {pool.submit(judge_group, g, k, args): (g.name, k) for g, k in todo}
        for future in concurrent.futures.as_completed(futures):
            row = future.result()
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            status = row.get("error") or (f"{row['seconds']}s in={row['usage'].get('promptTokenCount')} "
                                          f"out={row['usage'].get('candidatesTokenCount')}+"
                                          f"{row['usage'].get('thoughtsTokenCount', 0)} invalid={row['invalid']}")
            print(f"[judge] {row['group']} o{row['order_index']}: {status}", flush=True)
    print(f"[judge] spent this month ${rl_judge.spent():.3f}", flush=True)


def kendall(a: dict, b: dict) -> float | None:
    keys = sorted(set(a) & set(b))
    concordant = discordant = 0
    for i, x in enumerate(keys):
        for y in keys[i + 1:]:
            s = (a[x] - a[y]) * (b[x] - b[y])
            concordant += s > 0
            discordant += s < 0
    pairs = concordant + discordant
    return (concordant - discordant) / pairs if pairs else None


def summary(args: argparse.Namespace) -> None:
    rows = [json.loads(line) for line in Path(args.results).expanduser().read_text().splitlines()]
    by_group: dict = {}
    for row in rows:
        if row.get("rewards"):
            by_group.setdefault(row["group"], {})[row["order_index"]] = row
    taus, spread = [], []
    for group, calls in sorted(by_group.items()):
        rewards = [{int(k): v["reward"] for k, v in c["rewards"].items()} for c in calls.values()]
        tau = kendall(rewards[0], rewards[1]) if len(rewards) > 1 else None
        if tau is not None:
            taus.append(tau)
        mean = {s: sum(r[s] for r in rewards) / len(rewards) for s in rewards[0]}
        gates = {s: all(c["rewards"][str(s)]["gate"] for c in calls.values()) for s in mean}
        spread.append(max(mean.values()) - min(mean.values()))
        best = sorted(mean, key=mean.get, reverse=True)
        print(f"{group} tau={tau if tau is None else round(tau, 2)} gate_fail={sum(not g for g in gates.values())} "
              f"order={best} reward={[round(mean[s], 2) for s in best]}")
    if taus:
        print(f"mean Kendall tau between orders {sum(taus) / len(taus):.2f} over {len(taus)} groups; "
              f"mean reward spread {sum(spread) / len(spread):.2f}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="command", required=True)
    j = sub.add_parser("judge")
    j.add_argument("--groups", required=True, help="directory of prep.py group folders")
    j.add_argument("--out", required=True)
    j.add_argument("--orders", type=int, default=2)
    j.add_argument("--only", nargs="*")
    j.add_argument("--fps", type=float, default=4.0)
    j.add_argument("--thinking", default="MEDIUM")
    j.add_argument("--temperature", type=float, default=0.0)
    j.add_argument("--workers", type=int, default=6)
    j.add_argument("--retries", type=int, default=1)
    s = sub.add_parser("summary")
    s.add_argument("--results", required=True)
    args = p.parse_args()
    {"judge": judge, "summary": summary}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
