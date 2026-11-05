"""Online blind labeling page for seed groups, and the agreement of rubric rewards with its labels.

``build`` writes a page folder over the ``prep.py`` group folders, to be published under the eval
board root (``serve.py``): per group the first frame, the sheet, the proxy, and every seed under
a shuffled letter. The rater marks each sample good / ok / bad and picks the best one. Every
change is saved to the board server (``serve.py --answers``) under ``/answers/<name>.json`` and
kept in the browser as well, so a reload or a second device continues where the rater stopped.
The map from letters back to seeds goes to ``--key``, which must stay outside the published folder.
``agree`` turns the labels into within-group preferences (best over all others, good over ok over
bad) and reports, per reward, the share of those pairs it orders the same way, ties excluded.

    python tools/rl_rubric/label.py build --groups ~/rl1/judge --out ~/rl1/board/rl1 \\
        --key ~/rl1/board-key.json --answers-name rl1-rubric-labels
    python tools/rl_rubric/label.py agree --key ~/rl1/board-key.json \\
        --answers http://21.130.243.218:8768/answers/rl1-rubric-labels.json --results ~/rl1/rubric_v11.jsonl
"""

from __future__ import annotations

import argparse
import html
import json
import random
import shutil
import string
import urllib.request
from pathlib import Path

PAGE = """<!doctype html>
<html lang="zh"><head><meta charset="utf-8"><title>{title}</title>
<style>
body {{ font-family: -apple-system, "PingFang SC", sans-serif; margin: 16px auto; max-width: 1700px; color: #1a202c; }}
.group {{ border: 1px solid #cbd5e0; border-radius: 8px; padding: 12px; margin: 18px 0; }}
.group.done {{ border-color: #38a169; background: #f0fff4; }}
.inputs {{ display: grid; grid-template-columns: 1fr 1fr 2fr; gap: 8px; align-items: start; }}
.inputs img, .inputs video {{ width: 100%; }}
.samples {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; margin-top: 10px; }}
.sample {{ border: 2px solid transparent; border-radius: 6px; padding: 4px; }}
.sample.best {{ border-color: #d69e2e; }}
.sample video {{ width: 100%; background: #000; }}
.caption {{ font-size: 13px; color: #4a5568; margin: 6px 0; }}
button {{ font-size: 13px; padding: 3px 10px; margin: 2px; border-radius: 5px; border: 1px solid #a0aec0; background: #fff; cursor: pointer; }}
button.on {{ background: #2b6cb0; color: #fff; border-color: #2b6cb0; }}
button.best.on {{ background: #d69e2e; border-color: #d69e2e; }}
#bar {{ position: sticky; top: 0; background: #fff; padding: 8px 0; border-bottom: 1px solid #e2e8f0; z-index: 2; }}
#saved {{ margin-left: 12px; color: #718096; }}
#saved.error {{ color: #c53030; }}
small {{ color: #718096; }}
</style></head><body>
<div id="bar"><b>{title}</b>：每组 8 条样本来自同一模型、同一输入，只有随机种子不同。
每条标"好 / 一般 / 差"（按交互是否完成、有无瑕疵、身份、贴合 proxy、观感综合判断），再选出最好的一条。
已完成 <span id="count">0</span> / {total} 组。<span id="saved"></span>
<button onclick="exportAnswers()">导出备份</button></div>
{groups}
<script>
const NAME = "{answers_name}";
const URL_ = "/answers/" + NAME + ".json";
let answers = JSON.parse(localStorage.getItem(NAME) || "{{}}");
let pending = null;
function status(text, error) {{
  const s = document.getElementById("saved"); s.textContent = text; s.classList.toggle("error", !!error);
}}
function render() {{
  let done = 0;
  document.querySelectorAll(".group").forEach(g => {{
    const a = answers[g.dataset.id] || {{labels: {{}}}};
    const n = g.querySelectorAll(".sample").length;
    const complete = Object.keys(a.labels).length === n && a.best;
    g.classList.toggle("done", !!complete); if (complete) done++;
    g.querySelectorAll(".sample").forEach(s => {{
      const l = s.dataset.label;
      s.classList.toggle("best", a.best === l);
      s.querySelectorAll("button").forEach(b => b.classList.toggle("on",
        b.dataset.v === "best" ? a.best === l : a.labels[l] === b.dataset.v));
    }});
  }});
  document.getElementById("count").textContent = done;
}}
function save() {{
  clearTimeout(pending);
  pending = setTimeout(() => {{
    fetch(URL_, {{method: "POST", headers: {{"Content-Type": "application/json"}}, body: JSON.stringify(answers)}})
      .then(r => {{ if (!r.ok) throw new Error(r.status); status("已保存到服务器 " + new Date().toLocaleTimeString()); }})
      .catch(e => status("服务器保存失败（" + e.message + "），结果仍在本浏览器里", true));
  }}, 400);
}}
function mark(id, label, v) {{
  const a = answers[id] || (answers[id] = {{labels: {{}}}});
  if (v === "best") a.best = label; else a.labels[label] = v;
  a.updated = new Date().toISOString();
  localStorage.setItem(NAME, JSON.stringify(answers)); render(); save();
}}
function playAll(id) {{
  document.querySelectorAll(`.group[data-id="${{id}}"] video`).forEach(v => {{ v.currentTime = 0; v.play(); }});
}}
function exportAnswers() {{
  const blob = new Blob([JSON.stringify(answers, null, 1)], {{type: "application/json"}});
  const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = NAME + ".json"; a.click();
}}
fetch(URL_, {{cache: "no-store"}}).then(r => r.ok ? r.json() : {{}}).then(server => {{
  for (const [id, a] of Object.entries(server || {{}})) {{
    const local = answers[id];
    if (!local || (a.updated || "") >= (local.updated || "")) answers[id] = a;
  }}
  localStorage.setItem(NAME, JSON.stringify(answers)); render();
  status(Object.keys(server || {{}}).length ? "已载入服务器上的进度" : "");
}}).catch(() => {{ render(); status("连不上服务器，结果只存在本浏览器里", true); }});
render();
</script></body></html>
"""

GROUP = """<div class="group" data-id="{id}"><div><b>第 {n} 组</b> <small>{info}</small>
<button onclick="playAll('{id}')">全部从头播放</button></div>
<div class="caption">{caption}</div>
<div class="inputs"><div><small>首帧</small><img src="{first}"></div><div><small>设定图</small>{sheet}</div>
<div><small>proxy</small><video src="{proxy}" controls loop muted playsinline></video></div></div>
<div class="samples">{samples}</div></div>"""

SAMPLE = """<div class="sample" data-label="{label}"><b>{label}</b>
<video src="{src}" controls loop muted playsinline preload="metadata"></video>
<button data-v="good" onclick="mark('{id}','{label}','good')">好</button>
<button data-v="ok" onclick="mark('{id}','{label}','ok')">一般</button>
<button data-v="bad" onclick="mark('{id}','{label}','bad')">差</button>
<button class="best" data-v="best" onclick="mark('{id}','{label}','best')">最佳</button></div>"""


def build(args: argparse.Namespace) -> None:
    root, out, key_path = (Path(p).expanduser() for p in (args.groups, args.out, args.key))
    if key_path.resolve().is_relative_to(out.resolve()):
        raise SystemExit("--key must stay outside the published --out folder")
    out.mkdir(parents=True, exist_ok=True)
    groups = sorted(p for p in root.iterdir() if (p / "meta.json").exists())
    order = groups[:]
    random.Random(f"board:{args.seed}").shuffle(order)
    key, cards = {}, []
    for n, group in enumerate(order, 1):
        meta = json.loads((group / "meta.json").read_text())
        seeds = list(meta["seeds"])
        random.Random(f"board:{args.seed}:{meta['group']}").shuffle(seeds)
        labels = dict(zip(string.ascii_uppercase, seeds))
        key[meta["group"]] = labels
        # Published names carry the letter, not the seed, so the page reveals nothing.
        media = out / "media" / meta["group"]
        media.mkdir(parents=True, exist_ok=True)
        for name in ("first.jpg", "sheet.jpg", "proxy.mp4"):
            if (group / name).exists():
                shutil.copyfile(group / name, media / name)
        for letter, seed in labels.items():
            shutil.copyfile(group / "samples" / f"s{seed:03d}.mp4", media / f"{letter}.mp4")
        rel = f"media/{meta['group']}"
        samples = "".join(SAMPLE.format(label=l, id=meta["group"], src=f"{rel}/{l}.mp4") for l in labels)
        sheet = f'<img src="{rel}/sheet.jpg">' if meta["sheet"] else "<small>（无设定图）</small>"
        info = f"{meta['tag'] or ''} · proxy: {meta['proxy_kind']} · 目标: {meta['target_kind']}/{meta['target_style']}"
        cards.append(GROUP.format(id=meta["group"], n=n, info=html.escape(info), caption=html.escape(meta["prompt"]),
                                  first=f"{rel}/first.jpg", sheet=sheet, proxy=f"{rel}/proxy.mp4", samples=samples))
    page = PAGE.format(title=html.escape(args.title), total=len(cards), groups="\n".join(cards),
                       answers_name=args.answers_name)
    (out / "index.html").write_text(page, encoding="utf-8")
    key_path.parent.mkdir(parents=True, exist_ok=True)
    key_path.write_text(json.dumps(key, indent=1), encoding="utf-8")
    print(f"{out / 'index.html'}: {len(cards)} groups; key {key_path}")


LEVEL = {"good": 2, "ok": 1, "bad": 0}


def preferences(answer: dict, labels: dict[str, int]) -> list[tuple[int, int]]:
    """(winner seed, loser seed) pairs implied by one group's labels."""
    pairs = set()
    level = {labels[l]: LEVEL[v] for l, v in answer.get("labels", {}).items()}
    for a in level:
        for b in level:
            if level[a] > level[b]:
                pairs.add((a, b))
    best = answer.get("best")
    if best:
        for s in labels.values():
            if s != labels[best]:
                pairs.add((labels[best], s))
    return sorted(pairs)


def load_answers(source: str) -> dict:
    if source.startswith(("http://", "https://")):
        with urllib.request.urlopen(source, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    return json.loads(Path(source).expanduser().read_text())


def agree(args: argparse.Namespace) -> None:
    key = json.loads(Path(args.key).expanduser().read_text())
    answers = load_answers(args.answers)
    rows = [json.loads(line) for line in Path(args.results).expanduser().read_text().splitlines()]
    rewards: dict = {}
    for row in rows:
        if not row.get("rewards"):
            continue
        g = rewards.setdefault(row["group"], {})
        for seed, r in row["rewards"].items():
            for name in ("reward", "rank_reward", "abs_reward"):
                g.setdefault(name, {}).setdefault(int(seed), []).append(r[name])
            if r.get("overall_rank") is not None:
                g.setdefault("overall", {}).setdefault(int(seed), []).append(-r["overall_rank"])
        for dim, ranks in row["ranks"].items():
            if dim == "overall":
                continue
            for seed, value in ranks.items():
                g.setdefault(f"dim:{dim}", {}).setdefault(int(seed), []).append(-value)
    tally: dict = {}
    for group, answer in answers.items():
        if group not in rewards:
            continue
        pairs = preferences(answer, {l: int(s) for l, s in key[group].items()})
        for name, per_seed in rewards[group].items():
            mean = {s: sum(v) / len(v) for s, v in per_seed.items()}
            t = tally.setdefault(name, [0, 0, 0])
            for win, lose in pairs:
                if win not in mean or lose not in mean:
                    continue
                d = mean[win] - mean[lose]
                t[0 if d > 0 else 1 if d < 0 else 2] += 1
    print(f"{len(answers)} labelled groups")
    for name, (yes, no, tie) in sorted(tally.items(), key=lambda kv: -kv[1][0] / max(1, kv[1][0] + kv[1][1])):
        print(f"{name:24s} agree {yes:4d}  disagree {no:4d}  tie {tie:4d}  rate {yes / max(1, yes + no):.3f}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--groups", required=True)
    b.add_argument("--out", required=True, help="page folder, published under the eval board root")
    b.add_argument("--key", required=True, help="letter-to-seed map; keep it outside --out")
    b.add_argument("--answers-name", required=True, help="the page saves to /answers/<name>.json")
    b.add_argument("--title", default="RL rubric 校准标注")
    b.add_argument("--seed", type=int, default=0)
    a = sub.add_parser("agree")
    a.add_argument("--key", required=True)
    a.add_argument("--answers", required=True, help="answers file or its board URL")
    a.add_argument("--results", required=True)
    args = p.parse_args()
    {"build": build, "agree": agree}[args.command](args)


if __name__ == "__main__":
    main()
