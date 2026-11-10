"""A per-segment quality gate for long proxy takes, backed by an HTTP VLM judge.

A long take drifts one segment at a time (an extra training dummy, a door in a plain wall), and a
later segment only ever sees the last 34 frames of the one before, so a bad segment is best
caught and regenerated before the take moves on. A job opts in with::

    "gate": {"url": "http://HOST:7870/judge", "anchor": ".../anchor.png", "oc": ".../oc.jpg",
             "attempts": 3, "min_score": 6, "samples": 6}

``roll_segments`` calls the gate on SP rank 0 after every generated segment; a segment that fails
is regenerated with a new noise seed, up to ``attempts`` tries, and the best-judged try is kept.
The judge is FastVideo's ``scripts/h3_proxy/serve/segment_judge.py``; an unreachable or
unparseable judge passes the segment, so a judge outage never stalls a take.
"""

from __future__ import annotations

import base64
import json
import urllib.request
from collections.abc import Mapping
from pathlib import Path
from typing import Any


class HttpGate:
    def __init__(self, spec: Mapping[str, Any]) -> None:
        self.url = str(spec["url"])
        self.anchor = str(spec["anchor"])
        self.oc = str(spec["oc"]) if spec.get("oc") else None
        self.attempts = max(1, int(spec.get("attempts", 3)))
        self.min_score = float(spec.get("min_score", 6))
        self.samples = max(1, int(spec.get("samples", 6)))
        self.timeout = float(spec.get("timeout", 600))

    def __call__(self, index: int, attempt: int, *, proxy: Any, produced: Any, first: int, folder: Path) -> dict[str, Any]:
        """Judge the frames ``produced[first:]`` against ``proxy`` (both ``[T, H, W, 3]`` uint8)."""

        import numpy as np
        from PIL import Image

        folder.mkdir(parents=True, exist_ok=True)
        picks = np.linspace(first, len(produced) - 1, self.samples).round().astype(int)
        pairs = []
        for i in picks:
            p, g = folder / f"proxy-{i:03d}.jpg", folder / f"generated-{i:03d}.jpg"
            Image.fromarray(proxy[i]).save(p, quality=92)
            Image.fromarray(produced[i]).save(g, quality=92)
            pairs.append([_b64(p), _b64(g)])
        # Images travel as bytes: the judge may run on another pod, where these paths do not exist.
        request = {
            "anchor_b64": _b64(Path(self.anchor)),
            "oc_b64": _b64(Path(self.oc)) if self.oc else None,
            "pairs_b64": pairs,
            "tag": f"seg{index:02d} try{attempt}",
        }
        try:
            post = urllib.request.Request(
                self.url, data=json.dumps(request).encode(), headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(post, timeout=self.timeout) as response:
                verdict = json.loads(response.read())
        except Exception as exc:
            verdict = {"error": f"{type(exc).__name__}: {exc}"}
            print(f"[h3-proxy-gate] JUDGE ERROR seg{index:02d} try{attempt}: {verdict['error']} "
                  "(segment passes unjudged)", flush=True)
        verdict.pop("raw", None)
        score = verdict.get("score")
        usable = verdict.get("parsed") and score is not None and "error" not in verdict
        verdict["pass"] = (not verdict.get("failures") and float(score) >= self.min_score) if usable else True
        (folder / "verdict.json").write_text(json.dumps(verdict, ensure_ascii=False, indent=2) + "\n")
        return verdict


def _b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def verdict_rank(verdict: Mapping[str, Any]) -> tuple[int, float, int]:
    """Higher is better: a pass, then the judge's score, then fewer failure kinds."""

    score = verdict.get("score")
    return (
        int(bool(verdict.get("pass"))),
        float(score) if score is not None else -1.0,
        -len(verdict.get("failures") or ()),
    )


__all__ = ["HttpGate", "verdict_rank"]
