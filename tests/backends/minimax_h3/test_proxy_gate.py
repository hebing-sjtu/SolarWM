from __future__ import annotations

import dataclasses
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from solarwm.backends.minimax_h3 import inference, proxy_rollout  # noqa: E402
from solarwm.backends.minimax_h3.proxy_gate import verdict_rank  # noqa: E402
from solarwm.backends.minimax_h3.proxy_rollout import TakeSegment, _WriterState, roll_segments  # noqa: E402

LATENT = 2
SEED = 7


@dataclasses.dataclass(frozen=True)
class FakeBatch:
    cwm_system: str
    proxy_latents: Any
    target_latents: Any
    num_given_latent_frames: int = 10


class FakeReader:
    def read(self, path: Path) -> FakeBatch:
        role = "w0_omni" if path.name.startswith("s00") else "wn_omni"
        return FakeBatch(role, torch.zeros(24, 37, LATENT, LATENT), torch.zeros(24, 37, LATENT, LATENT))


class FakeCore:
    """Latents carry their noise seed, so the tail handed on names the try it came from."""

    def __init__(self) -> None:
        self.calls: list[tuple[int, float]] = []

    def generate(self, batch: FakeBatch, *, noise_seed: int, num_inference_steps: int) -> Any:
        self.calls.append((noise_seed, float(batch.target_latents[0, 0, 0, 0])))
        return torch.full((1, 24, 37, LATENT, LATENT), float(noise_seed))


class FakeMp4:
    def __init__(self, path: Path, fps: int) -> None:
        self.frames: list[Any] = []

    def extend(self, frames: Any) -> None:
        self.frames.extend(frames)

    def close(self) -> None:
        pass


class FakeGate:
    attempts = 3

    def __init__(self, scores: dict[int, list[tuple[bool, int]]]) -> None:
        self.scores = scores

    def __call__(self, index: int, attempt: int, *, proxy: Any, produced: Any, first: int, folder: Path) -> dict:
        passed, score = self.scores.get(index, [(True, 9)] * 3)[attempt]
        return {"pass": passed, "score": score, "failures": [] if passed else ["dummies"]}


@pytest.fixture
def rollout(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(inference, "decode_h3_latents",
                        lambda vae, latents, device: torch.full((1, 3, 124, 8, 8), float(latents.flatten()[0]) / 1e4))
    monkeypatch.setattr(proxy_rollout, "encode_h3_pixels",
                        lambda vae, pixels, device: torch.full((1, 24, 12, LATENT, LATENT), float(pixels.flatten()[0]) * 1e4))
    monkeypatch.setattr(proxy_rollout, "_proxy_pixels", lambda vae, latents, size, device: torch.zeros(1, 3, 124, *size))
    monkeypatch.setattr(proxy_rollout, "_Mp4", FakeMp4)
    monkeypatch.setattr(_WriterState, "vae", lambda self, config, device: object())
    core = FakeCore()
    runtime = SimpleNamespace(
        torch=torch,
        dist=SimpleNamespace(is_initialized=lambda: False),
        topology=SimpleNamespace(raw_rank=0, raw_world_size=1, dp_rank=0),
        device=torch.device("cpu"),
        core=core,
    )
    segments = tuple(
        TakeSegment(i, f"s{i:02d}", tmp_path / f"s{i:02d}.pt", i * 85, "w0_omni" if i == 0 else "wn_omni")
        for i in range(3)
    )

    def run(gate: Any = None) -> list[dict]:
        return roll_segments(
            runtime, FakeReader(), segments, config={"data": {"latent_height": LATENT, "latent_width": LATENT}},
            root=tmp_path / "out", fps=24, real_target=False, seed=SEED, steps=1, writer=_WriterState(), gate=gate,
        )

    return run, core


def test_ungated_take_is_one_try_per_segment(rollout) -> None:
    run, core = rollout
    records = run()
    assert [seed for seed, _ in core.calls] == [SEED, SEED + 1000, SEED + 2000]
    assert [r["noise_seed"] for r in records] == [SEED, SEED + 1000, SEED + 2000]
    assert all("gate" not in r for r in records)


def test_failed_segments_are_retried_and_the_kept_try_continues_the_take(rollout) -> None:
    run, core = rollout
    gate = FakeGate({1: [(False, 3), (False, 4), (True, 8)], 2: [(False, 5), (False, 7), (False, 2)]})
    records = run(gate)
    seeds = [seed for seed, _ in core.calls]
    assert seeds == [SEED, SEED + 1000, SEED + 1100, SEED + 1200, SEED + 2000, SEED + 2100, SEED + 2200]
    # segment 1 keeps its first passing try; segment 2 never passes and keeps its best score
    assert [r["noise_seed"] for r in records] == [SEED, SEED + 1200, SEED + 2100]
    assert [len(r["gate"]["tries"]) for r in records] == [1, 3, 3]
    # every try of segment 2 starts from the tail of segment 1's kept try, not its last try
    assert [round(given) for seed, given in core.calls[4:]] == [SEED + 1200] * 3


def test_http_gate_sends_image_bytes_and_scores_the_verdict(tmp_path: Path) -> None:
    import base64
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    import numpy as np
    from PIL import Image

    from solarwm.backends.minimax_h3.proxy_gate import HttpGate

    seen: list[dict] = []

    class Judge(BaseHTTPRequestHandler):
        def log_message(self, *_: Any) -> None:
            pass

        def do_POST(self) -> None:
            seen.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            body = json.dumps({"parsed": True, "score": 9, "failures": ["structures"]}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Judge)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    anchor = tmp_path / "anchor.png"
    Image.new("RGB", (8, 8)).save(anchor)
    frames = np.zeros((124, 8, 8, 3), dtype=np.uint8)
    try:
        gate = HttpGate({"url": f"http://127.0.0.1:{server.server_port}/judge", "anchor": str(anchor), "samples": 3})
        verdict = gate(1, 0, proxy=frames, produced=frames, first=34, folder=tmp_path / "try-0")
    finally:
        server.shutdown()
    assert verdict["pass"] is False and verdict["score"] == 9
    request = seen[0]
    assert len(request["pairs_b64"]) == 3 and "pairs" not in request
    assert base64.b64decode(request["anchor_b64"]) == anchor.read_bytes()

    down = HttpGate({"url": "http://127.0.0.1:9/judge", "anchor": str(anchor), "timeout": 2})
    verdict = down(1, 0, proxy=frames, produced=frames, first=34, folder=tmp_path / "try-1")
    assert verdict["pass"] is True and "error" in verdict


def test_verdict_rank_prefers_a_pass_then_the_score() -> None:
    assert verdict_rank({"pass": True, "score": 6}) > verdict_rank({"pass": False, "score": 9})
    assert verdict_rank({"pass": False, "score": 7}) > verdict_rank({"pass": False, "score": 5})
    assert verdict_rank({"pass": False, "score": 5, "failures": ["a"]}) > verdict_rank(
        {"pass": False, "score": 5, "failures": ["a", "b"]})
