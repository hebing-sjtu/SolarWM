import hashlib
import importlib.util
import json
import shutil
import subprocess
import threading
import urllib.request
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[2] / "tools" / "eval_board"


def load(name):
    spec = importlib.util.spec_from_file_location(f"eval_board_{name}", TOOLS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def video(path: Path, colour: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"color=c={colour}:s=64x32:r=24:d=2",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fake_task(root: Path, task: str, samples: dict[str, str], *, weight_source="live", config=None):
    task_dir = root / task
    if config is not None:
        task_dir.mkdir(parents=True, exist_ok=True)
        (task_dir / "resolved-config.json").write_text(json.dumps({"config": config}))
    for slot, (sample_id, colour) in enumerate(samples.items()):
        name = f"slot-{slot:06d}"
        slot_dir = task_dir / "proxy-inference" / f"wave-{slot:03d}" / "dp-rank-00000" / name
        colours = {"generated.mp4": colour, "target.mp4": "white", "proxy.mp4": "gray"}
        artifacts = [
            {"path": f"{name}/{file}", "digest": video(slot_dir / file, fill)}
            for file, fill in colours.items()
        ]
        manifest = {"artifacts": artifacts, "case": {"sample_id": sample_id, "prompt": ""}}
        (slot_dir / "manifest.json").write_text(json.dumps(manifest))
    (task_dir / "DONE.json").write_text(json.dumps({"weight_source": weight_source}))


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_export_dedupes_media_scores_tasks_and_reruns_without_encoding(tmp_path):
    export = load("export")
    loops = tmp_path / "loops"
    fake_task(
        loops, "loop/run-a/step-000000/correct", {"s1": "white", "s2": "red"}, weight_source="base"
    )
    fake_task(loops, "loop/run-a/step-000250/correct", {"s1": "red", "s2": "white"})
    fake_task(loops, "loop/run-a/step-000500/correct", {"s1": "red"})
    (loops / "loop/run-a/step-000500/static").mkdir(parents=True)  # still running: no DONE.json
    heldout = tmp_path / "heldout.jsonl"
    heldout.write_text(
        json.dumps(
            {
                "name": "s1",
                "prompt": "a car",
                "proxy_modality": "duv",
                "target_kind": "color",
                "tag": "drive",
            }
        )
        + "\n"
    )
    out = tmp_path / "board"
    argv = [
        "--loops-root",
        str(loops),
        "--loop",
        "loop",
        "--heldout",
        str(heldout),
        "--out",
        str(out),
        "--workers",
        "2",
        "--label",
        "run-a=Run A",
        "--label",
        "run-a/step-000000=Base model",
    ]

    assert export.main(argv) == 0
    index = json.loads((out / "index.json").read_text())
    assert [(t["step"], t["ablation"], t["label"]) for t in index["tasks"]] == [
        (0, "correct", "Base model"),
        (250, "correct", "Run A · step 250"),
        (500, "correct", "Run A · step 500"),
    ]
    assert index["tasks"][0]["weights"] == "base"
    assert index["samples"]["s1"]["modality"] == "duv"
    base = index["tasks"][0]["items"]
    assert base["s1"]["psnr_late"] == 100.0  # white generated against the white target
    assert base["s2"]["psnr_late"] < 30
    assert index["samples"]["s1"]["target"] == index["samples"]["s2"]["target"]
    media = sorted(p.name for p in (out / "media").iterdir())
    # Red, the gray proxy, and white: the white outputs are byte-identical to the white target.
    assert len(media) == 3
    assert (out / "index.html").read_bytes() == (TOOLS / "board.html").read_bytes()
    quality = sorted(p.name for p in (loops / "loop-quality").iterdir())
    assert quality == [f"run-a__step-{n}__correct.json" for n in ("000000", "000250", "000500")]

    stamps = {p.name: p.stat().st_mtime_ns for p in (out / "media").iterdir()}
    assert export.main(argv) == 0
    assert {p.name: p.stat().st_mtime_ns for p in (out / "media").iterdir()} == stamps


def stage_config(stage, steps, data="heldout32"):
    return {
        "train": {"stage": stage},
        "data": {"data_path": f"/cache/{data}", "align_proxy_reference_time": False},
        "validation": {"num_inference_steps": steps, "sample_count": 2, "noise_seed": 7},
    }


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_export_reads_stage_protocol_ema_roots_and_experiment_notes(tmp_path):
    export = load("export")
    live, ema = tmp_path / "loops", tmp_path / "loops-ema"
    samples = {"s1": "red"}
    fake_task(live, "loop/bd/step-000300/correct", samples, config=stage_config("stage0p5", 30))
    fake_task(
        live,
        "loop/bd-desc/step-000300/correct",
        samples,
        config=stage_config("stage0p5", 30, "heldout32_desc"),
    )
    fake_task(live, "loop/dmd/step-000500/correct", samples, config=stage_config("stage2", 4))
    fake_task(
        ema,
        "loop/dmd/step-000500/correct",
        samples,
        weight_source="ema",
        config=stage_config("stage2", 4),
    )
    notes = tmp_path / "runs.json"
    notes.write_text(
        json.dumps({"runs": {"dmd": {"label": "DMD run", "stage": "DMD", "note": "Two sentences."}}})
    )
    heldout = tmp_path / "heldout.jsonl"
    heldout.write_text(json.dumps({"name": "s1", "prompt": "", "proxy_modality": "duv"}) + "\n")
    out = tmp_path / "board"
    argv = ["--loops-root", str(live), "--loops-root", str(ema), "--loop", "loop"]
    argv += ["--heldout", str(heldout), "--out", str(out), "--experiments", str(notes)]

    assert export.main(argv) == 0
    index = json.loads((out / "index.json").read_text())
    tasks = {t["id"]: t for t in index["tasks"]}
    assert sorted(tasks) == [
        "loop/bd-desc/step-000300/correct",
        "loop/bd/step-000300/correct",
        "loop/dmd/step-000500/correct",
        "loop/dmd/step-000500/correct@ema",
    ]
    bd = tasks["loop/bd/step-000300/correct"]
    assert bd["stage"] == "BD"
    assert bd["protocol"] == {
        "ablation": "correct",
        "weights": "live",
        "eval_data": "heldout32",
        "steps": 30,
        "sample_count": 2,
        "noise_seed": 7,
        "aligned": False,
    }
    assert tasks["loop/bd-desc/step-000300/correct"]["protocol"]["eval_data"] == "heldout32_desc"
    assert tasks["loop/dmd/step-000500/correct"]["label"] == "DMD run · step 500"
    assert tasks["loop/dmd/step-000500/correct@ema"]["label"] == "DMD run · step 500 · ema"
    assert {t["stage"] for t in tasks.values() if t["run"] == "dmd"} == {"DMD"}
    assert index["experiments"]["dmd"] == {"label": "DMD run", "stage": "DMD", "note": "Two sentences."}

    notes.write_text(json.dumps({"runs": {"dmd": {"stage": "Stage2"}}}))
    with pytest.raises(SystemExit):
        export.main(argv)


def test_report_pairs_deltas_per_group_and_lists_tasks_not_evaluated_yet(tmp_path):
    report = load("report")
    index = {
        "generated_at": "2026-10-13T00:00:00+00:00",
        "samples": {
            "s1": {"kind": "color", "modality": "duv"},
            "s2": {"kind": "high", "modality": "depth"},
        },
        "tasks": [
            {
                "id": "a",
                "items": {
                    "s1": {"psnr_late": 20.0, "ssim_late": 0.5},
                    "s2": {"psnr_late": 18.0, "ssim_late": 0.4},
                },
            },
            {
                "id": "b",
                "items": {
                    "s1": {"psnr_late": 19.0, "ssim_late": 0.5},
                    "s2": {"psnr_late": 18.5, "ssim_late": 0.4},
                },
            },
        ],
    }
    path = tmp_path / "index.json"
    path.write_text(json.dumps(index))
    out = tmp_path / "reports" / "r.md"
    argv = [
        "--index",
        str(path),
        "--task",
        "A=a",
        "--task",
        "B=b",
        "--task",
        "C=c",
        "--pair",
        "A:B",
        "--out",
        str(out),
    ]
    assert report.main(argv) == 0
    text = out.read_text()
    assert "Not evaluated yet: `C`." in text
    row = next(
        line for line in text.splitlines() if line.startswith("| `A` - `B`") and "+0.25" in line
    )
    cells = [cell.strip() for cell in row.strip("|").split("|")]
    assert cells[1:4] == ["+0.25 ± 0.75 (1/2)", "+1.00 ± 0.00 (1/1)", "-0.50 ± 0.00 (0/1)"]
    with pytest.raises(SystemExit):
        report.main([*argv[:-2], "--pair", "A:Z", "--out", str(out)])


def test_serve_ranges_and_refuses_paths_outside_the_board(tmp_path):
    serve = load("serve")
    (tmp_path / "media").mkdir()
    (tmp_path / "media" / "a.mp4").write_bytes(bytes(range(100)))
    (tmp_path / "index.html").write_text("board")
    (tmp_path / ".incoming").mkdir()
    (tmp_path / ".incoming" / "x.mp4").write_bytes(b"partial")
    root = str(tmp_path.resolve())
    assert serve.resolve(root, "/") == str(tmp_path.resolve() / "index.html")
    assert serve.resolve(root, "/../" + tmp_path.name + "/index.html") is None
    assert serve.resolve(root, "/%2e%2e/etc/passwd") is None
    assert serve.resolve(root, "/.incoming/x.mp4") is None

    server = serve.Server(("127.0.0.1", 0), serve.make_handler(root))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        base = f"http://127.0.0.1:{server.server_address[1]}"
        request = urllib.request.Request(base + "/media/a.mp4", headers={"Range": "bytes=10-19"})
        with urllib.request.urlopen(request) as response:
            assert response.status == 206
            assert response.headers["Content-Range"] == "bytes 10-19/100"
            assert "immutable" in response.headers["Cache-Control"]
            assert response.read() == bytes(range(10, 20))
        with urllib.request.urlopen(base + "/") as response:
            assert response.read() == b"board"
            assert response.headers["Cache-Control"] == "no-cache"
    finally:
        server.shutdown()


def test_sync_fetches_media_that_is_missing_or_short_and_leaves_the_index_for_last(tmp_path):
    sync = load("sync")
    (tmp_path / "media").mkdir()
    (tmp_path / "media" / "have.mp4").write_bytes(b"12345")
    (tmp_path / "media" / "short.mp4").write_bytes(b"12")
    remote = {
        "media/have.mp4": 5,
        "media/short.mp4": 5,
        "media/new.mp4": 3,
        "index.json": 10,
        "index.html": 10,
    }
    assert sync.missing(remote, str(tmp_path)) == ["media/new.mp4", "media/short.mp4"]
    sizes = {"a": 3, "b": 3, "c": 5}
    assert list(sync.batches(["a", "b", "c"], sizes, limit=6)) == [["a", "b"], ["c"]]
    index = {
        "samples": {"s1": {"target": "media/t.mp4", "proxy": {"correct": "media/p.mp4"}}},
        "tasks": [{"items": {"s1": {"video": "media/g.mp4"}}}],
    }
    assert sync.referenced(index) == {"media/t.mp4", "media/p.mp4", "media/g.mp4"}
