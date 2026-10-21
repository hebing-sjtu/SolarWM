from __future__ import annotations

import json
from pathlib import Path

import pytest

from solarwm.backends.minimax_h3.proxy_serve import JOB_SCHEMA, Spool, job_segments
from solarwm.errors import DataContractError


def _job(tmp_path: Path, roles: list[str], **extra: object) -> dict[str, object]:
    segments = []
    for index, role in enumerate(roles):
        sample = tmp_path / f"s{index:02d}.pt"
        sample.write_bytes(b"")
        segments.append({"name": sample.stem, "pt": str(sample), "cwm_system": role})
    return {"schema": JOB_SCHEMA, "segments": segments, **extra}


def test_job_segments_chain_on_the_take_stride(tmp_path: Path) -> None:
    segments = job_segments(_job(tmp_path, ["w0_omni", "wn_omni", "wn_omni"], frames=250), base_role="w0_omni")
    assert [(s.index, s.start, s.cwm_system) for s in segments] == [
        (0, 0, "w0_omni"),
        (1, 85, "wn_omni"),
        (2, 170, "wn_omni"),
    ]


@pytest.mark.parametrize(
    ("roles", "extra", "message"),
    [
        (["wn_omni"], {}, "segment 0 must use"),
        (["w0_omni", "w0_omni"], {}, "segment 1 must use"),
        (["w0_omni"], {"frames": 125}, "exceed"),
        (["w0_omni"], {"schema": "other"}, "is not"),
    ],
)
def test_job_segments_refuse_jobs_the_take_cannot_make(
    tmp_path: Path, roles: list[str], extra: dict[str, object], message: str
) -> None:
    with pytest.raises(DataContractError, match=message):
        job_segments(_job(tmp_path, roles, **extra), base_role="w0_omni")


def test_spool_claims_oldest_and_finishes_atomically(tmp_path: Path) -> None:
    spool = Spool(tmp_path / "spool")
    for name in ("b", "a"):
        (spool.root / "queue" / f"{name}.json").write_text(json.dumps({"schema": JOB_SCHEMA}))
    (spool.root / "queue" / "bad.json").write_text("{")
    claimed = spool.claim()
    assert claimed is not None and claimed["id"] in {"a", "b"}
    assert (spool.root / "running" / f"{claimed['id']}.json").is_file()
    spool.progress(claimed["id"], 1, 2)
    spool.finish(claimed["id"], {"take": "x.mp4"})
    assert json.loads((spool.root / "done" / f"{claimed['id']}.json").read_text())["take"] == "x.mp4"
    assert not list((spool.root / "running").iterdir())
    rival = Spool(spool.root, "rival")
    while rival.claim() is not None:
        pass
    assert spool.claim() is None
    assert "unreadable" in json.loads((spool.root / "done" / "bad.json").read_text())["error"]
