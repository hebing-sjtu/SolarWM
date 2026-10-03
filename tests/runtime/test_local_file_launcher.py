from __future__ import annotations

from solarwm.runtime.local_file_launcher import main


def test_local_file_launcher_sets_rank_environment_without_master_address(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("SOLARWM_TEST_OUTPUT", str(tmp_path))
    monkeypatch.setenv("MASTER_ADDR", "forbidden")
    monkeypatch.setenv("MASTER_PORT", "1")
    code = """
import os
from pathlib import Path

assert "MASTER_ADDR" not in os.environ
assert "MASTER_PORT" not in os.environ
assert os.environ["SOLARWM_DISTRIBUTED_INIT_METHOD"].startswith("file:///")
rank = int(os.environ["RANK"])
assert int(os.environ["LOCAL_RANK"]) == rank
assert os.environ["WORLD_SIZE"] == "2"
Path(os.environ["SOLARWM_TEST_OUTPUT"], f"rank-{rank}").write_text("ok")
"""
    result = main(
        [
            "--nproc-per-node",
            "2",
            "--store-path",
            str(tmp_path / "store"),
            "--",
            "-c",
            code,
        ]
    )
    assert result == 0
    assert {path.name for path in tmp_path.glob("rank-*")} == {"rank-0", "rank-1"}
    assert not (tmp_path / "store").exists()
