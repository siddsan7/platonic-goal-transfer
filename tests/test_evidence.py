import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_evidence.py"
HEADER = "split,text_dim,seed,cluster,arm,episodes,successes,wrong_pad,timeouts\n"


def load_builder():
    spec = importlib.util.spec_from_file_location("build_evidence", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_committed_evidence_reproduces_reported_effects():
    """paired_counts.csv must reproduce every metrics.json effect and replay every example."""
    out = subprocess.run(
        [sys.executable, str(SCRIPT), "check"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert (
        "final-d1: test, width 1, 5 seed(s) x 256 clusters, trim_swap - goal_swap = +50.00" in out
    )
    assert "final-d2: test, width 2, 5 seed(s) x 256 clusters, trim_swap - goal_swap = +0.00" in out
    assert "replay: 4 logged episodes re-executed" in out


@pytest.mark.parametrize(
    ("body", "message"),
    [
        (
            "test,1,11,abc,clean,16,8,8,0\ntest,1,11,abc,clean,16,8,8,0\n",
            "duplicate row",
        ),
        ("test,1,11,abc,clean,16,8,7,0\n", "!= episodes"),
        ("test,1,11,abc,clean,16,17,-1,0\n", "negative count"),
    ],
)
def test_read_counts_rejects_invalid_rows(tmp_path, body, message):
    path = tmp_path / "paired_counts.csv"
    path.write_text(HEADER + body, encoding="utf-8")
    with pytest.raises(SystemExit, match=message):
        load_builder().read_counts(path)


def test_read_counts_accepts_consistent_rows(tmp_path):
    path = tmp_path / "paired_counts.csv"
    path.write_text(
        HEADER + "test,1,11,abc,clean,16,8,8,0\ntest,1,11,abc,trim_swap,16,16,0,0\n",
        encoding="utf-8",
    )
    assert len(load_builder().read_counts(path)) == 2
