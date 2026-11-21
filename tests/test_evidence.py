import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_committed_evidence_reproduces_reported_effects():
    """paired_counts.csv must reproduce every metrics.json effect and replay every example."""
    out = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "build_evidence.py"), "check"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert (
        "final-d1: test, width 1, 5 seed(s) x 256 clusters, trim_swap - goal_swap = +50.00" in out
    )
    assert "final-d2: test, width 2, 5 seed(s) x 256 clusters, trim_swap - goal_swap = +0.00" in out
    assert "replay: 4 logged episodes re-executed" in out
