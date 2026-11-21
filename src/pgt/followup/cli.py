"""Follow-up CLI. ``freeze`` and ``check`` touch no data. ``pilot`` and ``test`` run only after
every gate passes: committed plan and prediction at HEAD, matching freeze, a new
``followup-<phase>-<label>`` output directory and, for ``test``, a validated PASS pilot directory
from the same freeze. The phase then runs in a child process with a hard OS-level deadline.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from pgt.followup import prereg
from pgt.followup.design import FollowupDesign
from pgt.followup.prereg import PreregError

OUT_PREFIX = "followup-"
DEADLINE_GRACE_S = 60  # on top of the design's build + phase caps, for start-up and writing
EXIT_REFUSED, EXIT_DEADLINE = 2, 4


def _out_dir(path: Path, phase: str) -> Path:
    if not path.name.startswith(f"{OUT_PREFIX}{phase}-") or path.name == f"{OUT_PREFIX}{phase}-":
        raise PreregError(f"--out must be named {OUT_PREFIX}{phase}-<label>")
    if path.exists():
        raise PreregError(f"{path} already exists; never overwrite or resume a run")
    return path


def _load_json(path: Path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise PreregError(f"{path} is not a JSON object")
    return data


def validate_pilot(pilot_dir: Path, freeze: dict, design: FollowupDesign) -> dict:
    from pgt.followup.engine import verify_pilot

    try:
        return verify_pilot(Path(pilot_dir) / "manifest.json", freeze, design)
    except (ValueError, KeyError, TypeError) as exc:
        raise PreregError(str(exc)) from exc


def _gates(args, phase: str, design: FollowupDesign) -> dict:
    freeze = _load_json(args.freeze)
    prereg.verify_freeze(freeze, design, args.prediction, repo=prereg.REPO)
    _out_dir(args.out, phase)
    if phase == "test":
        validate_pilot(args.pilot, freeze, design)
    return freeze


def _execute(args) -> int:
    """Child-process body: re-check every gate, then run the phase in the engine."""
    design = FollowupDesign()
    freeze = _gates(args, args.phase, design)
    from pgt.followup import engine  # lazy: gates and their tests never load training code

    pilot = _load_json(Path(args.pilot) / "manifest.json") if args.phase == "test" else None
    manifest = engine.run_phase(args.phase, design, freeze, args.out, pilot=pilot)
    print(json.dumps({"status": manifest.get("status"), "out": str(args.out)}))
    return 0 if manifest.get("status") == "PASS" else 1


def run_with_deadline(cmd: list[str], seconds: float) -> int:
    """Run ``cmd`` and kill it at the deadline. The kill is done by the OS (TerminateProcess
    on Windows), so it also stops a single long training step."""
    try:
        return subprocess.run(cmd, timeout=seconds, check=False).returncode
    except subprocess.TimeoutExpired:
        print(f"refused: hard deadline of {seconds:.0f}s reached; the process was killed")
        return EXIT_DEADLINE


def deadline_s(phase: str, design: FollowupDesign) -> float:
    cap = design.cap_pilot_s if phase == "pilot" else design.cap_test_s
    return design.cap_build_s + cap + DEADLINE_GRACE_S


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m pgt.followup.cli")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("freeze", "check", "pilot", "test", "_run"):
        p = sub.add_parser(name)
        p.add_argument("--prediction", type=Path, required=True)
        if name == "freeze":
            p.add_argument("--write", type=Path, required=True)
            continue
        p.add_argument("--freeze", type=Path, required=True)
        if name == "check":
            continue
        if name == "_run":
            p.add_argument("--phase", choices=("pilot", "test"), required=True)
        p.add_argument("--out", type=Path, required=True)
        p.add_argument("--pilot", type=Path, required=(name == "test"))
    args = ap.parse_args(argv)
    design = FollowupDesign()
    try:
        if args.cmd == "freeze":
            record = prereg.make_freeze(design, args.prediction, repo=prereg.REPO)
            prereg.write_once(args.write, record)
            print(json.dumps(record, indent=2))
            return 0
        if args.cmd == "check":
            prereg.verify_freeze(_load_json(args.freeze), design, args.prediction, repo=prereg.REPO)
            print("freeze OK")
            return 0
        if args.cmd == "_run":
            return _execute(args)
        _gates(args, args.cmd, design)
    except (PreregError, OSError, json.JSONDecodeError, KeyError) as exc:
        print(f"refused: {exc}")
        return EXIT_REFUSED
    cmd = [
        sys.executable,
        "-m",
        "pgt.followup.cli",
        "_run",
        "--phase",
        args.cmd,
        "--prediction",
        str(args.prediction),
        "--freeze",
        str(args.freeze),
        "--out",
        str(args.out),
    ]
    if args.cmd == "test":
        cmd += ["--pilot", str(args.pilot)]
    return run_with_deadline(cmd, deadline_s(args.cmd, design))


if __name__ == "__main__":
    raise SystemExit(main())
