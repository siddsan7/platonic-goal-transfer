"""Follow-up CLI. ``freeze`` and ``check`` touch no data. ``pilot`` and ``test`` run only after
every gate passes: committed plan and prediction at HEAD, matching freeze, a new
``followup-<phase>-<label>`` output directory and, for ``test``, a validated PASS pilot directory
from the same freeze. The phase then runs in a private ``multiprocessing`` spawn child that the
parent kills at a hard OS-level deadline. There is no public subcommand that runs a phase without
that deadline.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
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


def _gates(
    prediction: Path, freeze_path: Path, out: Path, phase: str, pilot_dir, design: FollowupDesign
) -> tuple[dict, dict | None]:
    """Every pre-run gate. Returns the freeze and, for ``test``, the verified pilot record."""
    freeze = _load_json(freeze_path)
    prereg.verify_freeze(freeze, design, prediction, repo=prereg.REPO)
    _out_dir(out, phase)
    pilot = validate_pilot(pilot_dir, freeze, design) if phase == "test" else None
    return freeze, pilot


def _child(phase: str, prediction: Path, freeze_path: Path, out: Path, pilot_dir) -> int:
    """Spawn target (private): re-check every gate, then hand the freeze and the pilot record
    returned by verification straight to the engine."""
    design = FollowupDesign()
    try:
        freeze, pilot = _gates(prediction, freeze_path, out, phase, pilot_dir, design)
    except (PreregError, OSError, json.JSONDecodeError, KeyError) as exc:
        print(f"refused: {exc}")
        return EXIT_REFUSED
    from pgt.followup import engine  # lazy: gates and their tests never load training code

    manifest = engine.run_phase(phase, design, freeze, out, pilot=pilot)
    print(json.dumps({"status": manifest.get("status"), "out": str(out)}))
    return 0 if manifest.get("status") == "PASS" else 1


def _child_entry(*args) -> None:
    raise SystemExit(_child(*args))


def run_with_deadline(target, args: tuple, seconds: float) -> int:
    """Run ``target(*args)`` in a spawned child and kill it at the deadline. The kill is done by
    the OS (TerminateProcess on Windows), so it also stops a single long training step."""
    proc = multiprocessing.get_context("spawn").Process(target=target, args=args)
    proc.start()
    proc.join(seconds)
    if proc.is_alive():
        proc.kill()
        proc.join()
        print(f"refused: hard deadline of {seconds:.0f}s reached; the process was killed")
        return EXIT_DEADLINE
    return proc.exitcode if proc.exitcode is not None else EXIT_DEADLINE


def deadline_s(phase: str, design: FollowupDesign) -> float:
    cap = design.cap_pilot_s if phase == "pilot" else design.cap_test_s
    return design.cap_build_s + cap + DEADLINE_GRACE_S


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m pgt.followup.cli")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("freeze", "check", "pilot", "test"):
        p = sub.add_parser(name)
        p.add_argument("--prediction", type=Path, required=True)
        if name == "freeze":
            p.add_argument("--write", type=Path, required=True)
            continue
        p.add_argument("--freeze", type=Path, required=True)
        if name == "check":
            continue
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
        _gates(args.prediction, args.freeze, args.out, args.cmd, args.pilot, design)
    except (PreregError, OSError, json.JSONDecodeError, KeyError) as exc:
        print(f"refused: {exc}")
        return EXIT_REFUSED
    child_args = (args.cmd, args.prediction, args.freeze, args.out, args.pilot)
    return run_with_deadline(_child_entry, child_args, deadline_s(args.cmd, design))


if __name__ == "__main__":
    raise SystemExit(main())
