"""Freeze records and the prediction gate. No prediction text is ever written here.

The plan lives at the fixed path ``docs/followup.md``. The plan and the candidate's prediction
must both be committed at HEAD with no staged, unstaged or untracked change; their content is
read back from the HEAD blob. The prediction's commit and author come from Git, never from the
file. All digests normalise CRLF to LF and use repository-relative paths, so a clone, a merge
checkout or a different directory gives the same IDs.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

from pgt.followup.design import FollowupDesign

PKG = Path(__file__).resolve().parent
REPO = PKG.parents[2]
PLAN_PATH = "docs/followup.md"
ORIGINALS = ("docs/protocol.md",)  # the original study's protocol the follow-up builds on
PLACEHOLDERS = frozenset(
    {
        "todo",
        "tbd",
        "placeholder",
        "fill in",
        "fill me",
        "xxx",
        "lorem ipsum",
        "your name",
        "your prediction",
        "name",
        "prediction",
    }
)  # rejected only when the whole field is one of these, ignoring case and wrapping brackets


class PreregError(RuntimeError):
    pass


def _lf(data: bytes) -> bytes:
    return data.replace(b"\r\n", b"\n")


def sha(data: bytes) -> str:
    return hashlib.sha256(_lf(data)).hexdigest()


def git(*args: str, repo: Path = REPO) -> str:
    try:
        res = subprocess.run(
            ["git", "-C", str(repo), *args], capture_output=True, timeout=30, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise PreregError(f"git unavailable: {exc}") from exc
    if res.returncode != 0:
        raise PreregError(f"git {' '.join(args)} failed: {res.stderr.decode(errors='replace')}")
    return res.stdout.decode("utf-8", errors="replace")


def repo_rel(path: Path | str, repo: Path = REPO) -> str:
    p = Path(path)
    p = p if p.is_absolute() else Path.cwd() / p
    try:
        return p.resolve().relative_to(repo.resolve()).as_posix()
    except ValueError as exc:
        raise PreregError(f"{path} is not inside the repository {repo}") from exc


def committed_blob(rel: str, repo: Path = REPO) -> bytes:
    """Content of ``rel`` at HEAD; refuses untracked, staged or modified files."""
    if git("status", "--porcelain", "--untracked-files=all", "--", rel, repo=repo).strip():
        raise PreregError(f"{rel} has uncommitted, staged or untracked changes; commit it first")
    if not git("ls-files", "--", rel, repo=repo).strip():
        raise PreregError(f"{rel} is not tracked at HEAD")
    try:
        res = subprocess.run(
            ["git", "-C", str(repo), "show", f"HEAD:{rel}"],
            capture_output=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise PreregError(f"git unavailable: {exc}") from exc
    if res.returncode != 0:
        raise PreregError(f"{rel} is not in HEAD")
    on_disk = repo / rel
    if not on_disk.is_file() or _lf(on_disk.read_bytes()) != _lf(res.stdout):
        raise PreregError(f"{rel} on disk differs from HEAD")
    return res.stdout


def check_prediction(rel: str, repo: Path = REPO) -> dict:
    """The candidate's own committed prediction: JSON with non-placeholder ``author`` and
    ``prediction``. Commit and Git author are resolved from Git and returned alongside."""
    blob = committed_blob(rel, repo)
    try:
        data = json.loads(blob.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PreregError(f"prediction is not JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise PreregError("prediction must be a JSON object")
    for key in ("author", "prediction"):
        value = data.get(key)
        if not isinstance(value, str):
            raise PreregError(f"prediction field {key!r} must be a JSON string")
        text = value.strip()
        if not any(ch.isalpha() for ch in text):
            raise PreregError(f"prediction field {key!r} is empty")
        core = " ".join(text.lower().strip("<>[]{}()\"'`.:;!?-_* ").split())
        if core in PLACEHOLDERS:
            raise PreregError(f"prediction field {key!r} is a placeholder")
    commit, _, git_author = (
        git("log", "-1", "--format=%H%x00%an", "--", rel, repo=repo).strip().partition("\x00")
    )
    if not commit or not git_author.strip():
        raise PreregError("could not resolve the prediction's commit and author from Git")
    return {"path": rel, "commit": commit, "git_author": git_author, "author": data["author"]}


def code_digest() -> str:
    """The follow-up package plus the reused original ``pgt`` modules, by relative path."""
    m = hashlib.sha256()
    src = PKG.parents[1]
    for path in sorted([*PKG.glob("*.py"), *PKG.parent.glob("*.py")]):
        m.update(path.relative_to(src).as_posix().encode() + b"\0")
        m.update(_lf(path.read_bytes()))
    return m.hexdigest()


def make_freeze(design: FollowupDesign, prediction: Path | str, repo: Path = REPO) -> dict:
    rel = repo_rel(prediction, repo)
    pred = check_prediction(rel, repo)
    return {
        "design_digest": design.digest(),
        "plan": {"path": PLAN_PATH, "digest": sha(committed_blob(PLAN_PATH, repo))},
        "prediction": pred | {"digest": sha(committed_blob(rel, repo))},
        "originals": {p: sha((repo / p).read_bytes()) for p in ORIGINALS},
        "code_digest": code_digest(),
    }


def verify_freeze(
    freeze: dict, design: FollowupDesign, prediction: Path | str, repo: Path = REPO
) -> None:
    """Every ID must match the current committed files exactly; any drift blocks the run."""
    now = make_freeze(design, prediction, repo)
    drift = sorted(k for k in now.keys() | freeze.keys() if freeze.get(k) != now.get(k))
    if drift:
        raise PreregError(f"frozen IDs no longer match: {drift}")


def write_once(path: Path, record: dict) -> None:
    """Atomic, never overwrites: write a temp file, then hard-link it into place."""
    path = Path(path)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("x", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(record, indent=2) + "\n")
        f.flush()
        os.fsync(f.fileno())
    try:
        os.link(tmp, path)
    except FileExistsError as exc:
        raise PreregError(f"{path} exists; a freeze is written once") from exc
    finally:
        tmp.unlink()
