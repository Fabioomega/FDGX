"""Live control of a running scheduler through files in `state_dir/control/`.

The CLI drops a request file; the scheduler handles it at its next loop (~poll_seconds), deletes it and
writes a reply. While a scheduler runs it is the only writer of task state; without one, the CLI edits
the state files directly (`apply_offline`).
"""

from __future__ import annotations

import fnmatch
import json
import os
import time
import uuid
from pathlib import Path

from .state import StateStore, TaskState, atomic_write

OPS = ("hold", "release", "retry", "reset", "stop", "reload")
REPLY_TTL = 3600.0


def _dirs(state_dir: Path) -> tuple[Path, Path]:
    root = Path(state_dir) / "control"
    return root / "requests", root / "replies"


def match(task_id: str, patterns: list[str]) -> bool:
    """`proj/name` globs match the full id; patterns without '/' match the name in any project.
    No patterns match everything (callers decide whether that is allowed)."""
    if not patterns:
        return True
    name = task_id.split("/", 1)[-1]
    return any(fnmatch.fnmatchcase(task_id if "/" in p else name, p) for p in patterns)


def submit(state_dir: Path, op: str, patterns: list[str] | None = None, **opts) -> str:
    if op not in OPS:
        raise ValueError(f"unknown control op {op!r}")
    req_dir, _ = _dirs(state_dir)
    rid = uuid.uuid4().hex[:12]
    req = {"id": rid, "op": op, "patterns": list(patterns or []), "time": time.time(), **opts}
    atomic_write(req_dir / f"{time.time():.6f}-{rid}.json", json.dumps(req))
    return rid


def pending(state_dir: Path) -> list[tuple[Path, dict]]:
    req_dir, _ = _dirs(state_dir)
    if not req_dir.is_dir():
        return []
    out = []
    for p in sorted(req_dir.glob("*.json")):
        try:
            out.append((p, json.loads(p.read_text(encoding="utf-8"))))
        except (OSError, ValueError):
            p.unlink(missing_ok=True)  # half-written or garbage
    return out


def reply(state_dir: Path, rid: str, ok: bool, lines: list[str]) -> None:
    _, rep_dir = _dirs(state_dir)
    atomic_write(rep_dir / f"{rid}.json", json.dumps({"ok": ok, "lines": lines, "time": time.time()}))


def wait_reply(state_dir: Path, rid: str, timeout: float) -> dict | None:
    _, rep_dir = _dirs(state_dir)
    path = rep_dir / f"{rid}.json"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            time.sleep(0.1)
            continue
        path.unlink(missing_ok=True)
        return data
    return None


def discard_stale(state_dir: Path) -> list[dict]:
    """At scheduler start: drop requests meant for a previous scheduler and old replies."""
    dropped = []
    for path, req in pending(state_dir):
        path.unlink(missing_ok=True)
        dropped.append(req)
    _, rep_dir = _dirs(state_dir)
    if rep_dir.is_dir():
        for p in rep_dir.glob("*.json"):
            try:
                if time.time() - os.path.getmtime(p) > REPLY_TTL:
                    p.unlink()
            except OSError:
                pass
    return dropped


def retry_state(st: TaskState, reset_oom: bool = False) -> None:
    st.phase = st.failed_phase or "setup"
    st.attempts = 0
    st.next_at = 0.0
    if reset_oom:
        st.oom_level = 0
    st.event("retry requested")


def apply_offline(store: StateStore, op: str, patterns: list[str], *, kill: bool = False,
                  reset_oom: bool = False) -> list[str]:
    """hold / release / retry / reset on the state files when no scheduler is running."""
    lines = []
    for path, st in store.all():
        if not match(st.id, patterns):
            continue
        if op == "hold" and st.phase != "done" and not st.held:
            st.held = True
            st.event("held")
            lines.append(f"held {st.id}")
        elif op == "release" and st.held:
            st.held = False
            st.event("released")
            lines.append(f"released {st.id}")
        elif op == "retry" and st.phase == "failed":
            retry_state(st, reset_oom)
            lines.append(f"retry {st.id} from {st.phase}")
        elif op == "reset":
            path.unlink(missing_ok=True)
            lines.append(f"reset {st.id}: starts over from setup (done_if is checked again)")
            continue
        else:
            continue
        store.save_path(path, st)
    if op == "retry":
        for path in sorted((store.root.parent / "projects").glob("*.json")):
            try:
                epi = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            name = path.stem
            if str(epi.get("result", "")).startswith("failed") and matches_project(name, patterns):
                atomic_write(path, json.dumps({"fingerprint": None, "result": "retry requested"}))
                lines.append(f"retry project {name} epilogue")
    return lines or [f"nothing to {op}" + (f" matching {' '.join(patterns)}" if patterns else "")]


def matches_project(project: str, patterns: list[str]) -> bool:
    """Project-level targets (a failed project epilogue): no patterns, or a `proj/...` pattern matching
    `proj/` — e.g. 'proj/*' or '*'. A specific task pattern like 'proj/train-a' doesn't."""
    return not patterns or any(fnmatch.fnmatchcase(f"{project}/", p if "/" in p else f"*/{p}") for p in patterns)
