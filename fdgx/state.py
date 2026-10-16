"""Per-task state on disk: one small JSON file per task, always written atomically."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Iterator

# phase = the next phase to execute: setup -> run -> epilogue -> done, or failed.
ACTIVE_PHASES = ("setup", "run", "epilogue")
HISTORY_LEN = 40


@dataclass
class TaskState:
    id: str
    phase: str = "setup"
    attempts: int = 0  # failures of the current phase, counted against retry.max_attempts
    oom_level: int = 0  # how many on_oom steps are applied
    interruptions: int = 0  # times a phase was cut short by the scheduler/instance dying (not failures)
    running: bool = False
    held: bool = False  # `fdgx hold`: never start a phase until released
    pid: int | None = None
    pid_start: str | None = None
    gpus: list[int] | None = None
    host: str | None = None  # where the last phase was launched (a new container has a new hostname)
    failed_phase: str | None = None
    next_at: float = 0.0
    last_error: str | None = None
    updated_at: float = 0.0
    history: list[dict] = field(default_factory=list)

    def event(self, msg: str) -> None:
        self.history.append({"t": round(time.time(), 1), "msg": msg})
        del self.history[:-HISTORY_LEN]


def safe_name(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._") or "task"
    if safe != name:
        safe = f"{safe[:120]}-{hashlib.sha1(name.encode()).hexdigest()[:8]}"
    return safe


def atomic_write(path: Path, data: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


class StateStore:
    def __init__(self, state_dir: Path):
        self.root = Path(state_dir) / "tasks"

    def path(self, project: str, name: str) -> Path:
        return self.root / project / f"{safe_name(name)}.json"

    def load(self, project: str, name: str) -> TaskState | None:
        return self._read(self.path(project, name))

    def save(self, project: str, name: str, st: TaskState) -> None:
        st.updated_at = time.time()
        atomic_write(self.path(project, name), json.dumps(asdict(st), indent=1))

    def delete(self, project: str, name: str) -> None:
        self.path(project, name).unlink(missing_ok=True)

    def all(self) -> Iterator[tuple[Path, TaskState]]:
        if not self.root.exists():
            return
        for p in sorted(self.root.glob("*/*.json")):
            st = self._read(p)
            if st is not None:
                yield p, st

    def save_path(self, path: Path, st: TaskState) -> None:
        st.updated_at = time.time()
        atomic_write(path, json.dumps(asdict(st), indent=1))

    @staticmethod
    def _read(path: Path) -> TaskState | None:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, ValueError):
            # Corrupt file (should not happen with atomic writes): keep a copy and start the task over.
            try:
                os.replace(path, path.with_suffix(".corrupt"))
            except OSError:
                pass
            return None
        known = {f.name for f in fields(TaskState)}
        return TaskState(**{k: v for k, v in raw.items() if k in known})


LOCK_OFFSET = 1 << 16  # Windows locks a byte range: keep it away from the readable holder info


class FileLock:
    """Exclusive, non-blocking lock released automatically by the OS when the process dies.

    The file's content tells who holds it (pid, host, start time).
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self._fd: int | None = None

    def acquire(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            if os.name == "nt":
                import msvcrt

                os.lseek(fd, LOCK_OFFSET, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)
            return False
        import socket

        info = {"pid": os.getpid(), "host": socket.gethostname(), "started": time.strftime("%Y-%m-%d %H:%M:%S")}
        os.ftruncate(fd, 0)
        os.lseek(fd, 0, os.SEEK_SET)
        os.write(fd, json.dumps(info).encode())
        self._fd = fd
        return True

    def release(self) -> None:
        if self._fd is None:
            return
        try:
            if os.name == "nt":
                import msvcrt

                os.lseek(self._fd, LOCK_OFFSET, os.SEEK_SET)
                msvcrt.locking(self._fd, msvcrt.LK_UNLCK, 1)
        except OSError:
            pass
        os.close(self._fd)
        self._fd = None

    def holder(self) -> str:
        """Who holds (or last held) the lock, for error messages."""
        try:
            info = json.loads(self.path.read_text(encoding="utf-8")[:1000] or "{}")
            return f"pid {info.get('pid')} on {info.get('host')}, started {info.get('started')}"
        except (OSError, ValueError):
            return "unknown process"

    def held_by_other(self) -> bool:
        """True if another live process holds the lock."""
        if self.acquire():
            self.release()
            return False
        return True
