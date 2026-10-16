"""Launch and supervise one phase of a task as its own process tree."""

from __future__ import annotations

import os
import re
import signal
import subprocess
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

IS_WINDOWS = os.name == "nt"
TAIL_BYTES = 16384


@dataclass
class JobResult:
    returncode: int
    reason: str  # ok | failed | timeout | stalled | killed
    oom: bool
    duration: float
    tail: str

    @property
    def ok(self) -> bool:
        return self.reason == "ok"


def _last_line(data: bytes) -> str:
    for part in reversed(re.split(rb"[\r\n]+", data)):
        s = part.decode("utf-8", "replace").strip()
        if s:
            return s[:300]
    return ""


class Job:
    """A child process tree with its output streamed to a log file.

    Output is read in raw chunks (not lines) so tqdm-style `\\r` progress bars still count as
    activity for the stall watchdog.
    """

    def __init__(
        self,
        argv: list[str],
        *,
        cwd: Path,
        env: dict[str, str],
        log_path: Path,
        header: str = "",
        timeout: float | None = None,
        stall_timeout: float | None = None,
        oom_re: re.Pattern | None = None,
        kill_grace: float = 20.0,
    ):
        self.argv = argv
        self.cwd = cwd
        self.env = env
        self.log_path = Path(log_path)
        self.header = header
        self.timeout = timeout
        self.stall_timeout = stall_timeout
        self.oom_re = oom_re
        self.kill_grace = kill_grace

        self.proc: subprocess.Popen | None = None
        self.pid: int | None = None
        self.started = 0.0
        self.last_output = 0.0
        self.last_line = ""
        self.oom = False
        self.result: JobResult | None = None
        self._tail = b""
        self._kill_reason: str | None = None
        self._kill_time = 0.0
        self._hard_killed = False
        self._log = None
        self._reader: threading.Thread | None = None

    # ------------------------------------------------------------------ lifecycle

    def start(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = open(self.log_path, "ab", buffering=0)
        self._log.write(self.header.encode("utf-8", "replace"))
        kw: dict = (
            {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if IS_WINDOWS else {"start_new_session": True}
        )
        try:
            self.proc = subprocess.Popen(
                self.argv, cwd=self.cwd, env=self.env, stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, **kw,
            )
        except Exception:
            self._log.close()
            raise
        self.pid = self.proc.pid
        self.started = self.last_output = time.time()
        self._reader = threading.Thread(target=self._read, name=f"reader-{self.pid}", daemon=True)
        self._reader.start()

    def _read(self) -> None:
        stream = self.proc.stdout
        while True:
            try:
                chunk = stream.read1(65536)
            except (OSError, ValueError):
                break
            if not chunk:
                break
            self.last_output = time.time()
            try:
                self._log.write(chunk)
            except (OSError, ValueError):
                pass
            window = self._tail[-1024:] + chunk  # patterns may straddle chunk boundaries
            if not self.oom and self.oom_re is not None and self.oom_re.search(window.decode("utf-8", "replace")):
                self.oom = True
            self._tail = (self._tail + chunk)[-TAIL_BYTES:]
            line = _last_line(self._tail)
            if line:
                self.last_line = line

    def poll(self, now: float | None = None) -> JobResult | None:
        """None while running. Enforces timeout / stall watchdog and escalates kills."""
        if self.result is not None:
            return self.result
        now = now or time.time()
        rc = self.proc.poll()
        if rc is None:
            if self._kill_reason is None:
                if self.timeout and now - self.started > self.timeout:
                    self.kill("timeout")
                elif self.stall_timeout and now - self.last_output > self.stall_timeout:
                    self.kill("stalled")
            elif not self._hard_killed and now - self._kill_time > self.kill_grace:
                self._signal(hard=True)
            return None
        return self._finish(rc, now)

    def kill(self, reason: str = "killed") -> None:
        if self._kill_reason is None:
            self._kill_reason = reason
            self._kill_time = time.time()
            self._signal(hard=False)

    def wait(self, timeout: float) -> JobResult | None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            res = self.poll()
            if res is not None:
                return res
            time.sleep(0.1)
        return self.poll()

    def tail_text(self, n: int = 25) -> str:
        lines = [ln for ln in re.split(r"[\r\n]+", self._tail.decode("utf-8", "replace")) if ln.strip()]
        return "\n".join(lines[-n:])

    # ------------------------------------------------------------------ internals

    def _finish(self, rc: int, now: float) -> JobResult:
        if not IS_WINDOWS:
            self._signal(hard=True)  # reap anything the leader left behind (would keep holding VRAM)
        if self._reader:
            self._reader.join(timeout=5)
        reason = self._kill_reason or ("ok" if rc == 0 else "failed")
        self.result = JobResult(rc, reason, self.oom, now - self.started, self.tail_text())
        try:
            self._log.write(
                f"\n### fdgx: exit {rc} ({reason}{', OOM detected' if self.oom else ''}) after "
                f"{self.result.duration:.0f}s at {time.strftime('%Y-%m-%d %H:%M:%S')}\n".encode()
            )
            self._log.close()
        except (OSError, ValueError):
            pass
        if self.proc.stdout:
            try:
                self.proc.stdout.close()
            except OSError:
                pass
        return self.result

    def _signal(self, hard: bool) -> None:
        if self.proc is None:
            return
        if IS_WINDOWS:
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(self.pid)], capture_output=True)
            self._hard_killed = True
            return
        try:
            os.killpg(self.pid, signal.SIGKILL if hard else signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
        if hard:
            self._hard_killed = True


# ---------------------------------------------------------------------- /proc helpers (Linux)


def _stat_fields(pid: int) -> list[str] | None:
    try:
        with open(f"/proc/{pid}/stat") as f:
            stat = f.read()
    except OSError:
        return None
    return stat[stat.rfind(")") + 2 :].split()  # fields from #3 (state) onward


def proc_start_time(pid: int | None) -> str | None:
    if pid is None:
        return None
    f = _stat_fields(pid)
    return f[19] if f and len(f) > 19 else None


def kill_stale_group(pid: int | None, start: str | None) -> bool:
    """Kill a process group left over from a previous scheduler, if it is really still that process."""
    if pid is None or IS_WINDOWS or start is None:
        return False
    if proc_start_time(pid) != start:
        return False
    try:
        os.killpg(pid, signal.SIGKILL)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def session_members() -> dict[int, set[int]]:
    """session id -> pids. Our jobs are session leaders, so this maps a job pid to its whole tree."""
    res: dict[int, set[int]] = defaultdict(set)
    if IS_WINDOWS or not os.path.isdir("/proc"):
        return res
    for e in os.listdir("/proc"):
        if e.isdigit():
            f = _stat_fields(int(e))
            if f and len(f) > 3:
                res[int(f[3])].add(int(e))
    return res


# ---------------------------------------------------------------------- resources & files


def available_cpus() -> float:
    """CPUs this process may really use: cgroup quota, else CPU affinity, else the core count."""
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()[:2]
        if quota != "max":
            return max(1.0, int(quota) / int(period))
    except (OSError, ValueError):
        pass
    try:
        q = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
        p = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
        if q > 0 and p > 0:
            return max(1.0, q / p)
    except (OSError, ValueError):
        pass
    if hasattr(os, "sched_getaffinity"):
        return float(len(os.sched_getaffinity(0)))
    return float(os.cpu_count() or 1)


def clean_paths(workdir: Path, patterns: list[str]) -> list[str]:
    """Delete files/trees matching `patterns` (relative to workdir, globs allowed). Returns what was deleted.

    Raises ValueError for anything that resolves outside the workdir, or to the workdir itself.
    """
    import glob
    import shutil

    root = Path(workdir).resolve()
    targets: list[Path] = []
    for pat in patterns:
        matches = glob.glob(pat, root_dir=root, recursive=True) if any(c in pat for c in "*?[") else [pat]
        for rel in matches:
            p = (root / rel).resolve()
            if p == root or root not in p.parents:
                raise ValueError(f"clean path {pat!r} -> {p} is not inside the workdir {root}")
            targets.append(root / rel)
    deleted = []
    for p in targets:
        if p.is_dir() and not p.is_symlink():
            shutil.rmtree(p)
        elif p.exists() or p.is_symlink():
            p.unlink()
        else:
            continue
        deleted.append(p.relative_to(root).as_posix())
    return deleted


def killed_by_sigkill(returncode: int) -> bool:
    """SIGKILL directly (-9) or through a shell / uv wrapper (128 + 9)."""
    return not IS_WINDOWS and returncode in (-9, 137)
