"""The scheduling loop: project setup -> item discovery -> setup / run (GPU) / epilogue per task,
dependencies between tasks, project epilogues and hot reload of the config."""

from __future__ import annotations

import hashlib
import json
import os
import re
import signal
import socket
import subprocess
import threading
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any, Callable

from . import control, deps, gpulog
from .allocator import Request, allocate
from .config import Config, ConfigError, Project, TaskSpec, expand_tasks, render
from .envs import EnvManager, clean_base_env, default_shell, make_env_manager, thread_env
from .gpu import GpuProbe, Snapshot
from .heartbeat import Heartbeat
from .loader import load
from .runner import (Job, JobResult, available_cpus, clean_paths, kill_stale_group, killed_by_sigkill,
                     proc_start_time, session_members)
from .state import FileLock, StateStore, TaskState, atomic_write, safe_name

NEXT_PHASE = {"setup": "run", "run": "epilogue", "epilogue": "done"}

# Project fields that only take effect at the next scheduler start (re-syncing a venv under live jobs is unsafe).
RESTART_ONLY_PROJECT = ("workdir", "env_manager", "shell_prefix", "uv", "setup")
GPULOG_EVERY = 60.0
RESTART_ONLY_SCHEDULER = ("state_dir", "heartbeat_seconds", "shell", "uv", "nvidia_smi", "keepalive_cmd", "cpu_only",
                          "keepalive_seconds")


def fmt_dur(sec: float) -> str:
    sec = int(max(0, sec))
    h, r = divmod(sec, 3600)
    m, s = divmod(r, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def _first_line(s: str | None, n: int = 160) -> str:
    return (s or "").strip().splitlines()[0][:n] if (s or "").strip() else ""


class Task:
    def __init__(self, project: Project, spec: TaskSpec):
        self.project = project
        self.name = spec.name
        self.id = f"{project.name}/{spec.name}"
        self.update(project, spec)
        self.state = TaskState(id=self.id)
        self.checked = False  # done_if evaluated during this scheduler session
        self.live = True  # still produced by the current config
        self.cond: tuple[str, str] = (deps.OK, "")  # dependency condition (derived each tick)

    def update(self, project: Project, spec: TaskSpec) -> None:
        self.project = project
        self.tmpl = spec.tmpl
        self.item = spec.item
        self.needs = spec.needs

    @property
    def order(self) -> tuple[int, int, int]:
        return (-self.tmpl.priority, self.tmpl.index, self.item.index if self.item else 0)

    @property
    def disabled(self) -> bool:
        return not (self.tmpl.enabled and self.project.enabled)

    @property
    def parked(self) -> bool:
        """Held or disabled: never starts a phase on its own."""
        return self.state.held or self.disabled


class ProjectRT:
    """Runtime status of a project: `uv sync` (step 0) and project setup (step 1) run once per session."""

    def __init__(self, cfg: Project, env: EnvManager, epilogue_state: dict):
        self.cfg = cfg
        self.env = env
        self.status = "pending"  # pending | ready | failed
        self.removed = False  # no longer in the config: nothing new starts
        self.step = 0
        self.attempts = 0
        self.next_at = 0.0
        self.last_scan: float | None = None
        self.error: str | None = None
        self.epilogue = epilogue_state  # persisted: fingerprint + result of the last project epilogue
        self.epilogue_attempts = 0
        self.epilogue_next_at = 0.0
        self.epilogue_running_fp: str | None = None


@dataclass
class Active:
    key: str
    label: str
    phase: str
    job: Job
    gpus: list[int] = field(default_factory=list)
    vram_gb: float = 0.0
    task: Task | None = None
    project: ProjectRT | None = None
    cpus: float = 1.0
    ram_gb: float = 0.0
    vram_file: Any = None  # Path the job may write its VRAM usage to ($FDGX_VRAM_FILE)

    @property
    def uses_gpu(self) -> bool:
        return self.task is not None and self.phase == "run" and bool(self.gpus)

    @property
    def counts_as_cpu_phase(self) -> bool:
        """Setup / epilogue / project phases are limited by max_cpu_phases; run phases are not."""
        return not (self.task is not None and self.phase == "run")


class Scheduler:
    def __init__(self, cfg: Config, *, stay_alive: bool = False, out: Callable[[str], None] | None = None,
                 load_errors: list[str] | None = None, load_warnings: list[str] | None = None):
        self.cfg = cfg
        self.sc = cfg.scheduler
        self.stay_alive = stay_alive
        self._out = out or (lambda line: print(line, flush=True))
        self.store = StateStore(self.sc.state_dir)
        self.probe = GpuProbe(self.sc.nvidia_smi)
        self.oom_re = self._compile_oom(self.sc)
        self.base_env = clean_base_env()
        self.shell = self.sc.shell or default_shell()
        self.sc.state_dir.mkdir(parents=True, exist_ok=True)
        self.projects = {p.name: self._new_project(p) for p in cfg.projects}
        self.config_errors: set[str] = set(load_errors or [])
        self.config_warnings: set[str] = set(load_warnings or [])
        self._warned: set[str] = set()
        self._last_reload = 0.0
        self._last_status_write = 0.0
        self.tasks: dict[str, Task] = {}
        self.active: dict[str, Active] = {}
        self.snapshot: Snapshot | None = None
        self.counts: dict[str, int] = {}
        self.activity = "starting"
        self.waiting_since: dict[str, float] = {}
        self.stop_requested = False
        self._last_probe = 0.0
        self._last_probe_error: str | None = None
        self.cpu_budget = self.sc.cpus or available_cpus()
        self.hostname = socket.gethostname()
        self.last_tick = time.time()
        self.gpulog: gpulog.GpuLog | None = None
        self._gpulog_at = 0.0
        self._gpulog_warned: set[str] = set()
        self._last_gpu_job = 0.0
        self._wake = threading.Event()
        self._emit_lock = threading.Lock()
        self._log_file = open(self.sc.state_dir / "fdgx.log", "a", encoding="utf-8")

    @staticmethod
    def _compile_oom(sc) -> re.Pattern:
        return re.compile("|".join(f"(?:{p})" for p in sc.oom_patterns), re.IGNORECASE)

    def _epilogue_state_path(self, project: str):
        return self.sc.state_dir / "projects" / f"{safe_name(project)}.json"

    def _new_project(self, p: Project) -> ProjectRT:
        try:
            epi = json.loads(self._epilogue_state_path(p.name).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            epi = {}
        return ProjectRT(p, make_env_manager(p, self.sc), epi)

    def _warn_once(self, msg: str) -> None:
        if msg not in self._warned:
            self._warned.add(msg)
            self.event(f"warning: {msg}")

    # ================================================================== output

    def emit(self, line: str, *, log: bool = False) -> None:
        with self._emit_lock:
            try:
                self._out(line)
            except Exception:
                pass
            if log:
                try:
                    self._log_file.write(line + "\n")
                    self._log_file.flush()
                except (OSError, ValueError):
                    pass

    def event(self, msg: str) -> None:
        self.emit(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", log=True)

    def status_lines(self) -> list[str]:
        now = time.time()
        c = self.counts
        active = list(self.active.values())
        n_gpu = sum(1 for a in active if a.uses_gpu)
        head = (
            f"[{datetime.now():%Y-%m-%d %H:%M:%S}] fdgx alive | jobs {n_gpu} gpu + {len(active) - n_gpu} cpu"
            f" | queued {c.get('queued', 0)} waiting {c.get('waiting', 0)} blocked {c.get('blocked', 0)}"
            f" backoff {c.get('backoff', 0)} done {c.get('done', 0)} failed {c.get('failed', 0)}"
            + "".join(f" {k} {c[k]}" for k in ("held", "disabled") if c.get(k))
        )
        head += f" | cpus {sum(a.cpus for a in active):g}/{self.cpu_budget:g}"
        if self.sc.ram_gb is not None:
            head += f" ram {sum(a.ram_gb for a in active):g}/{self.sc.ram_gb:g} GB"
        snap = self.snapshot
        if snap is not None:
            if snap.error:
                head += f" | gpu probe error: {snap.error[:80]}"
            else:
                allowed = self.sc.gpus if self.sc.gpus is not None else sorted(snap.gpus)
                head += " | free GB " + " ".join(
                    f"{g}:{snap.gpus[g].free_gb:.0f}/{snap.gpus[g].total_gb:.0f}" for g in allowed if g in snap.gpus
                )
        if self.activity:
            head += f" | {self.activity}"
        late = now - self.last_tick
        if late > max(60.0, 10 * self.sc.poll_seconds):
            head += f" | LOOP STALLED {fmt_dur(late)}"  # main loop stuck (e.g. on a hung filesystem)
        lines = [head]
        for a in sorted(active, key=lambda a: a.label):
            gpus = ",".join(map(str, a.gpus)) if a.uses_gpu else "-"
            lines.append(
                f"    {a.label:<40} {a.phase:<8} gpus {gpus:<8} {fmt_dur(now - a.job.started):>7}"
                f" | {a.job.last_line[:140]}"
            )
        return lines

    # ================================================================== main loop

    def run(self) -> int:
        lock = FileLock(self.sc.state_dir / "fdgx.lock")
        if not lock.acquire():
            self.event(f"another fdgx scheduler holds {lock.path} ({lock.holder()}); exiting")
            return 3
        self._install_signals()
        heartbeat = Heartbeat(
            self.status_lines,
            self.emit,
            self.sc.heartbeat_seconds,
            keepalive_argv=[*self.shell, self.sc.keepalive_cmd] if self.sc.keepalive_cmd else None,
            keepalive_seconds=self.sc.keepalive_seconds,
        )
        try:
            self.event(f"fdgx starting: {self.cfg.path} (state in {self.sc.state_dir})")
            for e in sorted(self.config_errors):
                self.event(f"config error (source skipped): {e}")
            for w in sorted(self.config_warnings):
                self.event(f"config warning: {w}")
            self._last_reload = time.time()
            for req in control.discard_stale(self.sc.state_dir):
                self.event(f"discarded control request {req.get('op')} {' '.join(req.get('patterns', []))} "
                           "left over from a previous scheduler")
            self._recover()
            heartbeat.start()
            errors = 0
            while not self.stop_requested:
                try:
                    self.tick(time.time())
                    errors = 0
                except Exception:
                    # Dying would leave the instance idle (and killed); log it and keep going.
                    errors += 1
                    self.event(f"internal error ({errors} in a row):\n{traceback.format_exc()}")
                    if errors >= 20:
                        raise
                if self._finished() and not self.stay_alive:
                    break
                self._wake.wait(self.sc.poll_seconds)
                self._wake.clear()
        finally:
            self._shutdown()
            heartbeat.stop()
            lock.release()
        return self._summary()

    def request_stop(self) -> None:
        self.stop_requested = True
        self._wake.set()

    def tick(self, now: float) -> None:
        self._process_control()
        self._poll_active(now)
        if now - self._last_reload >= self.sc.rescan_seconds:
            self._last_reload = now
            self._reload()
        for prt in list(self.projects.values()):
            if prt.removed or not prt.cfg.enabled:
                continue
            self._advance_project(prt, now)
            if prt.status == "ready" and (prt.last_scan is None or now - prt.last_scan >= self.sc.rescan_seconds):
                self._scan(prt)
        self._update_conditions()
        self._maybe_probe(now)
        self._launch_cpu(now)  # first: tasks whose setup is skipped can get a GPU in this same tick
        self._update_conditions()
        self._launch_runs(now)
        self._launch_project_epilogues(now)
        self._check_gpulog(now)
        self._publish_counts(now)
        self.last_tick = time.time()

    def _install_signals(self) -> None:
        def handler(signum, _frame):
            if self.stop_requested:
                raise KeyboardInterrupt
            self.event(f"signal {signum}: stopping (jobs will resume on next start)")
            self.request_stop()

        sigs = [signal.SIGINT, signal.SIGTERM] + ([signal.SIGHUP] if hasattr(signal, "SIGHUP") else [])
        for s in sigs:
            try:
                signal.signal(s, handler)
            except (ValueError, OSError):  # not the main thread (tests)
                pass

    def _recover(self) -> None:
        """Tasks marked running belong to a scheduler that died: kill leftovers, resume at the same phase."""
        n = 0
        for path, st in self.store.all():
            if not st.running:
                continue
            if kill_stale_group(st.pid, st.pid_start):
                self.event(f"killed leftover process group {st.pid} of {st.id}")
            elif st.host and st.host != self.hostname:
                self.event(f"warning: {st.id} was last running on {st.host} (another container): a ghost copy "
                           "may still be running there and writing its outputs" + self._gpulog_hint())
            st.running = False
            st.pid = st.pid_start = None
            st.interruptions += 1
            st.event(f"interrupted in {st.phase}; resuming")
            self.store.save_path(path, st)
            n += 1
        if n:
            self.event(f"resuming {n} interrupted task(s) at the phase they were in")

    def _finished(self) -> bool:
        """Nothing running and nothing that could still start (waiting/blocked tasks can't move on their own)."""
        if self.active:
            return False
        if any(p.status == "pending" and not p.removed and p.cfg.enabled for p in self.projects.values()):
            return False
        self._update_conditions()  # phases may have advanced since the tick computed them
        if any(self._epilogue_fingerprint(p) is not None for p in self.projects.values()):
            return False
        for t in self.tasks.values():
            if not t.live or t.state.phase in ("done", "failed") or t.parked:
                continue
            if t.state.phase == "setup" and t.cond[0] != deps.OK:
                continue
            return False
        return True

    def _interrupted(self, t: Task, phase: str, why: str) -> None:
        """A phase cut short on purpose (shutdown, hold --kill): not a failure, resumes at the same phase."""
        st = t.state
        st.running = False
        st.pid = st.pid_start = None
        st.interruptions += 1
        st.event(f"interrupted in {phase}: {why}")
        self._save(t)

    def _shutdown(self) -> None:
        if self.active:
            self.event(f"stopping {len(self.active)} job(s)")
        for a in self.active.values():
            a.job.kill("killed")
        deadline = time.time() + self.sc.kill_grace + 5
        for a in list(self.active.values()):
            a.job.wait(max(0.0, deadline - time.time()))
            if a.task:
                self._interrupted(a.task, a.phase, "scheduler shutdown")
        self.active.clear()

    def _summary(self) -> int:
        self._publish_counts(time.time())
        c = self.counts
        live = [t for t in self.tasks.values() if t.live]
        failed = [t for t in live if t.state.phase == "failed"]
        not_started = [t for t in live if t.state.phase == "setup" and not t.parked and t.cond[0] != deps.OK]
        blocked = [t for t in not_started if t.cond[0] == deps.BLOCKED]
        parked = [t for t in live if t.parked and t.state.phase not in ("done", "failed")]
        bad_projects = [p for p in self.projects.values() if p.status == "failed" and not p.removed]
        # The last project epilogue run (it persists: an unchanged task set doesn't re-run it).
        bad_epilogues = [p for p in self.projects.values()
                         if not p.removed and p.cfg.enabled and p.cfg.epilogue.run
                         and str(p.epilogue.get("result", "")).startswith("failed")]
        self.event(
            f"fdgx exiting: done {c.get('done', 0)}, failed {len(failed)}, blocked {len(blocked)}, "
            f"waiting {c.get('waiting', 0)}, held {c.get('held', 0)}, disabled {c.get('disabled', 0)}, "
            f"unfinished {c.get('queued', 0) + c.get('backoff', 0)}"
        )
        for p in bad_projects:
            self.event(f"  project {p.cfg.name} FAILED setup: {_first_line(p.error)}")
        for p in bad_epilogues:
            self.event(f"  project {p.cfg.name} epilogue FAILED: {p.epilogue.get('result')}")
        for t in sorted(failed, key=lambda t: t.order):
            self.event(f"  FAILED {t.id} [{t.state.failed_phase}]: {_first_line(t.state.last_error)}")
        for t in sorted(not_started, key=lambda t: t.order):
            self.event(f"  {t.cond[0].upper()} {t.id}: {t.cond[1]}")
        for t in sorted(parked, key=lambda t: t.order):
            self.event(f"  {'HELD' if t.state.held else 'DISABLED'} {t.id} [{t.state.phase}]")
        self._write_status(time.time())
        self._log_file.close()
        if self.stop_requested:
            return 130
        # Parked work (held / disabled) and tasks waiting on it are intentional: not an error.
        return 2 if failed or blocked or bad_projects or bad_epilogues else 0

    def _publish_counts(self, now: float) -> None:
        c: dict[str, int] = defaultdict(int)
        for t in self.tasks.values():
            if not t.live:
                continue
            c[self._condition_label(t, now)] += 1
        self.counts = dict(c)
        if now - self._last_status_write >= 5:
            self._write_status(now)
        pending = [p.cfg.name for p in self.projects.values() if p.status == "pending"]
        if pending:
            self.activity = f"setting up {', '.join(pending)}"
        elif self._last_probe_error:
            self.activity = "waiting for nvidia-smi"
        elif c["queued"] and not any(a.uses_gpu for a in self.active.values()):
            self.activity = "waiting for free GPUs"
        else:
            self.activity = ""

    def _condition_label(self, t: Task, now: float) -> str:
        ph = t.state.phase
        if ph in ("done", "failed"):
            return ph
        if t.id in self.active:
            return "active"
        if t.state.held:
            return "held"
        if t.disabled:
            return "disabled"
        if ph == "setup" and t.cond[0] != deps.OK:
            return t.cond[0]
        if t.state.next_at > now:
            return "backoff"
        return "queued"

    @staticmethod
    def _project_status(p: ProjectRT) -> str:
        if p.removed:
            return "removed"
        if not p.cfg.enabled:
            return "disabled"
        return p.status

    def _write_status(self, now: float) -> None:
        """Snapshot for `fdgx status` (derived conditions are not in the task state files)."""
        self._last_status_write = now
        tasks = {}
        for t in self.tasks.values():
            if not t.live:
                continue
            a = self.active.get(t.id)
            tasks[t.id] = {
                "phase": t.state.phase, "condition": self._condition_label(t, now), "reason": t.cond[1],
                "active": a.phase if a else None, "gpus": a.gpus if a else None,
            }
        data = {
            "time": now, "pid": os.getpid(), "counts": self.counts,
            "projects": {n: self._project_status(p) for n, p in self.projects.items()},
            "config_errors": sorted(self.config_errors), "tasks": tasks,
        }
        try:
            atomic_write(self.sc.state_dir / "status.json", json.dumps(data, indent=1))
        except OSError as e:
            self._warn_once(f"cannot write status.json: {e}")

    # ================================================================== environment helpers

    def _env(self, project: Project, env: dict[str, str], ctx: dict[str, Any], gpus: list[int] | None,
             extra: dict[str, str] | None = None, cpus: float | None = None) -> dict[str, str]:
        e = dict(self.base_env)
        e.setdefault("PYTHONUNBUFFERED", "1")
        if cpus is not None:  # thread pools sized to the CPU budget, unless the config says otherwise
            e.update(thread_env(cpus))
        for k, v in env.items():
            e[k] = render(v, ctx, f"env {k}")
        e["FDGX_STATE_DIR"] = str(self.sc.state_dir)
        e["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # make indices match nvidia-smi
        e["CUDA_VISIBLE_DEVICES"] = "-1" if self.sc.cpu_only else (",".join(map(str, gpus)) if gpus else "")
        e.update(extra or {})
        return e

    @staticmethod
    def _project_phase_cpus(ph) -> float:
        return ph.cpus if ph.cpus is not None else 1.0

    def _fits(self, cpus: float, ram_gb: float) -> bool:
        """Room in the CPU / RAM budget. An oversized request still runs when nothing else does."""
        active = list(self.active.values())
        used_c = sum(a.cpus for a in active)
        if used_c > 0 and used_c + cpus > self.cpu_budget + 1e-9:
            return False
        if self.sc.ram_gb is not None and ram_gb > 0:
            used_r = sum(a.ram_gb for a in active)
            if used_r > 0 and used_r + ram_gb > self.sc.ram_gb + 1e-9:
                return False
        return True

    def _project_ctx(self, project: Project) -> dict[str, Any]:
        return {**project.vars, "project": project.name, "workdir": str(project.workdir)}

    def _task_ctx(self, t: Task, vars_: dict, phase: str, gpus: list[int]) -> dict[str, Any]:
        ctx = dict(vars_)
        ctx.update(
            item=t.item, project=t.project.name, workdir=str(t.project.workdir), task=t.name, phase=phase,
            gpus=",".join(map(str, gpus)), num_gpus=len(gpus), attempt=t.state.attempts + 1,
            oom_level=t.state.oom_level,
        )
        return ctx

    def _check(self, project: Project, cmd: str, ctx: dict[str, Any], env: dict[str, str]) -> bool:
        """Run a done_if / skip_if check in a plain shell inside the workdir. Exit 0 = true."""
        try:
            r = subprocess.run(
                [*self.shell, cmd], cwd=project.workdir, env=self._env(project, env, ctx, None, cpus=1),
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                timeout=self.sc.check_timeout,
            )
            return r.returncode == 0
        except subprocess.TimeoutExpired:
            self.event(f"warning: check timed out after {self.sc.check_timeout:.0f}s: {cmd}")
        except OSError as e:
            self.event(f"warning: check could not run ({e}): {cmd}")
        return False

    def _log_dir(self, p: Project):
        """Where this project's job logs go: `log_dir` (relative to the workdir), else state_dir/logs/<project>."""
        default = self.sc.state_dir / "logs" / p.name
        if not p.log_dir:
            return default
        try:
            return p.workdir / os.path.expanduser(render(p.log_dir, self._project_ctx(p), f"{p.name}.log_dir"))
        except ConfigError as e:
            self._warn_once(f"{e}; logging to {default}")
            return default

    def _log_path(self, p: Project, name: str, phase: str):
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        # "_project" (setup / epilogue logs) can't clash: safe_name() never keeps a leading underscore.
        sub = name if name == "_project" else safe_name(name)
        return self._log_dir(p) / sub / f"{stamp}-{phase}.log"

    def _save(self, t: Task) -> None:
        self.store.save(t.project.name, t.name, t.state)

    # ================================================================== projects

    def _advance_project(self, prt: ProjectRT, now: float) -> None:
        key = f"{prt.cfg.name}::project"
        if (prt.removed or not prt.cfg.enabled or prt.status != "pending" or key in self.active
                or now < prt.next_at):
            return
        p = prt.cfg
        cpus = self._project_phase_cpus(p.setup)
        if self._cpu_slots() <= 0 or not self._fits(cpus, 0):
            return
        if not p.workdir.is_dir():
            prt.status, prt.error = "failed", f"workdir {p.workdir} does not exist"
            self.event(f"project {p.name} FAILED: {prt.error}")
            return
        ctx = self._project_ctx(p)
        try:
            while prt.step <= 1:
                if prt.step == 0:
                    argv = prt.env.sync_argv()
                    label = "uv sync"
                else:
                    if p.setup.run is None or (p.setup.skip_if and self._check(
                        p, render(p.setup.skip_if, ctx, f"{p.name}.setup.skip_if"), ctx, p.env)):
                        argv = None
                    else:
                        argv = prt.env.wrap(render(p.setup.run, ctx, f"{p.name}.setup.run"))
                    label = "setup"
                if argv is None:
                    prt.step += 1
                    continue
                log_path = self._log_path(p, "_project", label.replace(" ", "-"))
                job = Job(
                    argv, cwd=p.workdir,
                    env=self._env(p, p.env, ctx, None, {"FDGX_LOG_FILE": str(log_path)}, cpus=cpus),
                    log_path=log_path,
                    header=f"### fdgx: {p.name} {label}: {argv}\n",
                    timeout=p.setup.timeout, stall_timeout=p.setup.stall_timeout,
                    kill_grace=self.sc.kill_grace,
                )
                job.start()
                self.active[key] = Active(key, f"{p.name} [project]", label, job, project=prt, cpus=cpus)
                self.event(f"> {p.name}: project {label}")
                return
        except (ConfigError, OSError) as e:
            prt.status, prt.error = "failed", str(e)
            self.event(f"project {p.name} FAILED: {e}")
            return
        prt.status = "ready"
        self.event(f"project {p.name} ready")

    def _project_done(self, prt: ProjectRT, a: Active, res: JobResult, now: float) -> None:
        p = prt.cfg
        if a.phase == "epilogue":
            self._project_epilogue_done(prt, a, res, now)
            return
        if res.ok:
            prt.step += 1
            prt.attempts = 0
            self.event(f"  {p.name}: project {a.phase} ok ({fmt_dur(res.duration)})")
            self._advance_project(prt, now)
            return
        prt.attempts += 1
        prt.error = f"{a.phase} {res.reason} rc={res.returncode}\n{res.tail}"
        if prt.attempts >= p.retry.max_attempts:
            prt.status = "failed"
            self.event(f"project {p.name} FAILED in {a.phase} ({res.reason}); see {a.job.log_path}")
        else:
            prt.next_at = now + p.retry.delay(prt.attempts)
            self.event(
                f"project {p.name}: {a.phase} {res.reason} (attempt {prt.attempts}/{p.retry.max_attempts}),"
                f" retrying in {fmt_dur(prt.next_at - now)}; see {a.job.log_path}"
            )

    # ================================================================== discovery

    def _scan(self, prt: ProjectRT) -> None:
        """Expand every template into tasks; adopt state from disk; mark tasks the config no longer makes."""
        p = prt.cfg
        new = 0
        produced: set[str] = set()
        unlisted: set[str] = set()  # templates whose items could not be listed: keep their tasks as they are
        for tmpl in p.tasks:
            self.activity = f"scanning items of {p.name}/{tmpl.name}"
            try:
                specs, warnings = expand_tasks(p, tmpl, lambda cmd: self._item_command(prt, cmd))
            except (ConfigError, OSError, subprocess.SubprocessError) as e:
                self._warn_once(f"cannot list tasks for {p.name}/{tmpl.name}: {e}")
                unlisted.add(tmpl.name)
                continue
            for w in warnings:
                self._warn_once(w)
            for spec in specs:
                tid = f"{p.name}/{spec.name}"
                produced.add(tid)
                t = self.tasks.get(tid)
                if t is None:
                    t = self.tasks[tid] = Task(p, spec)
                    new += 1
                else:
                    t.update(p, spec)
                    if tid in self.active:
                        continue
                self._load_or_discover(t)
        for t in self.tasks.values():
            if t.project.name == p.name and t.tmpl.name not in unlisted:
                if t.live and t.id not in produced:
                    self.event(f"{t.id}: no longer in the config; it won't be started again")
                t.live = t.id in produced
        prt.last_scan = time.time()
        if new:
            self.event(f"{p.name}: {new} new task(s), {len(self.tasks)} known in total")

    def _item_command(self, prt: ProjectRT, cmd: str) -> str:
        ctx = self._project_ctx(prt.cfg)
        r = subprocess.run(
            prt.env.wrap(render(cmd, ctx, "items.command")), cwd=prt.cfg.workdir,
            env=self._env(prt.cfg, prt.cfg.env, ctx, None, cpus=1), stdin=subprocess.DEVNULL,
            capture_output=True, text=True, timeout=self.sc.check_timeout,
        )
        if r.returncode != 0:
            raise ConfigError(f"items.command exited {r.returncode}: {_first_line(r.stderr)}")
        return r.stdout

    def _load_or_discover(self, t: Task) -> None:
        """Adopt the state on disk (it may have been edited by `fdgx retry/reset`), or create it."""
        st = self.store.load(t.project.name, t.name)
        if st is not None:
            t.state = st
            return
        t.state = TaskState(id=t.id)
        t.checked = False
        if t.tmpl.done_if:
            t.checked = True
            vars_, env, _ = t.project.effective(t.tmpl, 0)
            ctx = self._task_ctx(t, vars_, "check", [])
            try:
                if self._check(t.project, render(t.tmpl.done_if, ctx, f"{t.id}.done_if"), ctx, env):
                    t.state.phase = "done"
                    t.state.event("done_if satisfied at discovery")
            except ConfigError as e:
                self._fail(t, "setup", f"config: {e}")
                return
        self._save(t)

    # ================================================================== GPU state

    def _maybe_probe(self, now: float) -> None:
        if self.sc.cpu_only or now - self._last_probe < self.sc.gpu_poll_seconds:
            return
        self._last_probe = now
        self.snapshot = snap = self.probe.probe()
        if snap.error != self._last_probe_error:
            if snap.error:
                self.event(f"warning: GPU probe failed ({snap.error}); not starting new GPU jobs")
            elif self._last_probe_error:
                self.event("GPU probe recovered")
            self._last_probe_error = snap.error

    def _allowed_gpus(self, snap: Snapshot) -> list[int]:
        return [g for g in (self.sc.gpus if self.sc.gpus is not None else sorted(snap.gpus)) if g in snap.gpus]

    def usable_gb(self, now: float) -> dict[int, float] | None:
        """Free VRAM we may still hand out per GPU, or None when GPU state is unknown.

        free (nvidia-smi) - safety margin - VRAM our running jobs declared but may not have allocated yet.
        Per job, what it really uses is known when its processes show up in nvidia-smi, or when it writes
        $FDGX_VRAM_FILE. Otherwise (PID namespaces) its whole declaration is reserved during
        `launch_warmup`, and afterwards whatever part of it the GPU's used memory can't already contain,
        so the reservation lasts the whole run without counting memory twice.
        """
        snap = self.snapshot
        if snap is None or snap.error:
            return None
        runs = [a for a in self.active.values() if a.uses_gpu]
        members = session_members() if runs and snap.apps else {}
        extra: dict[int, float] = defaultdict(float)
        known: dict[int, float] = defaultdict(float)  # our usage we know about, per GPU
        unknown: dict[int, float] = defaultdict(float)  # declarations of post-warmup jobs we can't see
        busy: set[int] = set()
        for a in runs:
            busy.update(a.gpus)
            pids = members.get(a.job.pid, {a.job.pid})
            observed: dict[int, float] = defaultdict(float)
            seen = False
            for gi, pid, used in snap.apps:
                if pid in pids:
                    observed[gi] += used
                    seen = True
            reported = None if seen else self._reported_vram(a)
            for g in a.gpus:
                if seen:
                    extra[g] += max(0.0, a.vram_gb - observed[g])
                    known[g] += observed[g]
                elif reported is not None:
                    extra[g] += max(0.0, a.vram_gb - reported)
                    known[g] += reported
                elif now - a.job.started < self.sc.launch_warmup:
                    extra[g] += a.vram_gb
                else:
                    unknown[g] += a.vram_gb
        usable = {}
        for g in self._allowed_gpus(snap):
            gpu = snap.gpus[g]
            if (self.sc.max_gpu_util is not None and g not in busy and gpu.util is not None
                    and gpu.util > self.sc.max_gpu_util):
                continue  # someone else is hammering it
            room_for_unknown = max(0.0, gpu.used_gb - known[g])  # unknown jobs use at most this much
            extra[g] += max(0.0, unknown[g] - room_for_unknown)
            usable[g] = gpu.free_gb - self.sc.vram_safety_margin_gb - extra[g]
        return usable

    @staticmethod
    def _reported_vram(a: Active) -> float | None:
        """GB per GPU the job wrote to $FDGX_VRAM_FILE (bytes), if any."""
        if a.vram_file is None:
            return None
        try:
            return float(a.vram_file.read_text().split()[0]) / 1024 ** 3
        except (OSError, ValueError, IndexError):
            return None

    def _gpu_slots(self, gpus: list[int]) -> dict[int, int]:
        slots = {}
        for g in gpus:
            cap = self.sc.jobs_cap(g)
            if cap is not None:
                slots[g] = cap - sum(1 for a in self.active.values() if a.uses_gpu and g in a.gpus)
        return slots

    # ================================================================== launching

    def _ready(self, t: Task, phase: str, now: float) -> bool:
        prt = self.projects.get(t.project.name)
        return (
            t.live and not t.parked and t.state.phase == phase and t.id not in self.active
            and t.state.next_at <= now
            and prt is not None and prt.status == "ready" and not prt.removed
            and (phase != "setup" or t.cond[0] == deps.OK)  # dependencies gate the first phase only
        )

    def _cpu_slots(self) -> int:
        return self.sc.max_cpu_phases - sum(1 for a in self.active.values() if a.counts_as_cpu_phase)

    def _launch_runs(self, now: float) -> None:
        if self.stop_requested:
            return
        cands = []
        for t in sorted((t for t in self.tasks.values() if self._ready(t, "run", now)), key=lambda t: t.order):
            _, _, res = t.project.effective(t.tmpl, t.state.oom_level)
            if self.sc.cpu_only or res.max_gpus == 0:
                self._start_phase(t, "run")  # no GPU: only the CPU budget applies
            else:
                cands.append((t, res))
        if not cands:
            return
        running = sum(1 for a in self.active.values() if a.uses_gpu)
        if self.sc.max_running is not None and running >= self.sc.max_running:
            return
        usable = self.usable_gb(now)
        if usable is None:
            return
        snap = self.snapshot
        allowed = self._allowed_gpus(snap)
        all_visible = len(allowed) == len(self.sc.gpus) if self.sc.gpus is not None else bool(allowed)
        reqs, by_key = [], {}
        for t, res in cands:
            capable = [g for g in allowed if snap.gpus[g].total_gb - self.sc.vram_safety_margin_gb >= res.vram_gb]
            if res.max_gpus > 0 and all_visible and len(capable) < max(res.min_gpus, 1):
                self._fail(t, "run", f"can never fit: needs {res.min_gpus}x{res.vram_gb:g} GB but only "
                                     f"{len(capable)} usable GPU(s) are that large")
                continue
            since = self.waiting_since.setdefault(t.id, now)
            starving = self.sc.starvation is not None and now - since > self.sc.starvation
            reqs.append(Request(t.id, res.vram_gb, res.min_gpus, res.max_gpus, starving))
            by_key[t.id] = (t, res)
        for key, gpus in allocate(reqs, usable, self._gpu_slots(allowed)):
            if self.sc.max_running is not None and running >= self.sc.max_running:
                break
            t, res = by_key[key]
            if self._start_phase(t, "run", gpus, res.vram_gb):
                running += 1

    def _launch_cpu(self, now: float) -> None:
        if self.stop_requested:
            return
        slots = self._cpu_slots()
        if slots <= 0:
            return
        for t in sorted((t for t in self.tasks.values() if self._ready(t, "epilogue", now)), key=lambda t: t.order):
            if slots <= 0:
                return
            if self._start_phase(t, "epilogue"):
                slots -= 1
        # Setups run ahead of the GPU queue, but only up to `max_prepared` tasks waiting for a GPU.
        prepared = sum(1 for t in self.tasks.values()
                       if t.live and t.state.phase == "run" and t.id not in self.active)
        prepared += sum(1 for a in self.active.values() if a.task and a.phase == "setup")
        for t in sorted((t for t in self.tasks.values() if self._ready(t, "setup", now)), key=lambda t: t.order):
            if slots <= 0 or prepared >= self.sc.max_prepared:
                return
            if self._start_phase(t, "setup"):
                slots -= 1
                prepared += 1
            elif t.state.phase == "run":
                prepared += 1

    def _start_phase(self, t: Task, phase: str, gpus: list[int] | None = None, vram_gb: float = 0.0) -> bool:
        """Launch `phase` of `t`. Returns True if a process was started (False: skipped, done or failed)."""
        fresh = self.store.load(t.project.name, t.name)  # honor `fdgx retry/reset` done in the meantime
        if fresh is None:
            self._load_or_discover(t)
            return False
        t.state = fresh
        if fresh.phase != phase or fresh.next_at > time.time():
            return False
        st = t.state
        gpus = gpus or []
        vars_, env, res = t.project.effective(t.tmpl, st.oom_level)
        ph = t.tmpl.phase(phase)
        cpus = ph.cpus if ph.cpus is not None else res.cpus
        if not self._fits(cpus, res.ram_gb):
            return False
        ctx = self._task_ctx(t, vars_, phase, gpus)
        try:
            if not t.checked:
                t.checked = True
                if t.tmpl.done_if and self._check(t.project, render(t.tmpl.done_if, ctx, f"{t.id}.done_if"), ctx, env):
                    st.phase = "done"
                    st.event("done_if satisfied")
                    self._save(t)
                    self.event(f"= {t.id}: already done (done_if)")
                    return False
            if ph.run is None or (ph.skip_if and self._check(
                    t.project, render(ph.skip_if, ctx, f"{t.id}.{phase}.skip_if"), ctx, env)):
                self._advance(t, phase, skipped=ph.run is not None)
                return False
            cmd = render(ph.run, ctx, f"{t.id}.{phase}")
            clean = [render(c, ctx, f"{t.id}.{phase}.clean") for c in ph.clean]
            fdgx_env = {
                "FDGX_TASK": t.id, "FDGX_PHASE": phase, "FDGX_ATTEMPT": str(st.attempts + 1),
                "FDGX_OOM_LEVEL": str(st.oom_level), "FDGX_GPUS": ctx["gpus"], "FDGX_NUM_GPUS": str(len(gpus)),
            }
            log_path = self._log_path(t.project, t.name, phase)
            fdgx_env["FDGX_LOG_FILE"] = str(log_path)
            vram_file = None
            if phase == "run" and gpus:
                vram_file = self.sc.state_dir / "vram" / t.project.name / f"{safe_name(t.name)}.txt"
                fdgx_env["FDGX_VRAM_FILE"] = str(vram_file)
            child_env = self._env(t.project, env, ctx, gpus, fdgx_env, cpus=cpus)
        except ConfigError as e:
            self._fail(t, phase, f"config: {e}")
            return False
        if clean:
            try:
                deleted = clean_paths(t.project.workdir, clean)
            except ValueError as e:
                self._fail(t, phase, f"config: {e}")
                return False
            except OSError as e:
                self._on_failure(t, phase, JobResult(-1, f"clean failed: {e}", False, 0.0, ""), None, time.time())
                return False
            if deleted:
                st.event(f"cleaned {', '.join(deleted)}")
                self.event(f"  {t.id}: cleaned {', '.join(deleted)} before {phase}")
        if vram_file is not None:
            vram_file.parent.mkdir(parents=True, exist_ok=True)
            vram_file.unlink(missing_ok=True)
        argv = self._env_for(t).wrap(cmd)
        job = Job(
            argv, cwd=t.project.workdir, env=child_env, log_path=log_path,
            header=(f"### fdgx: {t.id} {phase} attempt {st.attempts + 1} oom_level {st.oom_level} "
                    f"gpus [{ctx['gpus']}] at {datetime.now():%Y-%m-%d %H:%M:%S}\n### {cmd}\n"),
            timeout=ph.timeout or t.tmpl.timeout,
            stall_timeout=ph.stall_timeout or t.tmpl.stall_timeout,
            oom_re=self.oom_re, kill_grace=self.sc.kill_grace,
        )
        try:
            job.start()
        except OSError as e:
            self._on_failure(t, phase, JobResult(-1, f"failed to start: {e}", False, 0.0, ""), None, time.time())
            return False
        st.running = True
        st.pid = job.pid
        st.pid_start = proc_start_time(job.pid)
        st.gpus = gpus
        st.host = self.hostname
        st.event(f"start {phase} attempt {st.attempts + 1} oom_level {st.oom_level} gpus {gpus}")
        self._save(t)
        self.active[t.id] = Active(t.id, t.id, phase, job, gpus, vram_gb, task=t, cpus=cpus,
                                   ram_gb=res.ram_gb, vram_file=vram_file)
        self.waiting_since.pop(t.id, None)
        extra = f" gpus [{ctx['gpus']}] {vram_gb:g} GB each" if phase == "run" and gpus else ""
        extra += f" cpus {cpus:g}" if cpus != 1 else ""
        notes = ([f"attempt {st.attempts + 1}"] if st.attempts else []) + (
            [f"oom level {st.oom_level}"] if st.oom_level and phase == "run" else [])
        self.event(f"> {t.id}: {phase}{extra}" + (f" ({', '.join(notes)})" if notes else ""))
        return True

    def _env_for(self, t: Task) -> EnvManager:
        return self.projects[t.project.name].env

    # ================================================================== dependencies

    def _update_conditions(self) -> None:
        live = [t for t in self.tasks.values() if t.live]
        needs = {t.id: t.needs for t in live if t.needs}
        conds: dict[str, tuple[str, str]] = {}
        if needs:
            phases = {t.id: t.state.phase for t in live}
            status = {n: self._project_status(p) for n, p in self.projects.items()}
            conds = deps.evaluate(needs, phases, status)
        for t in live:
            c = conds.get(t.id, (deps.OK, ""))
            if c[0] != t.cond[0] and t.state.phase == "setup" and t.id not in self.active:
                if c[0] == deps.BLOCKED:
                    self.event(f"! {t.id}: blocked: {c[1]}")
                elif t.cond[0] == deps.BLOCKED:
                    self.event(f"  {t.id}: unblocked")
            t.cond = c

    # ================================================================== project epilogue

    def _epilogue_fingerprint(self, prt: ProjectRT) -> str | None:
        """Fingerprint of the finished task set if the project epilogue is due now, else None."""
        p = prt.cfg
        if prt.removed or prt.status != "ready" or p.epilogue.run is None:
            return None
        if f"{p.name}::epilogue" in self.active:
            return None
        # Disabled tasks are not part of the project right now; held ones are (they block the epilogue).
        tasks = [t for t in self.tasks.values() if t.live and not t.disabled and t.project.name == p.name]
        if not tasks or any(t.id in self.active for t in tasks):
            return None
        when = p.epilogue_when or self.sc.project_epilogue_when
        for t in tasks:
            ph = t.state.phase
            if ph == "done":
                continue
            if when == "all_finished" and (ph == "failed" or (ph == "setup" and t.cond[0] == deps.BLOCKED)):
                continue
            return None
        fp = hashlib.sha1("\n".join(sorted(f"{t.id}:{t.state.phase}" for t in tasks)).encode()).hexdigest()
        return None if fp == prt.epilogue.get("fingerprint") else fp

    def _launch_project_epilogues(self, now: float) -> None:
        if self.stop_requested:
            return
        for prt in self.projects.values():
            if self._cpu_slots() <= 0:
                return
            cpus = self._project_phase_cpus(prt.cfg.epilogue)
            if now < prt.epilogue_next_at or not self._fits(cpus, 0):
                continue
            fp = self._epilogue_fingerprint(prt)
            if fp is None:
                continue
            p = prt.cfg
            ctx = self._project_ctx(p)
            try:
                if p.epilogue.skip_if and self._check(
                        p, render(p.epilogue.skip_if, ctx, f"{p.name}.epilogue.skip_if"), ctx, p.env):
                    self._record_epilogue(prt, fp, "skipped (skip_if)")
                    continue
                argv = prt.env.wrap(render(p.epilogue.run, ctx, f"{p.name}.epilogue.run"))
                log_path = self._log_path(p, "_project", "epilogue")
                env = self._env(p, p.env, ctx, None, {"FDGX_PROJECT": p.name, "FDGX_PHASE": "project-epilogue",
                                                      "FDGX_LOG_FILE": str(log_path)}, cpus=cpus)
                job = Job(
                    argv, cwd=p.workdir, env=env, log_path=log_path,
                    header=f"### fdgx: {p.name} project epilogue: {argv}\n",
                    timeout=p.epilogue.timeout, stall_timeout=p.epilogue.stall_timeout,
                    kill_grace=self.sc.kill_grace,
                )
                job.start()
            except (ConfigError, OSError) as e:
                self.event(f"! {p.name}: project epilogue FAILED: {e}")
                self._record_epilogue(prt, fp, f"failed: {e}")
                continue
            key = f"{p.name}::epilogue"
            self.active[key] = Active(key, f"{p.name} [project]", "epilogue", job, project=prt, cpus=cpus)
            prt.epilogue_running_fp = fp
            self.event(f"> {p.name}: project epilogue")

    def _project_epilogue_done(self, prt: ProjectRT, a: Active, res: JobResult, now: float) -> None:
        p = prt.cfg
        fp = prt.epilogue_running_fp
        if res.ok:
            self.event(f"  {p.name}: project epilogue ok ({fmt_dur(res.duration)})")
            self._record_epilogue(prt, fp, "ok")
            return
        prt.epilogue_attempts += 1
        what = f"{res.reason} rc={res.returncode}"
        if prt.epilogue_attempts >= p.retry.max_attempts:
            self.event(f"! {p.name}: project epilogue FAILED ({what}); see {a.job.log_path}")
            self._record_epilogue(prt, fp, f"failed: {what}")  # re-runs only when the task set changes
        else:
            prt.epilogue_next_at = now + p.retry.delay(prt.epilogue_attempts)
            self.event(
                f"x {p.name}: project epilogue {what} (attempt {prt.epilogue_attempts}/{p.retry.max_attempts}), "
                f"retrying in {fmt_dur(prt.epilogue_next_at - now)}; see {a.job.log_path}"
            )

    def _record_epilogue(self, prt: ProjectRT, fp: str | None, result: str) -> None:
        prt.epilogue = {"fingerprint": fp, "result": result, "time": round(time.time(), 1)}
        prt.epilogue_attempts = 0
        prt.epilogue_next_at = 0.0
        prt.epilogue_running_fp = None
        try:
            atomic_write(self._epilogue_state_path(prt.cfg.name), json.dumps(prt.epilogue, indent=1))
        except OSError as e:
            self._warn_once(f"cannot save project epilogue state: {e}")

    # ================================================================== machine GPU process log

    def _gpulog_needle(self) -> str | None:
        cfg = self.sc.gpu_process_log
        return (cfg.mine if cfg and cfg.mine else None) or gpulog.default_mine()

    def _check_gpulog(self, now: float) -> None:
        """Diagnostics only: warn about GPU memory under your name that FDGX doesn't account for."""
        if any(a.uses_gpu for a in self.active.values()):
            self._last_gpu_job = now
        cfg = self.sc.gpu_process_log
        if cfg is None or now - self._gpulog_at < GPULOG_EVERY:
            return
        self._gpulog_at = now
        self.gpulog = log = gpulog.read(cfg, now)
        if log.error:
            self._warn_once(f"gpu_process_log not usable: {log.error}")
            return
        # Our own jobs may linger in a log written every so often: only "idle" once that has passed.
        recently = now - self._last_gpu_job < log.age + 2 * GPULOG_EVERY
        for key, msg in gpulog.ghost_warnings(log, self._gpulog_needle(), 1 if recently else 0):
            if key not in self._gpulog_warned:
                self._gpulog_warned.add(key)
                self.event(f"warning: {msg}")

    def _gpulog_hint(self) -> str:
        cfg = self.sc.gpu_process_log
        if cfg is None:
            return ""
        log = gpulog.read(cfg)
        mine = [] if log.error else log.mine(self._gpulog_needle())
        if not mine:
            return ""
        return "; the GPU process log shows " + ", ".join(f"{c}: {gb:.0f} GB" for c, gb, _ in mine)

    # ================================================================== live control

    def _process_control(self) -> None:
        for path, req in control.pending(self.sc.state_dir):
            try:
                path.unlink()
            except OSError:
                continue  # someone else took it
            try:
                ok, lines = self._handle_control(req)
            except Exception as e:  # a bad request must not take the scheduler down
                ok, lines = False, [f"error: {type(e).__name__}: {e}"]
            for line in lines:
                self.event(f"control {req.get('op')}: {line}")
            self._last_status_write = 0.0  # refresh status.json at the end of this tick
            try:
                control.reply(self.sc.state_dir, str(req.get("id")), ok, lines)
            except OSError as e:
                self._warn_once(f"cannot write control reply: {e}")

    def _handle_control(self, req: dict) -> tuple[bool, list[str]]:
        op = req.get("op")
        patterns = [str(p) for p in req.get("patterns") or []]
        if op == "stop":
            self.request_stop()
            return True, [f"stopping: killing {sum(1 for a in self.active.values())} running job(s)"]
        if op == "reload":
            self._last_reload = time.time()
            self._reload()
            for prt in self.projects.values():
                prt.last_scan = None  # rescanned later in this tick
            errors = len(self.config_errors)
            return True, ["config reloaded" + (f" ({errors} source(s) with errors, see the log)" if errors else "")]
        if op not in ("hold", "release", "retry", "reset"):
            return False, [f"unknown operation {op!r}"]
        if op != "retry" and not patterns:
            return False, [f"{op} needs at least one task pattern, e.g. 'train-*' or 'proj/*'"]
        targets = sorted((t for t in self.tasks.values() if t.live and control.match(t.id, patterns)),
                         key=lambda t: t.order)
        lines = []
        for t in targets:
            st = t.state
            active = self.active.get(t.id)
            if op == "hold":
                if st.phase == "done" or st.held:
                    continue
                st.held = True
                st.event("held")
                self._save(t)
                if active and req.get("kill"):
                    active.job.kill("held")
                    lines.append(f"held {t.id}; killing its running {active.phase} phase")
                else:
                    lines.append(f"held {t.id}" + (f" (its running {active.phase} phase will finish)" if active else ""))
            elif op == "release":
                if not st.held:
                    continue
                st.held = False
                st.event("released")
                self._save(t)
                lines.append(f"released {t.id}")
            elif op == "retry":
                if st.phase != "failed":
                    continue
                control.retry_state(st, bool(req.get("reset_oom")))
                self._save(t)
                lines.append(f"retry {t.id} from {st.phase}")
            elif op == "reset":
                if active:
                    lines.append(f"skipped {t.id}: running right now (use `fdgx hold --kill` first)")
                    continue
                self.store.delete(t.project.name, t.name)
                self._load_or_discover(t)
                lines.append(f"reset {t.id}: now {t.state.phase}")
        if op == "retry":
            for name, prt in self.projects.items():
                if (str(prt.epilogue.get("result", "")).startswith("failed")
                        and control.matches_project(name, patterns)):
                    self._record_epilogue(prt, None, "retry requested")  # forget the fingerprint: due again
                    lines.append(f"retry project {name} epilogue")
        if not lines:
            lines.append(f"nothing to {op}" + (f" matching {' '.join(patterns)}" if patterns else ""))
        return True, lines

    # ================================================================== hot reload

    def _reload(self) -> None:
        self.activity = "reloading config"
        res = load(self.cfg.path, previous=self.cfg)
        errors = set(res.errors)
        for e in sorted(errors - self.config_errors):
            self.event(f"config error (keeping the previous version of that source): {e}")
        if self.config_errors and not errors:
            self.event("config errors resolved")
        self.config_errors = errors
        warnings = set(res.warnings)
        for w in sorted(warnings - self.config_warnings):
            self.event(f"config warning: {w}")
        self.config_warnings = warnings
        if res.config is not None and res.config is not self.cfg:
            self._apply_config(res.config)

    def _apply_config(self, new: Config) -> None:
        old_sc = self.sc
        changed = [f for f in RESTART_ONLY_SCHEDULER if getattr(new.scheduler, f) != getattr(old_sc, f)]
        if changed:
            self._warn_once(f"scheduler settings {', '.join(changed)} changed: they apply at the next start")
        self.sc = replace(new.scheduler, **{f: getattr(old_sc, f) for f in RESTART_ONLY_SCHEDULER})
        self.oom_re = self._compile_oom(self.sc)
        self.cpu_budget = self.sc.cpus or available_cpus()
        new_names = {p.name for p in new.projects}
        for name, prt in self.projects.items():
            if name not in new_names and not prt.removed:
                prt.removed = True
                for t in self.tasks.values():
                    if t.project.name == name:
                        t.live = False
                self.event(f"project {name} removed from the config: nothing new will start; running phases finish")
        projects = []
        for p in new.projects:
            prt = self.projects.get(p.name)
            if prt is None:
                self.projects[p.name] = self._new_project(p)
                self.event(f"project {p.name} added ({p.source})")
                projects.append(p)
                continue
            old = prt.cfg
            frozen = [f for f in RESTART_ONLY_PROJECT if getattr(p, f) != getattr(old, f)]
            if frozen:
                self._warn_once(f"project {p.name}: changes to {', '.join(frozen)} apply at the next start")
                p = replace(p, **{f: getattr(old, f) for f in RESTART_ONLY_PROJECT})
            if prt.removed:
                prt.removed = False
                self.event(f"project {p.name} is back in the config")
            prt.cfg = p
            prt.env.project = p
            prt.last_scan = None  # rescan now with the new templates
            projects.append(p)
        self.cfg = replace(new, scheduler=self.sc, projects=projects)

    # ================================================================== completion

    def _poll_active(self, now: float) -> None:
        for key, a in list(self.active.items()):
            res = a.job.poll(now)
            if res is None:
                continue
            del self.active[key]
            if a.project is not None:
                self._project_done(a.project, a, res, now)
                continue
            t = a.task
            if res.reason == "held":  # killed by `fdgx hold --kill`
                self._interrupted(t, a.phase, "killed by hold")
                self.event(f"  {t.id}: {a.phase} killed (held); it resumes at this phase when released")
                continue
            t.state.running = False
            t.state.pid = t.state.pid_start = None
            if res.ok:
                t.state.attempts = 0
                self.event(f"  {t.id}: {a.phase} ok ({fmt_dur(res.duration)})")
                self._advance(t, a.phase)
            else:
                self._on_failure(t, a.phase, res, a.job, now)

    def _advance(self, t: Task, phase: str, skipped: bool = False) -> None:
        st = t.state
        st.phase = NEXT_PHASE[phase]
        st.next_at = 0.0
        st.attempts = 0
        st.event(f"{phase} {'skipped (skip_if)' if skipped else 'done'}")
        self._save(t)
        if st.phase == "done":
            self.event(f"* {t.id}: DONE")

    def _fail(self, t: Task, phase: str, error: str) -> None:
        st = t.state
        st.phase, st.failed_phase, st.last_error, st.running = "failed", phase, error, False
        st.event(f"failed in {phase}: {_first_line(error)}")
        self._save(t)
        self.event(f"! {t.id}: FAILED in {phase}: {_first_line(error)}")

    def _on_failure(self, t: Task, phase: str, res: JobResult, job: Job | None, now: float) -> None:
        st = t.state
        log = f"; log: {job.log_path}" if job else ""
        host_oom = res.reason == "failed" and killed_by_sigkill(res.returncode)
        if host_oom:  # nothing we sent: most likely the kernel's OOM killer
            st.event("host_oom_suspected: killed by SIGKILL that fdgx didn't send")
            self.event(f"! {t.id}: killed by SIGKILL that fdgx didn't send: host out-of-memory killer suspected"
                       " (consider resources.ram_gb)")
        if res.oom and not host_oom and phase == "run" and st.oom_level < len(t.tmpl.on_oom):
            step = t.tmpl.on_oom[st.oom_level]
            st.oom_level += 1
            st.next_at = now + self.sc.oom_retry_delay
            st.last_error = f"run: OOM\n{res.tail}"
            st.event(f"OOM -> oom level {st.oom_level}")
            self._save(t)
            change = ", ".join(f"{k}={v}" for k, v in {**step.vars, **step.env, **step.resources}.items())
            self.event(f"~ {t.id}: out of memory -> retrying with on_oom[{st.oom_level - 1}] ({change}){log}")
            return
        reset_after = t.tmpl.retry.reset_after
        if reset_after and st.attempts and res.duration >= reset_after:
            st.event(f"ran {fmt_dur(res.duration)} before failing: retry budget reset")
            st.attempts = 0
        st.attempts += 1
        what = f"{res.reason}{' (OOM)' if res.oom else ''}{' (host OOM?)' if host_oom else ''} rc={res.returncode}"
        error = f"{phase}: {what}\n{res.tail}"
        if st.attempts >= t.tmpl.retry.max_attempts:
            self._fail(t, phase, error)
            if job:
                self.event(f"  log: {job.log_path}")
            return
        st.last_error = error
        st.next_at = now + t.tmpl.retry.delay(st.attempts)
        st.event(f"{phase} {what}; retry {st.attempts}/{t.tmpl.retry.max_attempts}")
        self._save(t)
        self.event(
            f"x {t.id}: {phase} {what} (attempt {st.attempts}/{t.tmpl.retry.max_attempts}), "
            f"retrying in {fmt_dur(st.next_at - now)}{log}"
        )
