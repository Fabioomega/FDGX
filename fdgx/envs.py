"""How commands of a project are executed: through uv (`uv run`) or a plain shell."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

from .config import Project, SchedulerCfg


def default_shell() -> list[str]:
    if os.name != "nt":
        return ["/bin/bash", "-c"] if Path("/bin/bash").exists() else ["/bin/sh", "-c"]
    # Local development on Windows: prefer Git Bash (never the WSL launcher in System32).
    candidates = [shutil.which("bash"), r"C:\Program Files\Git\bin\bash.exe", r"C:\Program Files\Git\usr\bin\bash.exe"]
    for c in candidates:
        if c and "system32" not in c.lower() and Path(c).exists():
            return [c, "-c"]
    return ["cmd", "/d", "/c"]


THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS")


def thread_env(cpus: float) -> dict[str, str]:
    """Thread-pool sizes for a process allowed `cpus` CPUs (torch, numba, OpenMP, BLAS read these)."""
    return {k: str(max(1, round(cpus))) for k in THREAD_VARS}


def clean_base_env() -> dict[str, str]:
    """os.environ without the scheduler's own virtualenv, so children never pick up FDGX's Python."""
    env = dict(os.environ)
    if sys.prefix != sys.base_prefix:
        venv_bin = os.path.normcase(os.path.join(sys.prefix, "Scripts" if os.name == "nt" else "bin"))
        parts = env.get("PATH", "").split(os.pathsep)
        env["PATH"] = os.pathsep.join(p for p in parts if os.path.normcase(p.rstrip("\\/")) != venv_bin)
        if os.path.normcase(env.get("VIRTUAL_ENV", "")) == os.path.normcase(sys.prefix):
            env.pop("VIRTUAL_ENV")
    env.pop("VIRTUAL_ENV_PROMPT", None)
    return env


class EnvManager:
    def __init__(self, project: Project, shell: list[str]):
        self.project = project
        self.shell = shell

    def sync_argv(self) -> list[str] | None:
        """Command that prepares the environment (None: nothing to do)."""
        return None

    def wrap(self, cmd: str) -> list[str]:
        """argv that runs shell command `cmd` inside the project's environment."""
        raise NotImplementedError


class ShellEnv(EnvManager):
    def wrap(self, cmd: str) -> list[str]:
        prefix = self.project.shell_prefix
        return [*self.shell, f"{prefix} {cmd}" if prefix else cmd]


class UvEnv(EnvManager):
    def __init__(self, project: Project, shell: list[str], uv: str):
        super().__init__(project, shell)
        self.uv = uv

    def sync_argv(self) -> list[str]:
        opts = self.project.uv
        argv = [self.uv, "sync", "--project", str(self.project.workdir)]
        frozen = opts.frozen if opts.frozen is not None else (self.project.workdir / "uv.lock").exists()
        if frozen:
            argv.append("--frozen")
        if opts.python:
            argv += ["--python", opts.python]
        for e in opts.extras:
            argv += ["--extra", e]
        if opts.all_extras:
            argv.append("--all-extras")
        for g in opts.groups:
            argv += ["--group", g]
        return argv

    def wrap(self, cmd: str) -> list[str]:
        # --no-sync: the project was synced once at startup; never touch the venv mid-run.
        return [self.uv, "run", "--no-sync", "--project", str(self.project.workdir), "--", *self.shell, cmd]


def make_env_manager(project: Project, sc: SchedulerCfg) -> EnvManager:
    shell = sc.shell or default_shell()
    if project.env_manager == "uv":
        return UvEnv(project, shell, sc.uv)
    return ShellEnv(project, shell)
