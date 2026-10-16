"""Reader for a machine-provided log of GPU processes (container name and memory per process).

The log is written by someone else and may list ghosts, so FDGX only uses it for diagnostics:
nvidia-smi's free memory (which already accounts for everything) stays the source of truth.
Expected lines (anything else is ignored):

    PID=3908736, Memoria=3908736, 41862 MiB, Nome=/jupyter-arthur.leonida.1.icnza6nynxukzj4svoofoh7to
"""

from __future__ import annotations

import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from .config import GpuLogCfg

_LINE = re.compile(r"PID=(\d+)\s*,\s*Memoria=[^,]*,\s*(\d+(?:\.\d+)?)\s*MiB\s*,\s*Nome=/?(\S+)")


@dataclass
class Entry:
    pid: int
    gb: float
    container: str


@dataclass
class GpuLog:
    entries: list[Entry] = field(default_factory=list)
    age: float = 0.0
    error: str | None = None

    def by_container(self) -> list[tuple[str, float, int]]:
        """(container, total GB, processes), largest first."""
        tot: dict[str, list] = defaultdict(lambda: [0.0, 0])
        for e in self.entries:
            tot[e.container][0] += e.gb
            tot[e.container][1] += 1
        return sorted(((c, gb, n) for c, (gb, n) in tot.items()), key=lambda x: -x[1])

    def mine(self, needle: str | None) -> list[tuple[str, float, int]]:
        return [row for row in self.by_container() if needle and needle in row[0]]


def default_mine() -> str | None:
    return os.environ.get("JUPYTERHUB_USER") or None


def parse(text: str) -> list[Entry]:
    out = []
    for line in text.splitlines():
        m = _LINE.search(line)
        if m:
            out.append(Entry(int(m.group(1)), float(m.group(2)) / 1024.0, m.group(3)))
    return out


def read(cfg: GpuLogCfg, now: float | None = None) -> GpuLog:
    now = now or time.time()
    path = Path(cfg.path)
    try:
        age = now - path.stat().st_mtime
        text = path.read_bytes().decode("utf-8", "replace")
    except OSError as e:
        return GpuLog(error=f"cannot read {path}: {e.strerror or e}")
    if age > cfg.max_age:
        return GpuLog(age=age, error=f"{path} is stale ({age / 60:.0f} min old)")
    return GpuLog(parse(text), age)


def ghost_warnings(log: GpuLog, needle: str | None, our_gpu_jobs: int) -> list[tuple[str, str]]:
    """Hints that something under your name holds GPU memory that FDGX doesn't account for.

    Returns (key, message); the key stays the same while the situation does, so callers warn once.
    """
    if log.error or not needle:
        return []
    mine = log.mine(needle)
    if len(mine) > 1:
        parts = ", ".join(f"{c}: {gb:.0f} GB" for c, gb, _ in mine)
        return [("multi:" + ",".join(sorted(c for c, _, _ in mine)),
                 f"GPU processes under your name appear in {len(mine)} containers ({parts}); "
                 "one may be a ghost from a previous container")]
    if mine and our_gpu_jobs == 0:
        c, gb, n = mine[0]
        return [(f"idle:{c}", f"{gb:.0f} GB in {n} GPU process(es) under your name ({c}) while fdgx runs no "
                              "GPU job: a ghost or another process of yours")]
    return []
