"""GPU discovery through nvidia-smi (or a fake backend for tests: FDGX_FAKE_GPUS)."""

from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

MIB_PER_GB = 1024.0


@dataclass
class Gpu:
    index: int
    uuid: str
    total_gb: float
    used_gb: float
    free_gb: float
    util: float | None = None


@dataclass
class Snapshot:
    gpus: dict[int, Gpu]
    apps: list[tuple[int, int, float]] = field(default_factory=list)  # (gpu index, pid, used GB)
    time: float = 0.0
    error: str | None = None


def _num(s: str) -> float | None:
    try:
        return float(s)
    except ValueError:  # "[N/A]", "[Not Supported]"
        return None


class GpuProbe:
    def __init__(self, nvidia_smi: str = "nvidia-smi", timeout: float = 30.0):
        self.nvidia_smi = nvidia_smi
        self.timeout = timeout

    def probe(self) -> Snapshot:
        now = time.time()
        fake = os.environ.get("FDGX_FAKE_GPUS")
        try:
            if fake:
                return Snapshot(self._fake(fake), [], now)
            gpus = self._query_gpus()
            return Snapshot(gpus, self._query_apps(gpus), now)
        except Exception as e:  # nvidia-smi hangs/crashes on bad days; never take the scheduler down
            return Snapshot({}, [], now, error=f"{type(e).__name__}: {e}")

    def _smi(self, *args: str) -> list[list[str]]:
        out = subprocess.run(
            [self.nvidia_smi, *args, "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=self.timeout, stdin=subprocess.DEVNULL,
        )
        if out.returncode != 0:
            raise RuntimeError((out.stderr or out.stdout).strip() or f"nvidia-smi exited {out.returncode}")
        return [[c.strip() for c in line.split(",")] for line in out.stdout.splitlines() if line.strip()]

    def _query_gpus(self) -> dict[int, Gpu]:
        rows = self._smi("--query-gpu=index,uuid,memory.total,memory.used,memory.free,utilization.gpu")
        gpus = {}
        for idx, uuid, total, used, free, util in rows:
            gpus[int(idx)] = Gpu(
                index=int(idx),
                uuid=uuid,
                total_gb=(_num(total) or 0) / MIB_PER_GB,
                used_gb=(_num(used) or 0) / MIB_PER_GB,
                free_gb=(_num(free) or 0) / MIB_PER_GB,
                util=_num(util),
            )
        return gpus

    def _query_apps(self, gpus: dict[int, Gpu]) -> list[tuple[int, int, float]]:
        by_uuid = {g.uuid: g.index for g in gpus.values()}
        try:
            rows = self._smi("--query-compute-apps=gpu_uuid,pid,used_memory")
        except Exception:
            return []
        apps = []
        for row in rows:
            if len(row) != 3 or row[0] not in by_uuid:
                continue
            pid, used = _num(row[1]), _num(row[2])
            if pid is not None:
                apps.append((by_uuid[row[0]], int(pid), (used or 0) / MIB_PER_GB))
        return apps

    @staticmethod
    def _fake(spec: str) -> dict[int, Gpu]:
        """'0:80,1:80:70' = index:total_gb[:used_gb]. '@path' reads the spec from a file (re-read every probe)."""
        if spec.startswith("@"):
            spec = Path(spec[1:]).read_text().strip()
        gpus = {}
        for part in spec.split(","):
            idx, total, *used = part.strip().split(":")
            u = float(used[0]) if used else 0.0
            gpus[int(idx)] = Gpu(int(idx), f"fake-{idx}", float(total), u, float(total) - u, None)
        return gpus
