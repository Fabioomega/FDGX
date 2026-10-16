import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from fdgx.allocator import Request, allocate
from fdgx.config import load_config
from fdgx.gpu import Gpu, Snapshot
from fdgx.runner import available_cpus, clean_paths
from fdgx.scheduler import Active, Scheduler

CFG = """
scheduler:
  state_dir: state
  vram_safety_margin_gb: 2
  launch_warmup: 300
  cpus: 4
  max_jobs_per_gpu: 2
projects:
  p:
    workdir: .
    env_manager: shell
    tasks:
      - name: t
        resources: {vram_gb: 40, cpus: 2, ram_gb: 8, gpus: {min: 1, max: 2}}
        retry: {reset_after: 30m}
        run: {run: "echo", clean: ["work/*.tmp"]}
"""


@pytest.fixture
def sched(tmp_path):
    (tmp_path / "jobs.yaml").write_text(CFG)
    return Scheduler(load_config(tmp_path / "jobs.yaml"), out=lambda _: None)


def gpu(used: float, total: float = 80.0) -> Snapshot:
    return Snapshot({0: Gpu(0, "u0", total, used, total - used)}, [], time.time())


def job(age: float, declared: float = 40.0, vram_file=None, cpus: float = 1.0) -> Active:
    j = SimpleNamespace(pid=999999, started=time.time() - age, last_line="")
    return Active("k", "k", "run", j, [0], declared, task=SimpleNamespace(), cpus=cpus, vram_file=vram_file)


def test_config_fields(sched):
    t = sched.cfg.projects[0].tasks[0]
    assert (t.resources.cpus, t.resources.ram_gb) == (2, 8)
    assert t.retry.reset_after == 1800 and t.run.clean == ["work/*.tmp"]
    assert sched.cpu_budget == 4 and sched.sc.jobs_cap(0) == 2


def test_reservation_warmup_and_whole_run(sched, tmp_path):
    sched.snapshot = gpu(used=0)
    sched.active = {"a": job(age=10)}  # in warmup: whole declaration reserved
    assert sched.usable_gb(time.time()) == {0: 80 - 2 - 40}
    sched.active = {"a": job(age=1000)}  # past warmup, GPU shows nothing used: still reserved
    assert sched.usable_gb(time.time()) == {0: 80 - 2 - 40}
    sched.snapshot = gpu(used=45)  # our 40 + 5 from others: already in `used`, not counted twice
    assert sched.usable_gb(time.time()) == {0: 35 - 2}
    sched.snapshot = gpu(used=25)  # job not at its peak yet: keep the remaining 15 reserved
    assert sched.usable_gb(time.time()) == {0: 55 - 2 - 15}


def test_reservation_from_vram_file(sched, tmp_path):
    f = tmp_path / "vram.txt"
    f.write_text(str(10 * 1024 ** 3))
    sched.snapshot = gpu(used=10)
    sched.active = {"a": job(age=10, vram_file=f)}
    assert sched.usable_gb(time.time()) == {0: 70 - 2 - 30}


def test_gpu_slots_and_allocator(sched):
    sched.active = {"a": job(age=10)}
    assert sched._gpu_slots([0, 1]) == {0: 1, 1: 2}
    reqs = [Request(k, 1, 1, 1) for k in "abc"]
    assert allocate(reqs, {0: 70}, {0: 2}) == [("a", [0]), ("b", [0])]


def test_cpu_budget(sched):
    assert sched._fits(100, 0)  # nothing running: an oversized task may still run alone
    sched.active = {"a": job(age=1, cpus=3)}
    assert sched._fits(1, 0) and not sched._fits(2, 0)
    assert available_cpus() >= 1


def test_loop_stalled_shows_in_heartbeat(sched):
    assert "LOOP STALLED" not in sched.status_lines()[0]
    sched.last_tick = time.time() - 125
    assert "LOOP STALLED 2m0" in sched.status_lines()[0]


def test_clean_paths(tmp_path):
    (tmp_path / "work" / "a.tmp").mkdir(parents=True)
    (tmp_path / "work" / "a.tmp" / "x").write_text("x")
    (tmp_path / "work" / "keep.txt").write_text("k")
    (tmp_path / "out.partial").write_text("p")
    assert sorted(clean_paths(tmp_path, ["work/*.tmp", "out.partial", "missing"])) == ["out.partial", "work/a.tmp"]
    assert (tmp_path / "work" / "keep.txt").exists()
    for bad in ("../elsewhere", ".", str(Path(tmp_path).parent / "x")):
        with pytest.raises(ValueError):
            clean_paths(tmp_path, [bad])
