"""End-to-end tests: fake GPUs (FDGX_FAKE_GPUS) + the dummy project's job.py run through a real shell."""

import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from fdgx import cli, control
from fdgx.config import load_config
from fdgx.scheduler import Scheduler
from fdgx.state import FileLock, StateStore, TaskState

JOB = Path(__file__).resolve().parents[1] / "examples" / "proj_dummy" / "job.py"
PY = Path(sys.executable).as_posix()

SCHED = """
scheduler:
  state_dir: state
  heartbeat_seconds: 0.5
  poll_seconds: 0.1
  gpu_poll_seconds: 0.1
  launch_warmup: 1h
  kill_grace: 2
  oom_retry_delay: 0
  max_cpu_phases: 4
  max_prepared: 4
  rescan_seconds: 0.5
"""


def project_yaml(task_yaml: str, extra: str = "", workdir: str = "proj") -> str:
    """`projects:` block for project p (env_manager shell, `{py}` = this Python)."""
    extra = "".join(f"    {line}\n" for line in extra.strip().splitlines())
    return (
        f'projects:\n  p:\n    workdir: {workdir}\n    env_manager: shell\n    vars: {{py: "{PY}"}}\n{extra}'
        "    tasks:\n" + "\n".join("      " + line for line in task_yaml.strip().splitlines()) + "\n"
    )


def make_proj(tmp_path: Path, items) -> Path:
    proj = tmp_path / "proj"
    (proj / "configs").mkdir(parents=True)
    shutil.copy(JOB, proj / "job.py")
    for n in items:
        (proj / "configs" / f"{n}.yaml").write_text("")
    return proj


def make(tmp_path: Path, task_yaml: str, items=("a", "b", "c"), extra: str = "") -> Path:
    make_proj(tmp_path, items)
    cfg = tmp_path / "jobs.yaml"
    cfg.write_text(SCHED + project_yaml(task_yaml, extra))
    return cfg


def make_included(tmp_path: Path, task_yaml: str, items=("a",), extra: str = "") -> tuple[Path, Path]:
    """Main config with only scheduler settings + include of proj/fdgx.yaml. Returns (main, project file)."""
    proj = make_proj(tmp_path, items)
    pfile = proj / "fdgx.yaml"
    pfile.write_text(project_yaml(task_yaml, extra, workdir="."))
    cfg = tmp_path / "fdgx.yaml"
    cfg.write_text(SCHED + "include: [proj/fdgx.yaml]\n")
    return cfg, pfile


LIVE: list = []


@pytest.fixture(autouse=True)
def stop_live_schedulers():
    """A failed assertion must not leave a --stay-alive scheduler running (pytest would never exit)."""
    yield
    while LIVE:
        LIVE.pop().stop()


class Live:
    """A scheduler running in a thread with --stay-alive, for tests that change things while it runs."""

    def __init__(self, cfg: Path):
        self.lines: list[str] = []
        self.sched = Scheduler(load_config(cfg), stay_alive=True, out=self.lines.append)
        self.thread = threading.Thread(target=self.sched.run, daemon=True)
        self.thread.start()
        LIVE.append(self)

    def wait(self, cond, timeout: float = 60, what: str = ""):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if cond():
                return
            time.sleep(0.1)
        self.stop()
        pytest.fail(f"timed out waiting for {what}:\n" + "\n".join(self.lines[-40:]))

    def saw(self, text: str) -> int:
        return sum(text in ln for ln in list(self.lines))

    def stop(self):
        self.sched.request_stop()
        self.thread.join(30)


def run_sched(cfg: Path, timeout: float = 90, stop_after: float | None = None):
    lines: list[str] = []
    sched = Scheduler(load_config(cfg), out=lines.append)
    result = {}
    th = threading.Thread(target=lambda: result.setdefault("rc", sched.run()))
    th.start()
    if stop_after is not None:
        time.sleep(stop_after)
        sched.request_stop()
    th.join(timeout)
    if th.is_alive():
        sched.request_stop()
        th.join(30)
        pytest.fail("scheduler did not finish:\n" + "\n".join(lines[-40:]))
    return result["rc"], lines


def state(cfg: Path, name: str):
    return StateStore(cfg.parent / "state").load("p", name)


def results(cfg: Path, stem: str) -> dict:
    return json.loads((cfg.parent / "proj" / "results" / f"{stem}.json").read_text())


TRAIN = """
- name: "t-{item.stem}"
  items: {glob: "configs/*.yaml"}
  done_if: "test -f results/{item.stem}.packed"
  vars: {bs: 32}
  resources: {vram_gb: 40, gpus: {min: 1, max: 1}}
  setup: '"{py}" job.py prepare --item {item} --step-seconds 0.1'
  run: '"{py}" job.py train --item {item} --steps 3 --step-seconds 0.1 --bs {bs} %s'
  epilogue: '"{py}" job.py pack --item {item}'
  retry: {max_attempts: %d, backoff: [0]}
  %s
"""


def train(extra_args="", max_attempts=3, extra_yaml=""):
    return TRAIN % (extra_args, max_attempts, extra_yaml)


@pytest.fixture(autouse=True)
def fake_gpus(monkeypatch):
    monkeypatch.setenv("FDGX_FAKE_GPUS", "0:80,1:80")


def test_happy_path(tmp_path):
    cfg = make(tmp_path, train())
    rc, lines = run_sched(cfg)
    assert rc == 0, "\n".join(lines)
    for s in "abc":
        assert (tmp_path / "proj" / "results" / f"{s}.packed").exists()
        assert (tmp_path / "proj" / "work" / f"{s}.prep").exists()
        assert state(cfg, f"t-{s}").phase == "done"
    # second run: everything is already done, nothing launches
    rc, lines = run_sched(cfg)
    assert rc == 0 and not any("> p/" in ln for ln in lines)


def test_multi_gpu(tmp_path):
    cfg = make(tmp_path, train().replace("gpus: {min: 1, max: 1}", "gpus: {min: 1, max: 2}"), items=("a",))
    rc, lines = run_sched(cfg)
    assert rc == 0, "\n".join(lines)
    assert results(cfg, "a")["gpus"] == "0,1"


def test_retry_then_success(tmp_path):
    cfg = make(tmp_path, train("--fail-first 2", max_attempts=3), items=("a",))
    rc, lines = run_sched(cfg)
    assert rc == 0, "\n".join(lines)
    st = state(cfg, "t-a")
    assert st.phase == "done"
    assert sum("retry" in h["msg"] for h in st.history) == 2


def test_retries_exhausted(tmp_path):
    cfg = make(tmp_path, train("--fail-first 5", max_attempts=2), items=("a",))
    rc, lines = run_sched(cfg)
    assert rc == 2
    st = state(cfg, "t-a")
    assert st.phase == "failed" and st.failed_phase == "run" and "simulated crash" in st.last_error


def test_oom_ladder(tmp_path):
    ladder = "on_oom: [{bs: 16}, {vars: {bs: 8}, resources: {vram_gb: 50}}]"
    cfg = make(tmp_path, train("--oom-above 8", max_attempts=1, extra_yaml=ladder), items=("a",))
    rc, lines = run_sched(cfg)
    assert rc == 0, "\n".join(lines)
    assert results(cfg, "a")["bs"] == 8
    st = state(cfg, "t-a")
    assert st.oom_level == 2 and st.phase == "done"


def test_stall_watchdog(tmp_path):
    task = train("--hang-at 1", max_attempts=1).replace(
        "run: '", "run:\n    stall_timeout: 1\n    run: '"
    )
    cfg = make(tmp_path, task, items=("a",))
    rc, lines = run_sched(cfg)
    assert rc == 2
    st = state(cfg, "t-a")
    assert st.phase == "failed" and st.last_error.startswith("run: stalled")


def test_done_if_skips(tmp_path):
    cfg = make(tmp_path, train(), items=("a", "b"))
    (tmp_path / "proj" / "results").mkdir()
    (tmp_path / "proj" / "results" / "a.packed").write_text("x")
    rc, lines = run_sched(cfg)
    assert rc == 0
    assert not (tmp_path / "proj" / "work" / "a.prep").exists()
    assert (tmp_path / "proj" / "work" / "b.prep").exists()


def test_heartbeat_while_gpus_full(tmp_path, monkeypatch):
    monkeypatch.setenv("FDGX_FAKE_GPUS", "0:80:79,1:80:79")
    cfg = make(tmp_path, train(), items=("a",))
    rc, lines = run_sched(cfg, stop_after=2.5)
    beats = [ln for ln in lines if "fdgx alive" in ln]
    assert len(beats) >= 3, "\n".join(lines)
    assert any("waiting for free GPUs" in ln for ln in beats)
    assert state(cfg, "t-a").phase == "run"  # set up, still waiting for a GPU


def test_resume_after_hard_kill(tmp_path):
    cfg = make(tmp_path, train().replace("--steps 3", "--steps 40"), items=("a",))
    env = dict(os.environ)
    proc = subprocess.Popen(
        [sys.executable, "-m", "fdgx", "run", str(cfg)], cwd=tmp_path, env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        **({"start_new_session": True} if os.name != "nt" else {}),
    )
    ckpt = tmp_path / "proj" / "work" / "a.ckpt"
    deadline = time.time() + 60
    while time.time() < deadline and not (ckpt.exists() and int(ckpt.read_text() or 0) >= 5):
        time.sleep(0.1)
    assert ckpt.exists(), "job never started"
    # simulate the instance dying: no clean shutdown
    if os.name == "nt":
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True)
    else:
        os.kill(proc.pid, signal.SIGKILL)  # the job's process group survives; recovery must kill it
    proc.wait(10)
    st = state(cfg, "t-a")
    assert st.running and st.phase == "run"

    rc, lines = run_sched(cfg)
    assert rc == 0, "\n".join(lines)
    st = state(cfg, "t-a")
    assert st.phase == "done" and st.interruptions == 1 and st.attempts == 0
    assert results(cfg, "a")["resumed_from"] >= 5


# ---------------------------------------------------------------------------- dependencies

def deps_tasks(train_args: str = "") -> str:
    evals = "".join(f"""
- name: "eval{s}-{{item.stem}}"
  items: {{glob: "configs/*.yaml"}}
  needs: ["train-{{item.stem}}"]
  resources: {{vram_gb: 10}}
  run: '"{{py}}" job.py eval --item {{item}} --suite {s} --steps 1 --step-seconds 0.1'
""" for s in ("A", "B"))
    return f"""
- name: "train-{{item.stem}}"
  items: {{glob: "configs/*.yaml"}}
  resources: {{vram_gb: 10}}
  run: '"{{py}}" job.py train --item {{item}} --steps 3 --step-seconds 0.1 {train_args}'
  retry: {{max_attempts: 1}}
""" + evals


def test_evals_wait_for_training(tmp_path):
    cfg = make(tmp_path, deps_tasks(), items=("a",))
    rc, lines = run_sched(cfg)
    assert rc == 0, "\n".join(lines)
    done = next(i for i, ln in enumerate(lines) if "* p/train-a: DONE" in ln)
    for s in "AB":
        start = next(i for i, ln in enumerate(lines) if f"> p/eval{s}-a: run" in ln)
        assert start > done
        assert (tmp_path / "proj" / "results" / f"a.{s}.json").exists()


def test_failed_dependency_blocks_until_retry(tmp_path):
    cfg = make(tmp_path, deps_tasks("--fail-first 1"), items=("a",))
    # without --stay-alive: nothing can progress, so the scheduler exits reporting blocked tasks
    rc, lines = run_sched(cfg)
    assert rc == 2
    assert any("BLOCKED p/evalA-a: dependency p/train-a failed" in ln for ln in lines)
    status = json.loads((tmp_path / "state" / "status.json").read_text())
    assert status["tasks"]["p/evalB-a"]["condition"] == "blocked"

    live = Live(cfg)
    live.wait(lambda: live.saw("blocked: dependency p/train-a failed") >= 2, what="blocked evals")
    assert cli.main(["retry", str(cfg)]) == 0  # picked up at the next rescan
    live.wait(lambda: (tmp_path / "proj" / "results" / "a.B.json").exists(), what="evals after retry")
    live.wait(lambda: (tmp_path / "proj" / "results" / "a.A.json").exists(), what="evals after retry")
    live.stop()
    assert live.saw("unblocked") >= 2


# ---------------------------------------------------------------------------- project epilogue

EPILOGUE = """epilogue: '"{py}" job.py summary'"""


def test_project_epilogue_runs_once(tmp_path):
    cfg = make(tmp_path, train(), items=("a", "b"), extra=EPILOGUE)
    rc, lines = run_sched(cfg)
    assert rc == 0, "\n".join(lines)
    summary = json.loads((tmp_path / "proj" / "results" / "summary.json").read_text())
    assert summary["results"] == ["a.json", "b.json"]
    assert sum("project epilogue ok" in ln for ln in lines) == 1
    rc, lines = run_sched(cfg)  # nothing changed: not again
    assert rc == 0 and not any("project epilogue" in ln for ln in lines)


def test_project_epilogue_when(tmp_path):
    cfg = make(tmp_path, train("--fail-first 5", max_attempts=1), items=("a",), extra=EPILOGUE)
    rc, lines = run_sched(cfg)  # default all_done: a failed task means no epilogue
    assert rc == 2 and not any("project epilogue" in ln for ln in lines)
    cfg.write_text(cfg.read_text().replace(EPILOGUE, "epilogue: {run: '\"{py}\" job.py summary', when: all_finished}"))
    rc, lines = run_sched(cfg)
    assert rc == 2 and any("project epilogue ok" in ln for ln in lines)


# ---------------------------------------------------------------------------- hot reload

QUICK = """
- name: "train-{item.stem}"
  items: {glob: "configs/*.yaml"}
  resources: {vram_gb: 10}
  run: '"{py}" job.py train --item {item} --steps %d --step-seconds 0.1'
  epilogue: '"{py}" job.py pack --item {item}'
"""


def test_hot_reload_adds_tasks_and_survives_bad_edits(tmp_path):
    cfg, pfile = make_included(tmp_path, QUICK % 3, extra=EPILOGUE)
    live = Live(cfg)
    live.wait(lambda: live.saw("project epilogue ok") == 1, what="first epilogue")

    extra_task = '\n- name: extra\n  resources: {vram_gb: 10}\n  run: \'"{py}" job.py train --item configs/x.yaml --steps 2 --step-seconds 0.1\'\n'
    pfile.write_text(project_yaml(QUICK % 3 + extra_task, EPILOGUE, workdir="."))
    live.wait(lambda: (tmp_path / "proj" / "results" / "x.json").exists(), what="added task")
    live.wait(lambda: live.saw("project epilogue ok") == 2, what="epilogue after new task")

    pfile.write_text("projects: [broken")
    live.wait(lambda: live.saw("config error") >= 1, what="config error logged")
    assert live.sched.tasks["p/extra"].live  # previous version kept
    live.stop()
    assert sum("config error" in ln for ln in live.lines) == 1  # logged once, not every rescan


def test_removing_include_lets_running_phase_finish(tmp_path):
    cfg, pfile = make_included(tmp_path, QUICK % 30)
    live = Live(cfg)
    ckpt = tmp_path / "proj" / "work" / "a.ckpt"
    live.wait(lambda: ckpt.exists(), what="training started")
    cfg.write_text(SCHED + "include: []\n")
    live.wait(lambda: live.saw("removed from the config") == 1, what="project removed")
    live.wait(lambda: live.saw("p/train-a: run ok") == 1, what="running phase to finish")
    time.sleep(1.0)
    live.stop()
    st = state(cfg, "train-a")
    assert st.phase == "epilogue"  # the next phase was not started
    assert not (tmp_path / "proj" / "results" / "a.packed").exists()


# ---------------------------------------------------------------------------- live control

def control_cmd(*argv) -> int:
    return cli.main([argv[0], *map(str, argv[1:])])


def test_hold_queued_task_then_release(tmp_path, monkeypatch):
    monkeypatch.setenv("FDGX_FAKE_GPUS", "0:80")  # one task at a time (40 GB each + warmup reservation)
    cfg = make(tmp_path, train().replace("vram_gb: 40", "vram_gb: 50"), items=("a", "b"))
    live = Live(cfg)
    live.wait(lambda: live.saw("> p/t-a: run") == 1, what="first task running")
    assert control_cmd("hold", cfg, "t-b") == 0
    live.wait(lambda: live.saw("* p/t-a: DONE") == 1, what="first task done")
    time.sleep(1.0)
    assert live.saw("> p/t-b: run") == 0  # held: its run phase never started
    status = tmp_path / "state" / "status.json"
    live.wait(lambda: json.loads(status.read_text())["tasks"]["p/t-b"]["condition"] == "held", what="held status")
    assert control_cmd("release", cfg, "t-b") == 0
    live.wait(lambda: live.saw("* p/t-b: DONE") == 1, what="released task done")
    live.stop()


def test_hold_kill_then_resume(tmp_path):
    cfg = make(tmp_path, train().replace("--steps 3", "--steps 40"), items=("a",))
    ckpt = tmp_path / "proj" / "work" / "a.ckpt"
    live = Live(cfg)
    live.wait(lambda: ckpt.exists() and int(ckpt.read_text() or 0) >= 5, what="training progress")
    assert control_cmd("hold", cfg, "t-a", "--kill") == 0
    live.wait(lambda: live.saw("killed (held)") == 1, what="kill")
    st = state(cfg, "t-a")
    assert (st.phase, st.held, st.attempts, st.interruptions, st.running) == ("run", True, 0, 1, False)
    assert control_cmd("release", cfg, "t-a") == 0
    live.wait(lambda: live.saw("* p/t-a: DONE") == 1, what="resumed task done")
    live.stop()
    assert results(cfg, "a")["resumed_from"] >= 5


def test_retry_and_reload_apply_immediately(tmp_path):
    cfg = make(tmp_path, train("--fail-first 1", max_attempts=1), items=("a",))
    cfg.write_text(cfg.read_text().replace("rescan_seconds: 0.5", "rescan_seconds: 1h"))
    live = Live(cfg)
    live.wait(lambda: (state(cfg, "t-a") or TaskState(id="")).phase == "failed", what="failure")
    assert control_cmd("retry", cfg) == 0
    live.wait(lambda: live.saw("* p/t-a: DONE") == 1, timeout=20, what="retry without waiting for a rescan")

    extra = '\n- name: extra\n  resources: {vram_gb: 10}\n  run: \'"{py}" job.py train --item configs/x.yaml --steps 1 --step-seconds 0.1\'\n'
    cfg.write_text(SCHED.replace("rescan_seconds: 0.5", "rescan_seconds: 1h")
                   + project_yaml(train("--fail-first 1", max_attempts=1) + extra))
    assert control_cmd("reload", cfg) == 0
    live.wait(lambda: live.saw("* p/extra: DONE") == 1, timeout=20, what="task added by reload")
    live.stop()


def test_stop_from_cli_kills_jobs(tmp_path, capsys):
    cfg = make(tmp_path, train().replace("--steps 3", "--steps 100"), items=("a",))
    ckpt = tmp_path / "proj" / "work" / "a.ckpt"
    live = Live(cfg)
    live.wait(lambda: ckpt.exists(), what="training started")
    assert control_cmd("stop", cfg) == 0
    assert "scheduler stopped" in capsys.readouterr().out
    live.thread.join(10)
    assert not live.thread.is_alive()
    st = state(cfg, "t-a")
    assert st.phase == "run" and not st.running and st.interruptions == 1


def test_enabled_false(tmp_path):
    disabled = train().replace('name: "t-{item.stem}"', 'name: "off-{item.stem}"') + "  enabled: false\n"
    cfg = make(tmp_path, train() + disabled, items=("a",))
    live = Live(cfg)
    live.wait(lambda: live.saw("* p/t-a: DONE") == 1, what="enabled task")
    time.sleep(0.5)
    assert live.saw("> p/off-a") == 0
    assert json.loads((tmp_path / "state" / "status.json").read_text())["tasks"]["p/off-a"]["condition"] == "disabled"
    cfg.write_text(cfg.read_text().replace("  enabled: false\n", ""))
    assert control_cmd("reload", cfg) == 0
    live.wait(lambda: live.saw("* p/off-a: DONE") == 1, what="re-enabled task")
    live.stop()


def test_disabled_project_does_nothing(tmp_path):
    cfg = make(tmp_path, train(), items=("a",), extra="enabled: false")
    rc, lines = run_sched(cfg)
    assert rc == 0, "\n".join(lines)
    assert not any("project p ready" in ln or "> p" in ln for ln in lines)


def test_offline_hold_and_stale_requests(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("FDGX_FAKE_GPUS", "0:80:79")  # discover tasks, run setups, but no GPU
    cfg = make(tmp_path, train(), items=("a", "b"))
    run_sched(cfg, stop_after=2.0)
    assert control_cmd("hold", cfg, "t-a") == 0  # no scheduler: edits the state file
    assert "held p/t-a" in capsys.readouterr().out
    control.submit(tmp_path / "state", "stop")  # left over: must not stop the next run
    monkeypatch.setenv("FDGX_FAKE_GPUS", "0:80,1:80")
    rc, lines = run_sched(cfg)
    assert rc == 0, "\n".join(lines)
    assert any("discarded control request stop" in ln for ln in lines)
    assert any("HELD p/t-a" in ln for ln in lines)
    assert state(cfg, "t-b").phase == "done" and state(cfg, "t-a").phase == "run"


# ---------------------------------------------------------------------------- resources & robustness

def run_sampling(cfg: Path, sample, cpu_only: bool = False, timeout: float = 60):
    """Run to completion while calling sample(scheduler) every 20 ms. Returns (rc, lines)."""
    lines: list[str] = []
    config = load_config(cfg)
    config.scheduler.cpu_only = cpu_only
    sched = Scheduler(config, out=lines.append)
    result = {}
    th = threading.Thread(target=lambda: result.setdefault("rc", sched.run()), daemon=True)
    th.start()
    deadline = time.time() + timeout
    while th.is_alive() and time.time() < deadline:
        sample(sched)
        time.sleep(0.02)
    if th.is_alive():
        sched.request_stop()
        th.join(30)
        pytest.fail("scheduler did not finish:\n" + "\n".join(lines[-40:]))
    return result["rc"], lines


def builds(name: str, cpus: int, items: str) -> str:
    return f"""
- name: "{name}-{{item}}"
  items: [{items}]
  resources: {{gpus: 0, cpus: {cpus}}}
  run: '"{{py}}" job.py train --item {name}-{{item}}.yaml --steps 4 --step-seconds 0.1'
"""


def test_cpu_budget_limits_concurrency(tmp_path):
    cfg = make(tmp_path, builds("b", 1, "1, 2, 3, 4") + builds("big", 2, "1"), items=())
    cfg.write_text(cfg.read_text().replace("scheduler:\n", "scheduler:\n  cpus: 4\n"))
    seen = []
    rc, lines = run_sampling(cfg, lambda s: seen.append(sum(a.cpus for a in list(s.active.values()))))
    assert rc == 0, "\n".join(lines)
    assert max(seen) <= 4 and max(seen) >= 3
    assert results(cfg, "b-1")["omp"] == "1" and results(cfg, "big-1")["omp"] == "2"
    assert results(cfg, "b-1")["gpus"] == ""  # CPU-only run phase: no GPU


def test_max_jobs_per_gpu(tmp_path, monkeypatch):
    monkeypatch.setenv("FDGX_FAKE_GPUS", "0:80")
    task = train().replace("vram_gb: 40", "vram_gb: 1").replace("--steps 3", "--steps 10")
    cfg = make(tmp_path, task, items=("a", "b", "c", "d"))
    cfg.write_text(cfg.read_text().replace("scheduler:\n", "scheduler:\n  max_jobs_per_gpu: 3\n  cpus: 16\n"))
    seen = []
    rc, lines = run_sampling(cfg, lambda s: seen.append(sum(1 for a in list(s.active.values()) if a.uses_gpu)))
    assert rc == 0, "\n".join(lines)
    assert max(seen) == 3


def test_vram_stays_reserved_after_warmup(tmp_path, monkeypatch):
    monkeypatch.setenv("FDGX_FAKE_GPUS", "0:80")  # fake GPUs never show our usage, like a PID namespace
    task = train().replace("vram_gb: 40", "vram_gb: 50").replace("--steps 3", "--steps 15")
    cfg = make(tmp_path, task, items=("a", "b"))
    cfg.write_text(cfg.read_text().replace("launch_warmup: 1h", "launch_warmup: 0.3"))
    seen = []
    rc, lines = run_sampling(cfg, lambda s: seen.append(sum(1 for a in list(s.active.values()) if a.uses_gpu)))
    assert rc == 0, "\n".join(lines)
    assert max(seen) == 1  # 80 - 2 - 50 < 50: the second job never joins, even after warmup


def test_reset_after_gives_long_runs_a_fresh_budget(tmp_path):
    task = train("--fail-first 3 --fail-delay 1.2", max_attempts=2).replace(
        "backoff: [0]}", "backoff: [0], reset_after: 1}")
    cfg = make(tmp_path, task, items=("a",))
    rc, lines = run_sched(cfg)
    assert rc == 0, "\n".join(lines)
    st = state(cfg, "t-a")
    assert st.phase == "done" and sum("retry budget reset" in h["msg"] for h in st.history) == 2


def test_clean_before_every_attempt(tmp_path):
    run = ("test ! -e work/{item.stem}.partial && mkdir -p work/{item.stem}.tmp && touch work/{item.stem}.partial"
           ' && "{py}" job.py train --item {item} --steps 2 --step-seconds 0.1 --fail-first 1')
    task = f"""
- name: "t-{{item.stem}}"
  items: {{glob: "configs/*.yaml"}}
  resources: {{vram_gb: 10}}
  run:
    run: '{run}'
    clean: ["work/{{item.stem}}.partial", "work/*.tmp"]
  retry: {{max_attempts: 2, backoff: [0]}}
"""
    cfg = make(tmp_path, task, items=("a",))
    rc, lines = run_sched(cfg)
    assert rc == 0, "\n".join(lines)
    assert any("cleaned work/a.partial, work/a.tmp before run" in ln for ln in lines)

    cfg.write_text(cfg.read_text().replace('"work/*.tmp"', '"../outside"'))
    StateStore(tmp_path / "state").delete("p", "t-a")
    (tmp_path / "proj" / "results" / "a.json").unlink()
    rc, lines = run_sched(cfg)
    assert rc == 2 and "not inside the workdir" in state(cfg, "t-a").last_error


@pytest.mark.skipif(os.name == "nt", reason="SIGKILL is POSIX-only")
def test_host_oom_suspected(tmp_path):
    cfg = make(tmp_path, train("--sigkill-first 1"), items=("a",))
    rc, lines = run_sched(cfg)
    assert rc == 0, "\n".join(lines)
    assert any("host out-of-memory killer suspected" in ln for ln in lines)
    st = state(cfg, "t-a")
    assert any("host_oom_suspected" in h["msg"] for h in st.history) and st.oom_level == 0


def test_cpu_dry_run(tmp_path, monkeypatch):
    monkeypatch.setenv("FDGX_FAKE_GPUS", "0:80:80")  # no free GPU memory at all
    cfg = make(tmp_path, train(), items=("a",))
    rc, lines = run_sampling(cfg, lambda s: None, cpu_only=True)
    assert rc == 0, "\n".join(lines)
    assert results(cfg, "a")["gpus"] == "-1"


def test_lock_holder_is_reported(tmp_path):
    cfg = make(tmp_path, train(), items=("a",))
    lock = FileLock(tmp_path / "state" / "fdgx.lock")
    assert lock.acquire()
    try:
        rc, lines = run_sched(cfg)
    finally:
        lock.release()
    assert rc == 3 and any(f"pid {os.getpid()} on " in ln for ln in lines)


# ---------------------------------------------------------------------------- logs next to the project

def test_project_log_dir_and_env(tmp_path):
    cfg = make(tmp_path, train(), items=("a",), extra='log_dir: "hpc/logs/{project}"')
    rc, lines = run_sched(cfg)
    assert rc == 0, "\n".join(lines)
    logs = list((tmp_path / "proj" / "hpc" / "logs" / "p" / "t-a").glob("*-run.log"))
    assert len(logs) == 1
    assert not (tmp_path / "state" / "logs" / "p").exists()  # nothing left in the scheduler's state dir
    assert (tmp_path / "state" / "fdgx.log").exists()  # scheduler-wide files stay there
    res = results(cfg, "a")
    assert Path(res["log_file"]).resolve() == logs[0].resolve()
    assert Path(res["state_dir"]).resolve() == (tmp_path / "state").resolve()


def test_default_log_dir_unchanged(tmp_path):
    cfg = make(tmp_path, train(), items=("a",))
    rc, lines = run_sched(cfg)
    assert rc == 0
    assert list((tmp_path / "state" / "logs" / "p" / "t-a").glob("*-run.log"))


def test_project_logs_go_to_underscore_project(tmp_path):
    cfg = make(tmp_path, train(), items=("a",), extra="log_dir: logs\nsetup: 'mkdir -p results'")
    rc, lines = run_sched(cfg)
    assert rc == 0, "\n".join(lines)
    assert list((tmp_path / "proj" / "logs" / "_project").glob("*-setup.log"))


def test_failed_project_epilogue_exits_2(tmp_path):
    extra = "epilogue: 'exit 3'\nretry: {max_attempts: 2, backoff: [0]}"
    cfg = make(tmp_path, train(), items=("a",), extra=extra)
    rc, lines = run_sched(cfg)
    assert rc == 2, "\n".join(lines)
    assert any("project p epilogue FAILED: failed: failed rc=3" in ln for ln in lines)
    rc, lines = run_sched(cfg)  # same task set: not re-run, but still reported
    assert rc == 2 and not any("> p: project epilogue" in ln for ln in lines)
    cfg.write_text(cfg.read_text().replace("epilogue: 'exit 3'", "epilogue: 'true'"))  # the fix
    assert control_cmd("retry", cfg, "t-a") == 0  # a task pattern doesn't touch the epilogue
    rc, lines = run_sched(cfg)
    assert rc == 2
    assert control_cmd("retry", cfg, "p/*") == 0  # no scheduler running: edits the state directly
    rc, lines = run_sched(cfg)
    assert rc == 0, "\n".join(lines)
    assert any("project epilogue ok" in ln for ln in lines)


# ---------------------------------------------------------------------------- thread caps everywhere

THREADS = 'echo "$OMP_NUM_THREADS $NUMBA_NUM_THREADS" >'


def test_project_phases_get_thread_caps(tmp_path):
    task = f"""
- name: t
  done_if: '{THREADS} check.txt; false'
  resources: {{vram_gb: 10, cpus: 2}}
  setup: {{run: '{THREADS} task_setup.txt', cpus: 3}}
  run: '{THREADS} task_run.txt'
"""
    extra = f"setup: '{THREADS} setup.txt'\nepilogue: '{THREADS} epilogue.txt'"
    cfg = make(tmp_path, task, items=(), extra=extra)
    rc, lines = run_sched(cfg)
    assert rc == 0, "\n".join(lines)
    out = {f: (tmp_path / "proj" / f).read_text().split() for f in
           ("setup.txt", "epilogue.txt", "check.txt", "task_setup.txt", "task_run.txt")}
    assert out == {"setup.txt": ["1", "1"], "epilogue.txt": ["1", "1"], "check.txt": ["1", "1"],
                   "task_setup.txt": ["3", "3"], "task_run.txt": ["2", "2"]}

    cfg.write_text(cfg.read_text().replace(f"setup: '{THREADS} setup.txt'",
                                           f"setup: {{run: '{THREADS} setup.txt', cpus: 4}}"))
    rc, lines = run_sched(cfg)
    assert rc == 0, "\n".join(lines)
    assert (tmp_path / "proj" / "setup.txt").read_text().split() == ["4", "4"]
