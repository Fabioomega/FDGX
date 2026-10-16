"""Load and validate the jobs YAML file, expand items and render command templates."""

from __future__ import annotations

import glob
import json
import re
import shlex
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePosixPath
from typing import Any, Callable


DEFAULT_OOM_PATTERNS = [
    r"CUDA out of memory",
    r"OutOfMemoryError",
    r"CUBLAS_STATUS_ALLOC_FAILED",
    r"CUDNN_STATUS_ALLOC_FAILED",
    r"cudaErrorMemoryAllocation",
    r"out of memory",
    r"RESOURCE_EXHAUSTED",
]

PHASES = ("setup", "run", "epilogue")


class ConfigError(Exception):
    pass


# --------------------------------------------------------------------------- helpers

_UNITS = {"": 1, "s": 1, "m": 60, "h": 3600, "d": 86400}


def parse_duration(value: Any, where: str) -> float | None:
    """Seconds from 90, '30s', '15m', '2h', '1h30m'. None stays None."""
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, str):
        s = value.replace(" ", "").lower()
        if re.fullmatch(r"(\d+(\.\d+)?[smhd]?)+", s):
            return sum(float(n) * _UNITS[u] for n, _, u in re.findall(r"(\d+(\.\d+)?)([smhd]?)", s))
    raise ConfigError(f"{where}: invalid duration {value!r} (use e.g. 90, '30s', '15m', '2h', '1h30m')")


def _check_keys(d: Any, allowed: set[str], where: str) -> dict:
    if d is None:
        return {}
    if not isinstance(d, dict):
        raise ConfigError(f"{where}: expected a mapping, got {type(d).__name__}")
    unknown = set(d) - allowed
    if unknown:
        raise ConfigError(f"{where}: unknown key(s) {sorted(unknown)}; allowed: {sorted(allowed)}")
    return d


def _num(v: Any, where: str, minimum: float | None = 0) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ConfigError(f"{where}: expected a number, got {v!r}")
    if minimum is not None and v < minimum:
        raise ConfigError(f"{where}: must be >= {minimum}, got {v}")
    return float(v)


def _int(v: Any, where: str, minimum: int | None = 0) -> int:
    if isinstance(v, bool) or not isinstance(v, int):
        raise ConfigError(f"{where}: expected an integer, got {v!r}")
    if minimum is not None and v < minimum:
        raise ConfigError(f"{where}: must be >= {minimum}, got {v}")
    return v


def _bool(v: Any, where: str) -> bool:
    if not isinstance(v, bool):
        raise ConfigError(f"{where}: expected true/false, got {v!r}")
    return v


def _opt_str(v: Any, where: str) -> str | None:
    if v is None:
        return None
    if not isinstance(v, str):
        raise ConfigError(f"{where}: expected a string, got {v!r}")
    return v


def _str_list(v: Any, where: str) -> list[str]:
    if v is None:
        return []
    if isinstance(v, str):
        return [v]
    if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
        raise ConfigError(f"{where}: expected a list of strings")
    return list(v)


def _env_map(v: Any, where: str) -> dict[str, str]:
    d = _check_keys(v, set(v or {}), where)
    return {str(k): str(val) for k, val in d.items()}


def _vars_map(v: Any, where: str) -> dict[str, Any]:
    d = _check_keys(v, set(v or {}), where)
    for k in d:
        if not isinstance(k, str) or not k.isidentifier():
            raise ConfigError(f"{where}: variable name {k!r} must be a valid identifier")
    return dict(d)


# --------------------------------------------------------------------------- dataclasses


@dataclass
class Phase:
    run: str | None = None
    skip_if: str | None = None
    timeout: float | None = None
    stall_timeout: float | None = None
    clean: list[str] = field(default_factory=list)  # deleted before every attempt (non-resumable work)
    cpus: float | None = None  # None: the task's resources.cpus (project setup/epilogue: 1)


@dataclass
class Resources:
    vram_gb: float = 0.0
    min_gpus: int = 1
    max_gpus: int = 1
    cpus: float = 1.0
    ram_gb: float = 0.0  # 0: not tracked


@dataclass
class Retry:
    max_attempts: int = 3
    backoff: list[float] = field(default_factory=lambda: [30.0, 120.0, 600.0])
    reset_after: float | None = 3600.0  # a phase that ran this long before failing gets a fresh budget

    def delay(self, failures: int) -> float:
        if not self.backoff:
            return 0.0
        return self.backoff[min(max(failures, 1), len(self.backoff)) - 1]


@dataclass
class OomStep:
    vars: dict[str, Any] = field(default_factory=dict)
    env: dict[str, str] = field(default_factory=dict)
    resources: dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskTemplate:
    name: str
    index: int
    items: Any = None
    done_if: str | None = None
    vars: dict[str, Any] = field(default_factory=dict)
    env: dict[str, str] = field(default_factory=dict)
    resources: Resources = field(default_factory=Resources)
    setup: Phase = field(default_factory=Phase)
    run: Phase = field(default_factory=Phase)
    epilogue: Phase = field(default_factory=Phase)
    retry: Retry = field(default_factory=Retry)
    on_oom: list[OomStep] = field(default_factory=list)
    timeout: float | None = None
    stall_timeout: float | None = None
    priority: int = 0
    needs: list[str] = field(default_factory=list)
    enabled: bool = True

    def phase(self, name: str) -> Phase:
        return getattr(self, name)


@dataclass
class UvOptions:
    python: str | None = None
    extras: list[str] = field(default_factory=list)
    all_extras: bool = False
    groups: list[str] = field(default_factory=list)
    frozen: bool | None = None  # None: --frozen when uv.lock exists


@dataclass
class Project:
    name: str
    workdir: Path
    env_manager: str = "uv"  # uv | shell
    shell_prefix: str | None = None
    uv: UvOptions = field(default_factory=UvOptions)
    env: dict[str, str] = field(default_factory=dict)
    vars: dict[str, Any] = field(default_factory=dict)
    setup: Phase = field(default_factory=Phase)
    epilogue: Phase = field(default_factory=Phase)
    epilogue_when: str | None = None  # None: scheduler.project_epilogue_when
    retry: Retry = field(default_factory=Retry)
    tasks: list[TaskTemplate] = field(default_factory=list)
    source: str = ""  # the config file / generator that defined this project
    enabled: bool = True
    log_dir: str | None = None  # templated, relative to workdir; None: state_dir/logs/<project>

    def effective(self, tmpl: TaskTemplate, oom_level: int) -> tuple[dict, dict, Resources]:
        """vars, env and resources for a task after applying the first `oom_level` OOM steps."""
        vars_ = {**self.vars, **tmpl.vars}
        env = {**self.env, **tmpl.env}
        res = tmpl.resources
        for i, step in enumerate(tmpl.on_oom[:oom_level]):
            vars_.update(step.vars)
            env.update(step.env)
            res = _parse_resources(step.resources, f"{tmpl.name}.on_oom[{i}].resources", base=res)
        return vars_, env, res


@dataclass
class GpuLogCfg:
    """A machine-provided log of GPU processes (container name, MiB). Diagnostics only."""

    path: Path
    mine: str | None = None  # substring of your container names
    max_age: float = 600.0


@dataclass
class SchedulerCfg:
    state_dir: Path
    gpus: list[int] | None = None
    vram_safety_margin_gb: float = 2.0
    heartbeat_seconds: float = 30.0
    poll_seconds: float = 2.0
    gpu_poll_seconds: float = 5.0
    rescan_seconds: float = 300.0
    launch_warmup: float = 300.0
    max_cpu_phases: int = 2
    max_prepared: int = 2
    max_running: int | None = None
    max_gpu_util: float | None = None
    starvation: float | None = 3600.0
    oom_retry_delay: float = 10.0
    kill_grace: float = 20.0
    check_timeout: float = 120.0
    shell: list[str] | None = None
    uv: str = "uv"
    nvidia_smi: str = "nvidia-smi"
    keepalive_cmd: str | None = None
    keepalive_seconds: float = 60.0
    project_epilogue_when: str = "all_done"
    cpus: float | None = None  # None: what this container may use (cgroup quota / affinity)
    ram_gb: float | None = None  # None: not enforced
    max_jobs_per_gpu: int | dict[int, int] | None = None
    gpu_process_log: GpuLogCfg | None = None
    cpu_only: bool = False  # `fdgx run --cpu`
    oom_patterns: list[str] = field(default_factory=lambda: list(DEFAULT_OOM_PATTERNS))

    def jobs_cap(self, gpu: int) -> int | None:
        if isinstance(self.max_jobs_per_gpu, dict):
            return self.max_jobs_per_gpu.get(gpu)
        return self.max_jobs_per_gpu


EPILOGUE_WHEN = ("all_done", "all_finished")


@dataclass
class Config:
    path: Path
    scheduler: SchedulerCfg
    projects: list[Project]


# --------------------------------------------------------------------------- parsing


def _parse_phase(v: Any, where: str) -> Phase:
    if v is None:
        return Phase()
    if isinstance(v, str):
        return Phase(run=v)
    d = _check_keys(v, {"run", "skip_if", "timeout", "stall_timeout", "clean", "cpus"}, where)
    return Phase(
        run=_opt_str(d.get("run"), f"{where}.run"),
        skip_if=_opt_str(d.get("skip_if"), f"{where}.skip_if"),
        timeout=parse_duration(d.get("timeout"), f"{where}.timeout"),
        stall_timeout=parse_duration(d.get("stall_timeout"), f"{where}.stall_timeout"),
        clean=_str_list(d.get("clean"), f"{where}.clean"),
        cpus=None if d.get("cpus") is None else _num(d["cpus"], f"{where}.cpus"),
    )


def _parse_resources(v: Any, where: str, base: Resources | None = None) -> Resources:
    d = _check_keys(v, {"vram_gb", "gpus", "cpus", "ram_gb"}, where)
    res = replace(base) if base else Resources()
    if "vram_gb" in d:
        res.vram_gb = _num(d["vram_gb"], f"{where}.vram_gb")
    if "cpus" in d:
        res.cpus = _num(d["cpus"], f"{where}.cpus")
    if "ram_gb" in d:
        res.ram_gb = _num(d["ram_gb"], f"{where}.ram_gb")
    if "gpus" in d:
        g = d["gpus"]
        if isinstance(g, int) and not isinstance(g, bool):
            res.min_gpus = res.max_gpus = _int(g, f"{where}.gpus")
        else:
            gd = _check_keys(g, {"min", "max"}, f"{where}.gpus")
            res.min_gpus = _int(gd.get("min", res.min_gpus), f"{where}.gpus.min")
            res.max_gpus = _int(gd.get("max", max(res.max_gpus, res.min_gpus)), f"{where}.gpus.max")
    if res.min_gpus > res.max_gpus:
        raise ConfigError(f"{where}: gpus.min ({res.min_gpus}) > gpus.max ({res.max_gpus})")
    return res


def _parse_retry(v: Any, where: str) -> Retry:
    d = _check_keys(v, {"max_attempts", "backoff", "reset_after"}, where)
    r = Retry()
    if "reset_after" in d:
        r.reset_after = parse_duration(d["reset_after"], f"{where}.reset_after") or None
    if "max_attempts" in d:
        r.max_attempts = _int(d["max_attempts"], f"{where}.max_attempts", minimum=1)
    if "backoff" in d:
        b = d["backoff"]
        b = b if isinstance(b, list) else [b]
        r.backoff = [parse_duration(x, f"{where}.backoff") for x in b]
    return r


def _parse_oom_step(v: Any, where: str) -> OomStep:
    if not isinstance(v, dict):
        raise ConfigError(f"{where}: expected a mapping")
    reserved = {"vars", "env", "resources"}
    if not set(v) & reserved:  # shorthand: `- {batch_size: 16}` means vars
        return OomStep(vars=_vars_map(v, where))
    d = _check_keys(v, reserved, where)
    step = OomStep(
        vars=_vars_map(d.get("vars"), f"{where}.vars"),
        env=_env_map(d.get("env"), f"{where}.env"),
        resources=dict(d.get("resources") or {}),
    )
    _parse_resources(step.resources, f"{where}.resources")  # validate early
    return step


_TASK_KEYS = {
    "name", "items", "done_if", "vars", "env", "resources", "setup", "run", "epilogue",
    "retry", "on_oom", "timeout", "stall_timeout", "priority", "needs", "enabled",
}


def _parse_task(v: Any, index: int, where: str) -> TaskTemplate:
    d = _check_keys(v, _TASK_KEYS, where)
    name = d.get("name")
    if not isinstance(name, str) or not name:
        raise ConfigError(f"{where}: 'name' is required")
    where = f"{where} ({name})"
    if d.get("run") is None:
        raise ConfigError(f"{where}: 'run' is required")
    items = d.get("items")
    if items is not None and "{" not in name:
        name += "-{item}"
    on_oom = d.get("on_oom") or []
    if not isinstance(on_oom, list):
        raise ConfigError(f"{where}.on_oom: expected a list")
    return TaskTemplate(
        name=name,
        index=index,
        items=items,
        done_if=_opt_str(d.get("done_if"), f"{where}.done_if"),
        vars=_vars_map(d.get("vars"), f"{where}.vars"),
        env=_env_map(d.get("env"), f"{where}.env"),
        resources=_parse_resources(d.get("resources"), f"{where}.resources"),
        setup=_parse_phase(d.get("setup"), f"{where}.setup"),
        run=_parse_phase(d.get("run"), f"{where}.run"),
        epilogue=_parse_phase(d.get("epilogue"), f"{where}.epilogue"),
        retry=_parse_retry(d.get("retry"), f"{where}.retry"),
        on_oom=[_parse_oom_step(s, f"{where}.on_oom[{i}]") for i, s in enumerate(on_oom)],
        timeout=parse_duration(d.get("timeout"), f"{where}.timeout"),
        stall_timeout=parse_duration(d.get("stall_timeout"), f"{where}.stall_timeout"),
        priority=_int(d.get("priority", 0), f"{where}.priority", minimum=None),
        needs=_str_list(d.get("needs"), f"{where}.needs"),
        enabled=_bool(d.get("enabled", True), f"{where}.enabled"),
    )


def _parse_uv(v: Any, where: str) -> UvOptions:
    d = _check_keys(v, {"python", "extras", "all_extras", "groups", "frozen"}, where)
    frozen = d.get("frozen")
    if frozen is not None and not isinstance(frozen, bool):
        raise ConfigError(f"{where}.frozen: expected true/false")
    return UvOptions(
        python=_opt_str(d.get("python"), f"{where}.python"),
        extras=_str_list(d.get("extras"), f"{where}.extras"),
        all_extras=bool(d.get("all_extras", False)),
        groups=_str_list(d.get("groups"), f"{where}.groups"),
        frozen=frozen,
    )


def _parse_project_epilogue(v: Any, where: str) -> tuple[Phase, str | None]:
    when = None
    if isinstance(v, dict) and "when" in v:
        v = dict(v)
        when = v.pop("when")
        if when not in EPILOGUE_WHEN:
            raise ConfigError(f"{where}.when: must be one of {', '.join(EPILOGUE_WHEN)}")
    return _parse_phase(v, where), when


def _parse_project(name: str, v: Any, base: Path, *, default_workdir: Path | None = None,
                   source: str = "") -> Project:
    where = f"projects.{name}"
    if not re.fullmatch(r"[A-Za-z0-9._-]+", str(name)):
        raise ConfigError(f"{where}: project names may only contain letters, digits, '.', '_' and '-'")
    d = _check_keys(
        v, {"workdir", "env_manager", "shell_prefix", "uv", "env", "vars", "setup", "epilogue", "retry", "tasks",
            "enabled", "log_dir"},
        where,
    )
    workdir = d.get("workdir")
    if workdir is None and default_workdir is not None:
        workdir = str(default_workdir)
    if not isinstance(workdir, str):
        raise ConfigError(f"{where}.workdir is required")
    env_manager = d.get("env_manager", "uv")
    if env_manager not in ("uv", "shell"):
        raise ConfigError(f"{where}.env_manager: must be 'uv' or 'shell'")
    tasks = d.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ConfigError(f"{where}.tasks: expected a non-empty list")
    epilogue, epilogue_when = _parse_project_epilogue(d.get("epilogue"), f"{where}.epilogue")
    templates = [_parse_task(t, i, f"{where}.tasks[{i}]") for i, t in enumerate(tasks)]
    dup = {t.name for t in templates if sum(o.name == t.name for o in templates) > 1}
    if dup:
        raise ConfigError(f"{where}.tasks: duplicate task names {sorted(dup)}")
    return Project(
        name=str(name),
        workdir=(base / Path(workdir).expanduser()).resolve(),
        env_manager=env_manager,
        shell_prefix=_opt_str(d.get("shell_prefix"), f"{where}.shell_prefix"),
        uv=_parse_uv(d.get("uv"), f"{where}.uv"),
        env=_env_map(d.get("env"), f"{where}.env"),
        vars=_vars_map(d.get("vars"), f"{where}.vars"),
        setup=_parse_phase(d.get("setup"), f"{where}.setup"),
        epilogue=epilogue,
        epilogue_when=epilogue_when,
        retry=_parse_retry(d.get("retry"), f"{where}.retry"),
        tasks=templates,
        source=source,
        enabled=_bool(d.get("enabled", True), f"{where}.enabled"),
        log_dir=_opt_str(d.get("log_dir"), f"{where}.log_dir"),
    )


_SCHED_KEYS = {
    "state_dir", "gpus", "vram_safety_margin_gb", "heartbeat_seconds", "poll_seconds", "gpu_poll_seconds",
    "rescan_seconds", "launch_warmup", "max_cpu_phases", "max_prepared", "max_running", "max_gpu_util",
    "starvation", "oom_retry_delay", "kill_grace", "check_timeout", "shell", "uv", "nvidia_smi",
    "keepalive_cmd", "keepalive_seconds", "extra_oom_patterns", "project_epilogue_when", "cpus", "ram_gb",
    "max_jobs_per_gpu", "gpu_process_log",
}


def _parse_scheduler(v: Any, base: Path) -> SchedulerCfg:
    w = "scheduler"
    d = _check_keys(v, _SCHED_KEYS, w)
    sc = SchedulerCfg(state_dir=(base / Path(d.get("state_dir", ".fdgx")).expanduser()).resolve())
    gpus = d.get("gpus", "all")
    if gpus != "all" and gpus is not None:
        if not isinstance(gpus, list):
            raise ConfigError(f"{w}.gpus: expected a list of GPU indices or 'all'")
        sc.gpus = [_int(g, f"{w}.gpus") for g in gpus]
    for key in ("vram_safety_margin_gb",):
        if key in d:
            setattr(sc, key, _num(d[key], f"{w}.{key}"))
    for key in ("heartbeat_seconds", "poll_seconds", "gpu_poll_seconds", "rescan_seconds", "launch_warmup",
                "oom_retry_delay", "kill_grace", "check_timeout", "keepalive_seconds"):
        if key in d:
            setattr(sc, key, parse_duration(d[key], f"{w}.{key}"))
    if "starvation" in d:
        sc.starvation = parse_duration(d["starvation"], f"{w}.starvation") or None
    for key in ("max_cpu_phases", "max_prepared"):
        if key in d:
            setattr(sc, key, _int(d[key], f"{w}.{key}", minimum=1))
    if d.get("max_running") is not None:
        sc.max_running = _int(d["max_running"], f"{w}.max_running", minimum=1)
    if d.get("max_gpu_util") is not None:
        sc.max_gpu_util = _num(d["max_gpu_util"], f"{w}.max_gpu_util")
    shell = d.get("shell")
    if shell is not None:
        sc.shell = shlex.split(shell) if isinstance(shell, str) else _str_list(shell, f"{w}.shell")
    for key in ("uv", "nvidia_smi", "keepalive_cmd"):
        if key in d:
            setattr(sc, key, _opt_str(d[key], f"{w}.{key}"))
    if "project_epilogue_when" in d:
        if d["project_epilogue_when"] not in EPILOGUE_WHEN:
            raise ConfigError(f"{w}.project_epilogue_when: must be one of {', '.join(EPILOGUE_WHEN)}")
        sc.project_epilogue_when = d["project_epilogue_when"]
    if d.get("cpus") is not None:
        sc.cpus = _num(d["cpus"], f"{w}.cpus")
    if d.get("ram_gb") is not None:
        sc.ram_gb = _num(d["ram_gb"], f"{w}.ram_gb")
    cap = d.get("max_jobs_per_gpu")
    if isinstance(cap, dict):
        sc.max_jobs_per_gpu = {_int(k, f"{w}.max_jobs_per_gpu"): _int(n, f"{w}.max_jobs_per_gpu.{k}")
                               for k, n in cap.items()}
    elif cap is not None:
        sc.max_jobs_per_gpu = _int(cap, f"{w}.max_jobs_per_gpu", minimum=1)
    log = d.get("gpu_process_log")
    if log is not None:
        ld = {"path": log} if isinstance(log, str) else _check_keys(log, {"path", "mine", "max_age"},
                                                                    f"{w}.gpu_process_log")
        if not isinstance(ld.get("path"), str):
            raise ConfigError(f"{w}.gpu_process_log.path is required")
        sc.gpu_process_log = GpuLogCfg(
            path=(base / Path(ld["path"]).expanduser()).resolve(),
            mine=_opt_str(ld.get("mine"), f"{w}.gpu_process_log.mine"),
            max_age=parse_duration(ld.get("max_age", 600), f"{w}.gpu_process_log.max_age"),
        )
    sc.oom_patterns += _str_list(d.get("extra_oom_patterns"), f"{w}.extra_oom_patterns")
    for p in sc.oom_patterns:
        try:
            re.compile(p)
        except re.error as e:
            raise ConfigError(f"{w}: bad OOM pattern {p!r}: {e}") from None
    return sc


def parse_projects(raw: Any, base: Path, source: str) -> list[Project]:
    """The `projects:` mapping of one config source. Paths and default workdir are relative to `base`."""
    if raw is None:
        return []
    if not isinstance(raw, dict):
        raise ConfigError("'projects' must be a mapping of project name -> project")
    return [_parse_project(n, p, base, default_workdir=base, source=source) for n, p in raw.items()]


def load_config(path: str | Path) -> Config:
    """Strict load (main file + includes): any error raises ConfigError."""
    from .loader import load

    res = load(path)
    if res.errors:
        raise ConfigError("\n".join(res.errors))
    return res.config


# --------------------------------------------------------------------------- items & templates


class Item:
    """A work item as seen by templates: `{item}`, `{item.stem}`, `{item.index}`, `{item.key}` for dict items."""

    _PATH_ATTRS = ("stem", "name", "suffix", "parent")

    def __init__(self, value: Any, index: int):
        self.value = value
        self.index = index

    def __str__(self) -> str:
        if isinstance(self.value, (dict, list)):
            return json.dumps(self.value, sort_keys=True)
        return str(self.value)

    def __format__(self, spec: str) -> str:
        return format(str(self), spec)

    def __getitem__(self, key: Any) -> Any:
        return self.value[key]

    def __getattr__(self, key: str) -> Any:
        if key.startswith("_"):
            raise AttributeError(key)
        if isinstance(self.value, dict) and key in self.value:
            return self.value[key]
        if key in self._PATH_ATTRS:
            return str(getattr(PurePosixPath(str(self.value)), key))
        raise AttributeError(f"item has no attribute {key!r}")


def _parse_lines(text: str) -> list[Any]:
    out: list[Any] = []
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if s.startswith("{"):
            try:
                out.append(json.loads(s))
                continue
            except json.JSONDecodeError:
                pass
        out.append(s)
    return out


def expand_items(spec: Any, workdir: Path, run_command: Callable[[str], str] | None = None) -> list[Any]:
    """Turn an `items:` spec into a list of raw item values ([None] when there is no spec)."""
    if spec is None:
        return [None]
    if isinstance(spec, list):
        return list(spec)
    d = _check_keys(spec, {"glob", "command", "range", "file"}, "items")
    if len(d) != 1:
        raise ConfigError("items: use exactly one of glob / command / range / file (or a plain list)")
    kind, val = next(iter(d.items()))
    if kind == "glob":
        pats = val if isinstance(val, list) else [val]
        found: set[str] = set()
        for pat in pats:
            found.update(Path(p).as_posix() for p in glob.glob(str(pat), root_dir=workdir, recursive=True))
        return sorted(found)
    if kind == "range":
        args = val if isinstance(val, list) else [val]
        if not 1 <= len(args) <= 3 or not all(isinstance(a, int) for a in args):
            raise ConfigError("items.range: expected N, [start, stop] or [start, stop, step]")
        return list(range(*args))
    if kind == "file":
        return _parse_lines((workdir / str(val)).read_text(encoding="utf-8"))
    if run_command is None:
        raise ConfigError("items.command cannot be expanded here")
    return _parse_lines(run_command(str(val)))


class _Context(dict):
    def __missing__(self, key: str) -> Any:
        raise ConfigError(f"unknown template variable {{{key}}}")


def render(template: str, ctx: dict[str, Any], where: str = "template") -> str:
    try:
        return template.format_map(_Context(ctx))
    except ConfigError as e:
        raise ConfigError(f"{where}: {e}; available: {', '.join(sorted(ctx))}") from None
    except (AttributeError, IndexError, KeyError, TypeError, ValueError) as e:
        raise ConfigError(f"{where}: cannot render {template!r}: {e} (write {{{{ and }}}} for literal braces)") from None


@dataclass
class TaskSpec:
    """One concrete task produced by a template and an item."""

    tmpl: TaskTemplate
    item: Item | None
    name: str
    needs: list[str]


def expand_tasks(
    project: Project, tmpl: TaskTemplate, run_command: Callable[[str], str] | None = None
) -> tuple[list[TaskSpec], list[str]]:
    """Expand a template into tasks (name and `needs` rendered per item). Returns (tasks, warnings)."""
    items = expand_items(tmpl.items, project.workdir, run_command)
    out: list[TaskSpec] = []
    warnings: list[str] = []
    seen: set[str] = set()
    for i, raw in enumerate(items):
        item = Item(raw, i) if tmpl.items is not None else None
        ctx = {**project.vars, **tmpl.vars, "item": item, "project": project.name}
        where = f"{project.name}.{tmpl.name}"
        name = render(tmpl.name, ctx, f"{where}.name")
        if name in seen:
            warnings.append(f"duplicate task name {project.name}/{name} (item {raw!r}) - skipped")
            continue
        seen.add(name)
        needs = [render(n, ctx, f"{where}.needs") for n in tmpl.needs]
        out.append(TaskSpec(tmpl, item, name, needs))
    return out, warnings
