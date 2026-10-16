"""Load the main config and its includes (YAML files and generators), isolating errors per source."""

from __future__ import annotations

import glob
import os
import shlex
import subprocess
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .config import Config, ConfigError, Project, SchedulerCfg, _check_keys, _parse_scheduler, parse_projects
from .envs import clean_base_env, default_shell, thread_env

MAIN = "<main>"


@dataclass
class LoadResult:
    config: Config | None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class _Source:
    key: str  # stable identity: file path, or "command:<cmd>@<dir>"
    kind: str  # yaml | command
    base: Path  # relative paths and default workdir
    path: Path | None = None
    command: str | None = None
    env_manager: str | None = None  # generators: uv | shell (None: uv when a pyproject.toml is found)


def _read_yaml(path: Path) -> Any:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as e:
        raise ConfigError(f"cannot read {path}: {e}") from None


def _uv_project_of(path: Path) -> Path | None:
    for d in (path, *path.parents):
        if (d / "pyproject.toml").is_file():
            return d
    return None


def generator_argv(command: str, cwd: Path, sc: SchedulerCfg, env_manager: str | None = None) -> list[str]:
    """argv for a config generator: the uv project containing `cwd`, or a plain shell.

    Loading config never installs anything: uv runs with --no-sync (which also implies --frozen), so a
    venv uv doesn't fully own (driver-matched torch, pinned numpy, ...) is never modified.
    """
    shell = sc.shell or default_shell()
    uv_project = _uv_project_of(cwd) if env_manager in (None, "uv") else None
    if env_manager == "uv" and uv_project is None:
        raise ConfigError(f"env_manager uv: no pyproject.toml in {cwd} or above")
    if uv_project is not None:
        return [sc.uv, "run", "--no-sync", "--project", str(uv_project), "--", *shell, command]
    return [*shell, command]


def run_generator(command: str, cwd: Path, sc: SchedulerCfg, env_manager: str | None = None) -> str:
    """Run a config generator (see generator_argv). Returns stdout."""
    argv = generator_argv(command, cwd, sc, env_manager)
    try:
        r = subprocess.run(
            argv, cwd=cwd, env={**clean_base_env(), **thread_env(1)}, stdin=subprocess.DEVNULL,
            capture_output=True, text=True,
            timeout=sc.check_timeout,
        )
    except subprocess.TimeoutExpired:
        raise ConfigError(f"generator timed out after {sc.check_timeout:.0f}s: {command}") from None
    except OSError as e:
        raise ConfigError(f"generator could not run ({e}): {command}") from None
    if r.returncode != 0:
        tail = "\n".join((r.stderr or r.stdout).strip().splitlines()[-5:])
        raise ConfigError(f"generator exited {r.returncode}: {command}\n{tail}")
    return r.stdout


def _expand_includes(raw: Any, main_dir: Path, errors: list[str], warnings: list[str]) -> list[_Source]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        errors.append("include: expected a list")
        return []
    out: list[_Source] = []
    for i, entry in enumerate(raw):
        where = f"include[{i}]"
        if isinstance(entry, dict):
            try:
                d = _check_keys(entry, {"command", "dir", "env_manager"}, where)
                if not isinstance(d.get("command"), str):
                    raise ConfigError(f"{where}: 'command' is required")
                if d.get("env_manager") not in (None, "uv", "shell"):
                    raise ConfigError(f"{where}.env_manager: must be 'uv' or 'shell'")
                cwd = (main_dir / Path(os.path.expanduser(str(d.get("dir", "."))))).resolve()
            except ConfigError as e:
                errors.append(str(e))
                continue
            out.append(_Source(f"command:{d['command']}@{cwd}", "command", cwd, command=d["command"],
                               env_manager=d.get("env_manager")))
            continue
        if not isinstance(entry, str):
            errors.append(f"{where}: expected a path, a glob or {{command, dir, env_manager}}")
            continue
        pattern = str(main_dir / os.path.expanduser(entry))
        if any(c in entry for c in "*?["):
            paths = [Path(p) for p in sorted(glob.glob(pattern, recursive=True))]
            paths = [p for p in paths if p.is_file() and not p.name.startswith(".") and not p.name.endswith("~")]
            if not paths:
                warnings.append(f"include {entry!r} matches no files")
        else:
            paths = [Path(pattern)]
        for p in paths:
            p = p.resolve()
            if p.suffix == ".py":
                out.append(_Source(str(p), "command", p.parent, path=p, command=f"python {shlex.quote(p.name)}"))
            else:
                out.append(_Source(str(p), "yaml", p.parent, path=p))
    return out


def _load_source(src: _Source, sc: SchedulerCfg) -> list[Project]:
    if src.kind == "yaml":
        if not src.path.is_file():
            raise ConfigError(f"included file {src.path} does not exist")
        raw = _read_yaml(src.path)
    else:
        if src.path is not None and not src.path.is_file():
            raise ConfigError(f"included file {src.path} does not exist")
        try:
            raw = yaml.safe_load(run_generator(src.command, src.base, sc, src.env_manager))
        except yaml.YAMLError as e:
            raise ConfigError(f"generator output is not valid JSON/YAML: {e}") from None
    if raw is None:
        return []
    if isinstance(raw, dict) and "scheduler" in raw:
        raise ConfigError("'scheduler' settings belong in the main config file only")
    d = _check_keys(raw, {"projects"}, "top level")
    return parse_projects(d.get("projects"), src.base, src.key)


def load(path: str | Path, previous: Config | None = None) -> LoadResult:
    """Load the main config and every include.

    A broken main file keeps `previous` entirely; a broken include keeps the projects it defined in
    `previous`. Project names must be unique across sources: on a clash, the source that already owned
    the name in `previous` (else the first one listed) wins and the other keeps its previous projects.
    """
    path = Path(path).resolve()
    errors: list[str] = []
    warnings: list[str] = []
    try:
        raw = _read_yaml(path) or {}
        d = _check_keys(raw, {"scheduler", "include", "projects"}, f"{path.name}: top level")
        sc = _parse_scheduler(d.get("scheduler"), path.parent)
    except ConfigError as e:
        return LoadResult(previous, [f"{path}: {e}"], [])

    prev_by_source: dict[str, list[Project]] = defaultdict(list)
    prev_owner: dict[str, str] = {}
    for p in previous.projects if previous else []:
        prev_by_source[p.source].append(p)
        prev_owner[p.name] = p.source

    loaded: dict[str, list[Project]] = {}
    try:
        loaded[MAIN] = parse_projects(d.get("projects"), path.parent, MAIN)
    except ConfigError as e:
        errors.append(f"{path}: {e}")
        loaded[MAIN] = prev_by_source[MAIN]
    for src in _expand_includes(d.get("include"), path.parent, errors, warnings):
        if src.key in loaded:
            warnings.append(f"{src.key} is included twice")
            continue
        try:
            loaded[src.key] = _load_source(src, sc)
        except (ConfigError, OSError) as e:
            errors.append(f"{src.key}: {e}")
            loaded[src.key] = prev_by_source[src.key]

    excluded: dict[str, set[str]] = defaultdict(set)
    while True:  # resolve name clashes; each pass excludes one more (source, name) pair
        owners: dict[str, list[str]] = defaultdict(list)
        for key, projects in loaded.items():
            for p in projects:
                owners[p.name].append(key)
        clashes = {n: ks for n, ks in owners.items() if len(ks) > 1}
        if not clashes:
            break
        for name, keys in clashes.items():
            winner = prev_owner[name] if prev_owner.get(name) in keys else keys[0]
            for k in keys:
                if k != winner:
                    errors.append(f"{k}: project {name!r} is already defined by {winner}")
                    excluded[k].add(name)
                    loaded[k] = [p for p in prev_by_source[k] if p.name not in excluded[k]]
            break

    projects = [p for ps in loaded.values() for p in ps]
    if not projects and not errors:
        warnings.append("no projects defined (use 'projects:' or 'include:')")
    return LoadResult(Config(path=path, scheduler=sc, projects=projects), errors, warnings)
