"""Command line: fdgx run | status | list | retry | reset | hold | release | stop | reload | gpus."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter

from . import control, deps, gpulog
from .config import Config, ConfigError, expand_tasks, render
from .gpu import GpuProbe
from .loader import load
from .state import FileLock, StateStore

REPLY_TIMEOUT = 15.0


def _stdout_safe() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace", line_buffering=True)
        except (AttributeError, ValueError):
            pass


def _load_lenient(path: str) -> Config:
    """Load what can be loaded; report broken sources on stderr. Raises only if the main file is unusable."""
    res = load(path)
    if res.config is None:
        raise ConfigError("\n".join(res.errors))
    for e in res.errors:
        print(f"config error: {e}", file=sys.stderr)
    for w in res.warnings:
        print(f"config warning: {w}", file=sys.stderr)
    return res.config


def cmd_run(args) -> int:
    from .scheduler import Scheduler

    res = load(args.config)
    if res.config is None:
        for e in res.errors:
            print(f"config error: {e}", file=sys.stderr)
        return 1
    res.config.scheduler.cpu_only = args.cpu
    return Scheduler(res.config, stay_alive=args.stay_alive, load_errors=res.errors,
                     load_warnings=res.warnings).run()


def _live_status(state_dir) -> dict:
    try:
        return json.loads((state_dir / "status.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def cmd_status(args) -> int:
    cfg = _load_lenient(args.config)
    state_dir = cfg.scheduler.state_dir
    store = StateStore(state_dir)
    running = FileLock(state_dir / "fdgx.lock").held_by_other()
    print(f"scheduler: {'RUNNING' if running else 'not running'}   state: {state_dir}")
    live = _live_status(state_dir)
    derived = live.get("tasks", {})
    if live:
        age = time.time() - live.get("time", 0)
        print(f"conditions as of {age:.0f}s ago" + ("" if running else " (last run)"))
        for e in live.get("config_errors", []):
            print(f"config error: {e.splitlines()[0]}")
    states = [st for _, st in store.all()]
    if not states:
        print("no tasks discovered yet")
        return 0
    now = time.time()
    by_project: dict[str, list] = {}
    for st in states:
        by_project.setdefault(st.id.split("/", 1)[0], []).append(st)
    for project, sts in sorted(by_project.items()):
        def label(st):
            if st.running:
                return "running " + st.phase
            if st.held and st.phase != "done":
                return "held"
            d = derived.get(st.id, {})
            if d.get("condition") == "disabled":
                return "disabled"
            return d.get("condition", st.phase) if st.phase == "setup" else st.phase

        counts = Counter(label(st) for st in sts)
        print(f"\n{project}: " + ", ".join(f"{n} {k}" for k, n in sorted(counts.items())))
        for st in sts:
            d = derived.get(st.id, {})
            if st.running:
                print(f"  RUN     {st.id} [{st.phase}] gpus {st.gpus or '-'}" + (" (held)" if st.held else ""))
            elif st.held and st.phase != "done":
                print(f"  HELD    {st.id} [{st.phase}]")
            elif st.phase == "failed":
                err = (st.last_error or "").strip().splitlines()
                print(f"  FAILED  {st.id} [{st.failed_phase}] {err[0] if err else ''}")
            elif st.phase == "setup" and d.get("condition") == "blocked":
                print(f"  BLOCKED {st.id}: {d.get('reason', '')}")
            elif args.verbose and st.phase == "setup" and d.get("condition") == "waiting":
                print(f"  waiting {st.id}: {d.get('reason', '')}")
            elif args.verbose or (st.phase != "done" and st.next_at > now):
                extra = f" retry in {st.next_at - now:.0f}s" if st.next_at > now else ""
                extra += f" oom_level {st.oom_level}" if st.oom_level else ""
                print(f"  {st.phase:<7} {st.id}{extra}")
    return 0


def cmd_list(args) -> int:
    res = load(args.config)
    for e in res.errors:
        print(f"config error: {e}", file=sys.stderr)
    cfg = res.config
    if cfg is None:
        return 1
    store = StateStore(cfg.scheduler.state_dir)
    problems = list(res.errors)
    needs: dict[str, list[str]] = {}
    known: set[str] = set()
    for p in cfg.projects:
        print(f"{p.name}  ({p.workdir}, env: {p.env_manager})")
        for tmpl in p.tasks:
            try:
                specs, warnings = expand_tasks(p, tmpl)
            except ConfigError as e:
                print(f"  {tmpl.name}: tasks not listable here ({e})")
                continue
            for w in warnings:
                print(f"  warning: {w}")
            vars_, _, rs = p.effective(tmpl, 0)
            for spec in specs:
                tid = f"{p.name}/{spec.name}"
                known.add(tid)
                if spec.needs:
                    needs[tid] = spec.needs
                st = store.load(p.name, spec.name)
                line = f"  {spec.name:<40} {st.phase if st else 'new':<8} {rs.min_gpus}-{rs.max_gpus} x {rs.vram_gb:g} GB"
                line += f", {rs.cpus:g} cpus" if rs.cpus != 1 else ""
                line += f", {rs.ram_gb:g} GB RAM" if rs.ram_gb else ""
                if args.verbose:
                    ctx = {**vars_, "item": spec.item, "project": p.name, "workdir": str(p.workdir),
                           "task": spec.name, "phase": "run", "gpus": "<gpus>", "num_gpus": rs.max_gpus,
                           "attempt": 1, "oom_level": 0}
                    try:
                        cmd = render(tmpl.run.run, ctx, f"{tid}.run")
                    except ConfigError as e:
                        cmd = f"<{e}>"
                        problems.append(str(e))
                    line += f" | needs: {', '.join(spec.needs) or '-'} | run: {cmd}"
                print(line)
    resolved = deps.resolve(needs, known, {p.name: "ready" for p in cfg.projects})
    for tid, (cond, reason) in sorted(resolved.issues.items()):
        if cond == deps.BLOCKED:
            problems.append(f"{tid}: {reason}")
        else:
            print(f"note: {tid}: {reason}", file=sys.stderr)
    for comp in deps.cycles(resolved.edges):
        problems.append("dependency cycle: " + " -> ".join(comp + [comp[0]]))
    for prob in problems[len(res.errors):]:
        print(f"error: {prob}", file=sys.stderr)
    return 1 if problems else 0


def _control(args, op: str, **opts) -> int:
    """Send `op` to the running scheduler, or apply it to the state files when none runs."""
    cfg = _load_lenient(args.config)
    state_dir = cfg.scheduler.state_dir
    patterns = getattr(args, "patterns", []) or []
    lock = FileLock(state_dir / "fdgx.lock")
    if not lock.held_by_other():
        if op in ("stop", "reload"):
            print("scheduler not running")
            return 1
        if op != "retry" and not patterns:
            print(f"{op} needs at least one task pattern, e.g. 'train-*' or 'proj/*'")
            return 1
        for line in control.apply_offline(StateStore(state_dir), op, patterns, **opts):
            print(line)
        return 0
    rid = control.submit(state_dir, op, patterns, **opts)
    rep = control.wait_reply(state_dir, rid, REPLY_TIMEOUT)
    if rep is None:
        print("request queued; the running scheduler will apply it at its next loop")
        return 0
    for line in rep.get("lines", []):
        print(line)
    if op == "stop" and rep.get("ok"):
        deadline = time.time() + cfg.scheduler.kill_grace + 15
        while time.time() < deadline and lock.held_by_other():
            time.sleep(0.2)
        print("scheduler stopped" if not lock.held_by_other() else "scheduler is still stopping")
    return 0 if rep.get("ok") else 1


def cmd_hold(args) -> int:
    return _control(args, "hold", kill=args.kill)


def cmd_release(args) -> int:
    return _control(args, "release")


def cmd_retry(args) -> int:
    return _control(args, "retry", reset_oom=args.reset_oom)


def cmd_reset(args) -> int:
    return _control(args, "reset")


def cmd_stop(args) -> int:
    return _control(args, "stop")


def cmd_reload(args) -> int:
    return _control(args, "reload")


def cmd_gpus(args) -> int:
    margin = 0.0
    probe = GpuProbe()
    cfg = None
    if args.config:
        cfg = _load_lenient(args.config)
        probe = GpuProbe(cfg.scheduler.nvidia_smi)
        margin = cfg.scheduler.vram_safety_margin_gb
    snap = probe.probe()
    if snap.error:
        print(f"GPU probe failed: {snap.error}")
    for g in snap.gpus.values():
        util = f"{g.util:.0f}%" if g.util is not None else "?"
        print(f"GPU {g.index}: {g.free_gb:6.1f} / {g.total_gb:6.1f} GB free (usable {g.free_gb - margin:6.1f}), util {util}")
    for gi, pid, used in snap.apps:
        print(f"  pid {pid} on GPU {gi}: {used:.1f} GB")
    log_cfg = cfg.scheduler.gpu_process_log if cfg else None
    if log_cfg is not None:
        log = gpulog.read(log_cfg)
        print(f"\nGPU processes by container, from {log_cfg.path} (not authoritative: may list ghosts)")
        if log.error:
            print(f"  {log.error}")
        else:
            needle = log_cfg.mine or gpulog.default_mine()
            print(f"  {log.age:.0f}s old; '*' = yours" + ("" if needle else " (set gpu_process_log.mine to mark yours)"))
            for container, gb, n in log.by_container():
                mark = "*" if needle and needle in container else " "
                print(f"  {mark} {gb:7.1f} GB  {n:>2} proc  {container}")
            for _, msg in gpulog.ghost_warnings(log, needle, 1):
                print(f"  warning: {msg}")
    return 1 if snap.error else 0


def main(argv: list[str] | None = None) -> int:
    _stdout_safe()
    ap = argparse.ArgumentParser(prog="fdgx", description="Restart-safe GPU job scheduler.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("run", help="run the scheduler until all work is done")
    p.add_argument("config")
    p.add_argument("--stay-alive", action="store_true", help="keep running (and rescanning items) when idle")
    p.add_argument("--cpu", action="store_true",
                   help="dry run without GPUs: every task runs as CPU-only with CUDA_VISIBLE_DEVICES=-1")
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("status", help="show task states from disk")
    p.add_argument("config")
    p.add_argument("-v", "--verbose", action="store_true", help="list every task")
    p.set_defaults(fn=cmd_status)

    p = sub.add_parser("list", help="expand items and show the tasks the config defines")
    p.add_argument("config")
    p.add_argument("-v", "--verbose", action="store_true", help="also show rendered run commands")
    p.set_defaults(fn=cmd_list)

    pattern_help = "task globs: 'train-*' matches names in any project, 'proj/*' full ids"

    p = sub.add_parser("retry", help="re-queue failed tasks (from the phase that failed)")
    p.add_argument("config")
    p.add_argument("patterns", nargs="*", help=pattern_help + " (default: all failed)")
    p.add_argument("--reset-oom", action="store_true", help="also drop the on_oom level back to 0")
    p.set_defaults(fn=cmd_retry)

    p = sub.add_parser("reset", help="forget the state of tasks so they start over")
    p.add_argument("config")
    p.add_argument("patterns", nargs="+", help=pattern_help)
    p.set_defaults(fn=cmd_reset)

    p = sub.add_parser("hold", help="park tasks: they start no new phase until released")
    p.add_argument("config")
    p.add_argument("patterns", nargs="+", help=pattern_help)
    p.add_argument("--kill", action="store_true", help="also kill a running phase (it resumes on release)")
    p.set_defaults(fn=cmd_hold)

    p = sub.add_parser("release", help="un-park held tasks")
    p.add_argument("config")
    p.add_argument("patterns", nargs="+", help=pattern_help)
    p.set_defaults(fn=cmd_release)

    p = sub.add_parser("stop", help="stop the running scheduler and kill all its jobs")
    p.add_argument("config")
    p.set_defaults(fn=cmd_stop)

    p = sub.add_parser("reload", help="make the running scheduler reload its config now")
    p.add_argument("config")
    p.set_defaults(fn=cmd_reload)

    p = sub.add_parser("gpus", help="show what the scheduler sees on the GPUs")
    p.add_argument("config", nargs="?")
    p.set_defaults(fn=cmd_gpus)

    args = ap.parse_args(argv)
    try:
        return args.fn(args)
    except ConfigError as e:
        print(f"config error: {e}", file=sys.stderr)
        return 1
