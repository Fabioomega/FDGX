# FDGX

A small, restart-safe GPU job scheduler for **shared, unreliable machines**.

FDGX was built for a DGX instance where:

- the whole instance is killed as soon as it stops printing (or looks idle), while everything on disk survives;
- other people grab and release GPUs and VRAM at any moment;
- Docker/VMs are not allowed, so jobs run as plain processes;
- every job is written to be restartable from wherever it stopped.

You describe your projects, their tasks and the dependencies between them in YAML (or generate them
in Python), each project in its own repo. FDGX works out what is still pending, picks GPUs based on the
VRAM that is free *right now*, runs tasks once what they need is done, retries failures (changing the
configuration on out-of-memory errors), and keeps printing a heartbeat the whole time. Edit the config
while it runs and it picks the changes up. Kill it at any moment, start it again, and it carries on
where it was.

## Quick start

```bash
git clone <this repo> fdgx && cd fdgx
cp examples/fdgx.yaml fdgx.yaml      # scheduler settings + the list of project files to include
./bootstrap.sh fdgx.yaml
```

`bootstrap.sh` is safe to run every time the instance comes back. It:

1. installs [uv](https://docs.astral.sh/uv/) if missing (pin a version with `FDGX_UV_VERSION=0.7.13`);
2. keeps uv, its Python interpreters and its wheel cache under `FDGX_HOME` (default `./.fdgx-home`),
   so a restart re-downloads nothing;
3. syncs FDGX's own environment and starts `fdgx run fdgx.yaml`.

The scheduler exits when nothing is left that can run (exit code `0` = all done, or only parked work
left; `2` = a task failed or is blocked, or a project's setup or epilogue failed). Pass `--stay-alive` to keep it running and picking up new tasks and config changes:
`./bootstrap.sh fdgx.yaml --stay-alive`.

Try it without GPUs using fake ones:

```bash
FDGX_FAKE_GPUS="0:80,1:80" uv run fdgx run examples/fdgx.yaml
```

## Concepts

```
fdgx.yaml      scheduler settings + include: [project files / generators]
Project        workdir, environment (uv or shell), env vars
├─ setup       once per scheduler start, before any task       (e.g. uv sync + download weights)
├─ Task        one unit of work per item: a config file, dataset, seed...; may `need` other tasks
│  ├─ setup     optional, CPU; e.g. preprocess              (runs ahead while GPUs are busy)
│  ├─ run       the program itself                          (the only phase that holds GPUs)
│  └─ epilogue  optional, CPU; e.g. pack this task's results
└─ epilogue    once all tasks are finished                     (e.g. summarize / upload)
```

- **Project setup** runs once per scheduler start, before any of the project's tasks: `uv sync`, then
  the project's `setup` command (skipped when its `skip_if` check passes).
- **Tasks** usually only need `run`. Work that needs its own GPU slot or waits on something (an eval
  after a training run, a data build before training) is a separate task with `needs:`, not a phase.
- **Items** come from a plain list, `{glob: "configs/*.yaml"}`, `{range: [0, 10]}`, `{file: items.txt}`
  (one per line; JSON object lines become dict items) or `{command: "python list_items.py"}`.
  Items are re-listed every `rescan_seconds`, so new files are picked up.
- **Checks** (`done_if`, `skip_if`) are shell commands run in the project's workdir; exit code 0 means
  "yes". They run in a plain shell, not inside the project's venv, so keep them cheap (`test -f ...`).

## Config files: includes and generators

The main file holds the scheduler settings and lists where projects come from. Each project file can
live in its project's own repo:

```yaml
# fdgx.yaml
scheduler: { ... }
include:
  - ~/dsa-ml/fdgx.yaml                                   # YAML with a `projects:` mapping
  - ~/dsa-ml/fdgx_pack.py                                # generator (see below)
  - ~/queue/*.yaml                                       # globs: sorted; .* and *~ ignored; may match nothing
  - { command: "python make_pack.py --big", dir: ~/other }
  - { command: ".venv/bin/python hpc/fdgx_pack.py", dir: ~/dsa-ml, env_manager: shell }
projects: { ... }                                        # inline projects are fine too
```

- Included files may only contain `projects:`; scheduler settings belong in the main file.
- Relative paths are resolved against the file that contains them, and a project's `workdir`
  defaults to that file's directory.
- Project names must be unique across all files.

**Generators** produce the config in code: cross products, per-architecture settings, dependencies.
A `.py` include is *run* (not imported) inside the uv project that contains it
(`uv run --no-sync python file.py`), so it can import your project's code and dependencies. It prints the
same structure as a project file, as JSON or YAML, on stdout. `{command, dir}` runs any command the same
way: in `dir`'s uv project, or a plain shell when there is none or with `env_manager: shell` (for a venv uv
doesn't manage, call its interpreter directly, as in the example above). Generators have `check_timeout`
to finish. See `examples/proj_dummy/fdgx_pack.py`, which produces exactly the tasks of
`examples/proj_dummy/fdgx.yaml`.

**Loading config never installs anything.** Generators always run with `--no-sync` (which also means the
lock file isn't updated), at the first load as at every reload, so a hand-tuned venv (a driver-matched torch,
a numpy pinned for numba) is never touched. If the venv doesn't exist yet, the generator just fails and the
error is logged. Likewise, a project whose venv uv must not manage should use `env_manager: shell`: then
FDGX never runs `uv sync` for it.

**Errors are isolated per source.** A broken include, a failing generator or a name clash only affects
that source: it keeps its previous version (or is skipped at first load) and the error is logged; the
other sources load normally. Run `fdgx list -v fdgx.yaml` to check a config before using it.

## Dependencies

```yaml
- name: "eval-ood-{item.stem}"
  items: { glob: "configs/*.yaml" }
  needs: ["train-{item.stem}"]          # also "other_project/name" and globs like "build-*" or "data/*"
  run: "python eval.py --config {item} --suite ood"
```

- `needs` entries are rendered like `name` (with `{item}` and vars). A bare name means the same project.
- A task's first phase starts only when every task it needs is `done` (a passing `done_if` counts).
- Until then it is **waiting**. If something it needs failed, is blocked, belongs to a project whose
  setup failed, doesn't exist, or the needs form a cycle, it is **blocked**. A glob that matches nothing
  yet means waiting, so tasks generated later can still satisfy it.
- These conditions are derived, not stored: `fdgx retry` of the failed dependency unblocks its
  dependents automatically. Only the first phase is gated; a task that already started isn't stopped.
- The heartbeat and `fdgx status` show `waiting` and `blocked` counts, `fdgx status` the reasons, and
  `fdgx list` reports unknown names and cycles.

## Project epilogue

```yaml
projects:
  dsa:
    epilogue:
      run: "python -m hpc.summarize"
      when: all_finished        # default: scheduler.project_epilogue_when (all_done)
```

Runs once when every task of the project is finished: `all_done` = all `done`; `all_finished` = each is
`done`, `failed` or `blocked`. It runs again when tasks were added later (hot reload, new items) and have
finished too. Failures are retried with the project's `retry`; after that it is reported as failed (exit
code 2) until the task set changes, or until `fdgx retry` with no pattern or a `proj/*` pattern re-queues
it, e.g. after you fixed the script. Its state is in `state_dir/projects/<name>.json`.

## Hot reload

Every `rescan_seconds` FDGX re-reads the main file, the includes and the generators (without `uv sync`,
so a venv that jobs are using is never modified), and re-lists items:

- new projects and tasks are picked up;
- changed commands, vars, resources, priorities, `needs`, retries and epilogues apply to the **next**
  phase that starts; running processes are never touched;
- a project or task that disappears from the config starts nothing new, but a phase that is running
  finishes. Its state files stay on disk, so putting it back resumes it;
- changes to a project's `workdir`, `env_manager`, `shell_prefix`, `uv` or `setup`, and to the scheduler's
  `state_dir`, `heartbeat_seconds`, `shell`, `uv`, `nvidia_smi` and `keepalive_*`, apply at the next start;
- an invalid edit is logged once and the previous version keeps running.

Because stopping FDGX kills its jobs (see [Stopping](#stopping)), this is how a pack changes mid-run:
edit the project file (or re-run your generator script), then `fdgx reload` to apply it now. Running
training is untouched; anything that has to restart resumes from its checkpoints.

## Live control

On a machine that dies when the console goes quiet, you don't want to stop the scheduler to change
something. Everything below works **while it runs**, from another terminal:

```bash
uv run fdgx hold    fdgx.yaml 'train-arch3*' [--kill]   # park tasks: no new phase starts
uv run fdgx release fdgx.yaml 'train-arch3*'            # un-park them
uv run fdgx retry   fdgx.yaml ['proj/*'] [--reset-oom]  # re-queue failed tasks (and failed project epilogues) now
uv run fdgx reset   fdgx.yaml 'proj/eval-*'             # forget state; start over (done_if checked again)
uv run fdgx reload  fdgx.yaml                           # reload the config now instead of at the next rescan
uv run fdgx stop    fdgx.yaml                           # stop the scheduler and kill all its jobs
```

- **Patterns:** `train-*` matches task names in any project; `proj/train-*` matches full ids.
- **hold** lets a running phase finish; `--kill` kills it instead. A kill is an interruption, not a
  failure: no retry is used, and the phase resumes (from your checkpoint) on **release**. Tasks that
  need a held task wait, and so does the project epilogue.
- **reset** refuses tasks that are running: `hold --kill` them first.
- Held tasks and `enabled: false` (on a task template or a whole project) are "parked": they don't keep
  the scheduler alive and don't make it exit with an error. A disabled project runs no setup; tasks that
  depend on it wait.
- How it works: the CLI drops a request file into `state_dir/control/requests/`; the scheduler handles
  it at its next loop (about `poll_seconds`), replies, and the CLI prints the reply. While a scheduler
  runs it is the only process that writes task state. Without a running scheduler, `hold`, `release`,
  `retry` and `reset` edit the state files directly. Requests left over from a scheduler that died are
  discarded at the next start, so an old `stop` can't stop a new run.

## Stopping

`fdgx stop`, Ctrl-C, SIGTERM or SIGHUP stop the scheduler **and kill every job** it started (SIGTERM, then SIGKILL
after `kill_grace`), so nothing stale keeps holding GPU memory or file locks. Interrupted tasks resume at
the same phase on the next start. If the scheduler itself dies without cleaning up (`kill -9`, a crash),
the next start kills any processes it left behind and relaunches them from their checkpoints.

If a task was last running in **another container** (the hostname changed), FDGX can't see or kill that
copy: it logs that a ghost may still be running there and writing the task's outputs, plus what the
[GPU process log](#the-machines-gpu-process-log) says about your containers.

## How it stays correct across restarts

Each task has a small JSON state file in `state_dir/tasks/<project>/`, written atomically
(write to a temp file, fsync, rename). It records the *next phase to run*, retry and OOM counters, and a
short history.

On start, FDGX:

1. takes a lock on `state_dir/fdgx.lock`, so two schedulers can't share one state directory;
2. finds tasks marked as running (their scheduler died), kills any of their processes still alive
   (Linux) and resumes them **at the same phase**; an interruption is not counted as a failure;
3. evaluates `done_if` for new items and once per session before a task's next phase, and `skip_if`
   before each phase.

Your programs must therefore be safe to re-run from the start of any phase: resume from checkpoints,
and make epilogues idempotent.

> If a task has its own `epilogue`, **write `done_if` against the epilogue's output**, not the run's.
> If it checks only the training results, a task whose epilogue was interrupted will be marked done and
> the epilogue skipped. Use `run.skip_if` for "the run already produced its results".

## GPU selection

Every few seconds FDGX reads `nvidia-smi`. For each allowed GPU:

```
usable = free (nvidia-smi) − vram_safety_margin_gb − VRAM our running jobs declared but may not use yet
```

What a running job really uses is known when its processes show up in `nvidia-smi`, or when it reports
it through `$FDGX_VRAM_FILE`. Inside a container with its own PID namespace neither is true by default,
so FDGX reserves the job's whole `vram_gb` during `launch_warmup`, and after that whatever part of it the
GPU's used memory can't already contain. The reservation lasts **for the whole run** (training peaks on its
longest batch, hours in), and memory the job already holds is not counted twice.

Reporting is two lines in a PyTorch job, e.g. once per epoch (bytes, peak per GPU):

```python
if path := os.environ.get("FDGX_VRAM_FILE"):
    Path(path).write_text(str(torch.cuda.max_memory_reserved()))
```

- Tasks are ordered by `priority` (higher first), then config order, then item order.
- Each task gets as many fitting GPUs as are free, between `gpus.min` and `gpus.max`, preferring the GPUs
  with the most free memory.
- A task that has waited longer than `starvation` stops smaller tasks from being scheduled past it,
  so GPUs drain until it fits.
- A task that can never fit on this machine fails immediately with a clear message.
- With `max_gpu_util`, GPUs that someone else is using heavily are avoided.
- With `max_jobs_per_gpu` (a number, or `{gpu index: number}`), small jobs can't crowd a GPU that
  training runs share, even when VRAM is free.

If another user grabs the memory between the check and the allocation, the job simply hits an OOM and
goes through the normal OOM path below.

## CPU and RAM

Every phase of a task uses `resources.cpus` (default 1) and, if set, `resources.ram_gb` from the
scheduler's budget: `scheduler.cpus` (default: what the container may use, from its cgroup quota or CPU
affinity) and `scheduler.ram_gb` (default: not enforced). A phase mapping can override the CPU count for
that phase alone, e.g. `setup: {run: ..., cpus: 8}`. Project setup (including its `uv sync`) and the project
epilogue use 1 CPU unless their mapping sets `cpus` — give a setup that runs a test suite more, e.g.
`setup: {run: "pytest -q", cpus: 8}`. A phase waits until the budget has room; a request larger than the
whole budget still runs when nothing else does.

Every process FDGX starts gets `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and
`NUMBA_NUM_THREADS` set to its CPU count, unless your project/task `env` sets them, so torch, numba and BLAS
don't size their pools to every core of a many-core machine. That covers task phases, project setup and
epilogue, and also `done_if` / `skip_if` checks, `items.command` and config generators (1 thread each).

Tasks with `gpus: 0` (data builds, CPU evals) run their `run` phase without a GPU and without waiting for
`nvidia-smi`; they count against the CPU budget, not `max_running` or `max_cpu_phases` (which only limits
setup / epilogue / project phases).

## Failures, retries and OOM

| What happened | What FDGX does |
|---|---|
| Phase exits non-zero | Retry after `retry.backoff[n]`, up to `retry.max_attempts` per phase. Then `failed`. A phase that ran at least `retry.reset_after` (default 1h) before failing starts with a fresh budget, so a 4-day run can die several times from NCCL / Xid errors, while fast crash loops still stop quickly. |
| Killed by a SIGKILL FDGX didn't send | Most likely the host's OOM killer: logged as `host_oom_suspected`, retried as a normal failure (it doesn't move up `on_oom`; set `ram_gb` instead). |
| Output matches an OOM pattern during `run` | Apply the next `on_oom` step and retry soon (`oom_retry_delay`), **without** using a retry. Once the ladder is exhausted, OOMs count as normal failures. |
| No output for `stall_timeout` | Kill the process tree (SIGTERM, then SIGKILL after `kill_grace`) and treat as a failure. |
| Runs longer than `timeout` | Same as above. |
| Scheduler / instance dies | Resume at the same phase on next start; not a failure. |

Work that can't resume (e.g. a data build) lists its partial outputs in `clean:`; they are deleted
before **every** attempt of that phase, so it always starts from empty output. Paths are templated and
may be globs; anything resolving outside the workdir is refused.

```yaml
run:
  run: "python -m hpc.build --arch {arch}"
  clean: ["data/{arch}.tmp", "data/{arch}/*.partial"]
```

`on_oom` steps are applied cumulatively and can change `vars`, `env` and `resources`:

```yaml
on_oom:
  - { bs: 16 }                                         # shorthand for { vars: { bs: 16 } }
  - { vars: { bs: 8 }, resources: { vram_gb: 70, gpus: 2 } }
  - { env: { PYTORCH_CUDA_ALLOC_CONF: "expandable_segments:True" } }
```

OOM detection looks for `CUDA out of memory`, `OutOfMemoryError`, `CUBLAS_STATUS_ALLOC_FAILED`,
`out of memory` and similar. Add your own with `scheduler.extra_oom_patterns`.

## Staying alive

A dedicated thread prints a status line every `heartbeat_seconds`, whatever the main loop is doing,
including while it is only waiting for GPUs:

```
[2026-10-15 11:22:15] fdgx alive | jobs 3 gpu + 0 cpu | queued 0 waiting 6 blocked 0 backoff 0 done 0 failed 0 | cpus 3/16 | free GB 0:41/80 1:30/80
    dummy/train-a                            run      gpus 0,1          11s | step 18/20 loss 0.0556
```

Events (launches, completions, retries, blocked tasks, config reloads) are printed as they happen and also appended to
`state_dir/fdgx.log`. Each phase's full output goes to `<log_dir>/<task>/<time>-<phase>.log`, where
`log_dir` is set per project (relative to its workdir, e.g. `hpc/logs`, so job output lives next to the
project's results and can be bundled with them) and defaults to `state_dir/logs/<project>`. Project setup
and epilogue logs go to `<log_dir>/_project/`. Scheduler-wide files (`fdgx.log`, task states,
`status.json`) always stay in `state_dir`.

If the instance judges "idle" by more than console output, set `keepalive_cmd` (run every
`keepalive_seconds`).

Because the heartbeat has its own thread, it would keep printing even if the main loop got stuck (for
example on a hung filesystem). So the status line ends with `| LOOP STALLED 7m` once the main loop is
late by more than a minute (or 10 × `poll_seconds`).

## The machine's GPU process log

Some machines publish a log of every GPU process with its memory and container, e.g.

```
PID=3908736, Memoria=3908736, 41862 MiB, Nome=/jupyter-arthur.leonida.1.icnza6nynxukzj4svoofoh7to
```

Point FDGX at it with `scheduler.gpu_process_log: {path: ..., mine: "jupyter-<you>."}` (`mine` defaults
to `$JUPYTERHUB_USER`). It is written by someone else and may list ghosts, so FDGX uses it **only for
diagnostics**, never for allocation (nvidia-smi's free memory already accounts for everything):

- `fdgx gpus fdgx.yaml` shows usage per container, largest first, with yours marked;
- the scheduler warns (once per situation) when GPU memory under your name shows up while it runs no GPU
  job, or under more than one of your containers (one is probably a ghost from a previous container);
- the "ghost from another container" warning on resume includes what the log says.

The log is ignored when missing, unreadable or older than `max_age` (default 10 min).

## Configuration

`examples/fdgx.yaml` (scheduler) and `examples/proj_dummy/fdgx.yaml` (project) are the annotated
reference. Relative paths are resolved against the directory of the file that contains them. Durations accept `90`, `"30s"`, `"15m"`, `"2h"`, `"1h30m"`.

### `scheduler`

| Key | Default | Meaning |
|---|---|---|
| `gpus` | `all` | GPU indices FDGX may use, e.g. `[0, 1, 2, 3]` |
| `vram_safety_margin_gb` | `2` | Never plan to use the last N GB of a GPU |
| `heartbeat_seconds` | `30` | Status line interval |
| `max_cpu_phases` | `2` | Concurrent setup / epilogue / project-setup processes |
| `max_prepared` | `2` | Tasks allowed to finish setup while still waiting for a GPU |
| `max_running` | — | Cap on concurrent GPU jobs |
| `max_jobs_per_gpu` | — | Cap on our jobs per GPU: a number, or `{gpu index: number}` |
| `cpus` | container's CPUs | CPU budget shared by all phases (see [CPU and RAM](#cpu-and-ram)) |
| `ram_gb` | — | RAM budget, for tasks that declare `resources.ram_gb` |
| `max_gpu_util` | — | Skip GPUs (without our jobs) busier than this % |
| `launch_warmup` | `5m` | How long a new job's VRAM stays reserved when its usage can't be observed |
| `starvation` | `1h` | Wait time after which a task blocks smaller tasks from jumping ahead (`0` = off) |
| `rescan_seconds` | `300` | Reload the config, re-list items and reload task states edited by the CLI |
| `project_epilogue_when` | `all_done` | Default trigger for project epilogues (`all_done` or `all_finished`) |
| `oom_retry_delay` | `10` | Delay before retrying after an OOM |
| `kill_grace` | `20` | Seconds between SIGTERM and SIGKILL |
| `check_timeout` | `120` | Timeout for `done_if` / `skip_if` / `items.command` |
| `poll_seconds`, `gpu_poll_seconds` | `2`, `5` | Main loop and nvidia-smi intervals |
| `state_dir` | `.fdgx` | Task state and logs; must be on the persistent disk |
| `shell` | `/bin/bash -c` | Shell used for commands and checks |
| `uv`, `nvidia_smi` | `uv`, `nvidia-smi` | Executables |
| `keepalive_cmd`, `keepalive_seconds` | —, `60` | Optional extra activity |
| `gpu_process_log` | — | `{path, mine, max_age}`: the machine's GPU process log, for diagnostics |
| `extra_oom_patterns` | `[]` | Additional regexes that mean "out of memory" |

### `projects.<name>`

| Key | Meaning |
|---|---|
| `workdir` | Project directory; all commands run here. Default: the directory of the file defining the project |
| `env_manager` | `uv` (default): `uv sync` at start, commands run via `uv run --no-sync`. `shell`: plain shell, with optional `shell_prefix` (e.g. `source ~/miniconda3/bin/activate env &&`) |
| `uv` | `{ python, extras, all_extras, groups, frozen }`, passed to `uv sync`. `--frozen` is used automatically when `uv.lock` exists |
| `env`, `vars` | Environment variables / template variables for all tasks of the project |
| `setup` | One-time project setup: a command, or `{ run, skip_if, timeout, stall_timeout, cpus }` (`cpus` default 1) |
| `epilogue` | Runs when all tasks are finished: a command, or `{ run, skip_if, timeout, stall_timeout, when, cpus }` |
| `retry` | Retry policy for the project setup and epilogue |
| `tasks` | List of task templates (below) |
| `enabled` | `false` parks the whole project: no setup, no tasks (default `true`) |
| `log_dir` | Where its job logs go, relative to `workdir`, templated (`{project}`, vars). Default: `state_dir/logs/<project>` |

### Task templates

| Key | Meaning |
|---|---|
| `name` | Task name template, e.g. `"train-{item.stem}"` (`-{item}` is appended if there's no placeholder) |
| `items` | Item source (see [Concepts](#concepts)); omit for a single task |
| `done_if` | Check meaning "this item is completely finished" |
| `needs` | Tasks that must be `done` first: names, `project/name` or globs (templated) |
| `priority` | Higher runs first (default `0`) |
| `enabled` | `false` parks the template's tasks: discovered and shown, never started (default `true`) |
| `vars`, `env` | Template variables / environment variables |
| `resources` | `{ vram_gb: <per GPU>, gpus: N or { min, max }, cpus: 1, ram_gb: 0 }`; `gpus: 0` runs without a GPU |
| `run` | The program: a command, or `{ run, skip_if, timeout, stall_timeout, clean, cpus }` (required; `cpus` overrides `resources.cpus` for this phase) |
| `setup`, `epilogue` | Optional per-task phases, same forms (rarely needed: prefer project setup/epilogue and `needs`) |
| `timeout`, `stall_timeout` | Defaults for all phases of the task |
| `retry` | `{ max_attempts: 3, backoff: [30s, 2m, 10m], reset_after: 1h }` (defaults shown) |
| `on_oom` | List of override steps applied on successive OOMs |

### Templates

Commands, names, env values and checks are Python format strings:

| Variable | Value |
|---|---|
| `{item}` | The item (`{item.stem}`, `{item.name}`, `{item.parent}`, `{item.suffix}`, `{item.index}`; `{item.key}` or `{item[key]}` for dict items) |
| `{gpus}`, `{num_gpus}` | e.g. `"0,3"` and `2` (empty / `0` outside `run`) |
| `{attempt}`, `{oom_level}`, `{phase}` | Retry number, OOM steps applied, current phase |
| `{task}`, `{project}`, `{workdir}` | Names and paths |
| any `vars` key | After `on_oom` overrides |

Write `{{` and `}}` for literal braces (e.g. `awk '{{print $1}}'`).

Every phase also gets these environment variables: `CUDA_VISIBLE_DEVICES` (empty outside `run`),
`CUDA_DEVICE_ORDER=PCI_BUS_ID` (so indices match `nvidia-smi`), `PYTHONUNBUFFERED=1`, the thread-count
variables (see [CPU and RAM](#cpu-and-ram)), `FDGX_TASK`, `FDGX_PHASE`, `FDGX_ATTEMPT`, `FDGX_OOM_LEVEL`,
`FDGX_GPUS`, `FDGX_NUM_GPUS`, `FDGX_LOG_FILE` (this phase's own log, e.g. for an epilogue to copy) and
`FDGX_STATE_DIR` (FDGX's state directory), and in GPU run phases `FDGX_VRAM_FILE`. Multi-GPU example:

```yaml
run: "torchrun --nproc_per_node {num_gpus} train.py --config {item} --bs {bs}"
```

## Command line

```bash
uv run fdgx run fdgx.yaml [--stay-alive]     # the scheduler (bootstrap.sh calls this)
uv run fdgx run fdgx.yaml --cpu              # dry run without GPUs: all tasks CPU-only, CUDA_VISIBLE_DEVICES=-1
uv run fdgx status fdgx.yaml [-v]            # task states, blocked tasks with reasons (-v: also waiting)
uv run fdgx list fdgx.yaml [-v]              # load + expand everything, check needs; -v: needs and run commands
uv run fdgx retry fdgx.yaml ['proj/*'] [--reset-oom]   # re-queue failed tasks from the phase that failed
uv run fdgx reset fdgx.yaml 'proj/train-a'   # forget a task's state; it starts over from setup
uv run fdgx hold fdgx.yaml 'train-*' [--kill]   # park tasks (see Live control)
uv run fdgx release fdgx.yaml 'train-*'      # un-park them
uv run fdgx reload fdgx.yaml                 # make the running scheduler reload its config now
uv run fdgx stop fdgx.yaml                   # stop the running scheduler and kill its jobs
uv run fdgx gpus [fdgx.yaml]                 # what FDGX sees on the GPUs (+ the GPU process log, if set)
```

With a scheduler running, `retry`, `reset`, `hold`, `release`, `reload` and `stop` go through it (see
[Live control](#live-control)) and apply within a couple of seconds; without one, the first four edit
the state files directly. `status` reads the conditions a running scheduler writes to
`state_dir/status.json`.

## Repository layout

```
bootstrap.sh          install uv if missing -> uv sync -> fdgx run
fdgx/
  config.py           YAML parsing and validation, item and task expansion, templates
  loader.py           main file + includes + generators, per-source error isolation
  deps.py             `needs` resolution, cycles, waiting/blocked conditions
  scheduler.py        main loop: project setup/epilogue, discovery, deps, allocation, phases, retries, reload
  allocator.py        pure GPU allocation logic
  gpu.py              nvidia-smi probe (and FDGX_FAKE_GPUS for testing)
  runner.py           child process trees: logs, OOM detection, stall/timeout watchdog, kills, clean, CPU count
  envs.py             uv / shell environment managers
  state.py            atomic per-task JSON state, lock file
  control.py          live control: request/reply files, offline edits of state
  gpulog.py           reader for the machine's GPU process log (diagnostics only)
  heartbeat.py        status-line thread and keepalive
  cli.py              command line
examples/
  fdgx.yaml           annotated main config (scheduler settings + include)
  proj_dummy/         fake uv project used by the example and tests:
    fdgx.yaml         annotated project file: trainings, evals with `needs`, project setup and epilogue
    fdgx_pack.py      the same project generated in Python
    job.py            prepare / checkpointed train / pack / eval / summary
tests/                unit tests and end-to-end tests with fake GPUs
```

## Development

```bash
uv sync
uv run pytest
```

The end-to-end tests use `FDGX_FAKE_GPUS` (`"index:total_gb[:used_gb],..."`, or `"@file"` to read the
spec from a file on every probe) and run `examples/proj_dummy/job.py` through a real shell, covering
multi-GPU allocation, retries, the OOM ladder, the stall watchdog, `done_if`, the heartbeat, resuming
after the scheduler is killed, dependencies (waiting, blocked, unblocked by `retry`), project epilogues,
hot reload (added tasks, removed includes, invalid edits), and live control (hold / hold --kill /
release, immediate retry and reload, stop, `enabled: false`, offline edits), and resources (CPU budget,
jobs per GPU, whole-run VRAM reservation, `reset_after`, `clean`, host-OOM detection, `--cpu`). On Windows, commands run through Git Bash when it is installed.
