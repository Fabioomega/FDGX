"""The dummy project of fdgx.yaml, generated in Python.

fdgx runs this file inside the project's uv environment (so project code and dependencies are
importable) and reads the config from stdout as JSON. It produces the same tasks as fdgx.yaml.
"""

import json
from pathlib import Path

SUITES = ["ood", "id"]
configs = sorted(p.as_posix() for p in Path("configs").glob("*.yaml"))  # cwd: this file's directory

tasks = []
for cfg in configs:
    stem = Path(cfg).stem
    tasks.append({
        "name": f"train-{stem}",
        "done_if": f"test -f results/{stem}.json",
        "vars": {"bs": 32, "steps": 20},  # kept as vars so on_oom can override them
        "resources": {"vram_gb": 20, "gpus": {"min": 1, "max": 2}, "cpus": 2},
        "setup": {"skip_if": f"test -f work/{stem}.prep", "run": f"python job.py prepare --item {cfg}"},
        "run": {"run": f"python job.py train --item {cfg} --bs {{bs}} --steps {{steps}} --oom-above 16",
                "stall_timeout": "30m"},
        "retry": {"max_attempts": 3, "backoff": ["30s", "2m", "10m"], "reset_after": "1h"},
        "on_oom": [{"bs": 16}, {"vars": {"bs": 8}, "resources": {"vram_gb": 30}}],
    })
for suite in SUITES:
    for cfg in configs:
        stem = Path(cfg).stem
        tasks.append({
            "name": f"eval-{suite}-{stem}",
            "needs": [f"train-{stem}"],
            "done_if": f"test -f results/{stem}.{suite}.json",
            "resources": {"vram_gb": 5},
            "run": f"python job.py eval --item {cfg} --suite {suite} --steps 5",
        })

print(json.dumps({
    "projects": {
        "dummy": {
            "env_manager": "uv",
            "env": {"HF_HOME": "{workdir}/.hf"},
            "setup": {"skip_if": "test -d work", "run": "mkdir -p work results"},
            "epilogue": {"run": "python job.py summary"},
            "retry": {"max_attempts": 5, "backoff": ["30s", "2m"]},
            "tasks": tasks,
        }
    }
}))
