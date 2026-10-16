from pathlib import Path

import pytest

from fdgx.allocator import Request, allocate
from fdgx.config import ConfigError, Item, expand_items, load_config, parse_duration, render
from fdgx.envs import make_env_manager
from fdgx.state import StateStore, TaskState, safe_name


def write_cfg(tmp_path: Path, text: str) -> Path:
    (tmp_path / "proj").mkdir(exist_ok=True)
    p = tmp_path / "jobs.yaml"
    p.write_text(text)
    return p


BASIC = """
projects:
  p:
    workdir: proj
    tasks:
      - name: t
        items: [a, b]
        vars: {bs: 32}
        resources: {vram_gb: 10, gpus: {min: 1, max: 4}}
        run: "python x.py --bs {bs}"
        on_oom:
          - {bs: 16}
          - {vars: {bs: 8}, env: {X: "1"}, resources: {vram_gb: 20, gpus: 2}}
"""


def test_parse_duration():
    assert parse_duration(90, "x") == 90
    assert parse_duration("30s", "x") == 30
    assert parse_duration("15m", "x") == 900
    assert parse_duration("1h30m", "x") == 5400
    with pytest.raises(ConfigError):
        parse_duration("soon", "x")


def test_load_and_oom_ladder(tmp_path):
    cfg = load_config(write_cfg(tmp_path, BASIC))
    p = cfg.projects[0]
    t = p.tasks[0]
    assert t.name == "t-{item}"  # items without a placeholder get one
    assert p.env_manager == "uv"
    vars_, env, res = p.effective(t, 0)
    assert (vars_["bs"], res.vram_gb, res.min_gpus, res.max_gpus) == (32, 10, 1, 4)
    vars_, env, res = p.effective(t, 1)
    assert vars_["bs"] == 16 and res.vram_gb == 10
    vars_, env, res = p.effective(t, 2)
    assert vars_["bs"] == 8 and env == {"X": "1"} and (res.vram_gb, res.min_gpus, res.max_gpus) == (20, 2, 2)


def test_unknown_key_is_an_error(tmp_path):
    with pytest.raises(ConfigError, match="unknown key"):
        load_config(write_cfg(tmp_path, BASIC.replace("vars: {bs: 32}", "varz: {bs: 32}")))


def test_items_and_render(tmp_path):
    (tmp_path / "cfgs").mkdir()
    for n in ("b", "a"):
        (tmp_path / "cfgs" / f"{n}.yaml").write_text("")
    items = expand_items({"glob": "cfgs/*.yaml"}, tmp_path)
    assert items == ["cfgs/a.yaml", "cfgs/b.yaml"]
    assert expand_items({"range": [2, 5]}, tmp_path) == [2, 3, 4]
    (tmp_path / "list.txt").write_text('x\n# comment\n\n{"model": "m1", "lr": 0.1}\n')
    assert expand_items({"file": "list.txt"}, tmp_path) == ["x", {"model": "m1", "lr": 0.1}]
    assert expand_items({"command": "whatever"}, tmp_path, lambda c: "p\nq\n") == ["p", "q"]

    it = Item("cfgs/a.yaml", 0)
    assert render("{item.stem}-{item.parent}-{bs}", {"item": it, "bs": 4}) == "a-cfgs-4"
    assert render("{item.model}/{item[lr]}", {"item": Item({"model": "m1", "lr": 0.1}, 1)}) == "m1/0.1"
    assert render("awk '{{print $1}}'", {}) == "awk '{print $1}'"
    with pytest.raises(ConfigError, match="unknown template variable"):
        render("{nope}", {})


def test_allocator():
    # most-free first, as many as allowed
    assert allocate([Request("a", 10, 1, 2)], {0: 20, 1: 50, 2: 30}) == [("a", [1, 2])]
    # min not satisfiable -> skipped, smaller task behind it still runs (backfill)
    assert allocate([Request("big", 40, 2, 2), Request("small", 10, 1, 1)], {0: 50, 1: 20}) == [("small", [0])]
    # starving big task blocks backfill
    assert allocate([Request("big", 40, 2, 2, starving=True), Request("small", 10, 1, 1)], {0: 50, 1: 20}) == []
    # capacity is consumed within one round
    assert allocate([Request("a", 30, 1, 1), Request("b", 30, 1, 1), Request("c", 30, 1, 1)], {0: 70}) == [
        ("a", [0]), ("b", [0])]
    # CPU-only run phase
    assert allocate([Request("cpu", 0, 0, 0)], {}) == [("cpu", [])]


def test_uv_wrapping(tmp_path):
    cfg = load_config(write_cfg(tmp_path, BASIC))
    p = cfg.projects[0]
    em = make_env_manager(p, cfg.scheduler)
    assert em.sync_argv() == ["uv", "sync", "--project", str(p.workdir)]
    (p.workdir / "uv.lock").write_text("")
    p.uv.extras = ["cuda"]
    assert em.sync_argv() == ["uv", "sync", "--project", str(p.workdir), "--frozen", "--extra", "cuda"]
    argv = em.wrap("python a.py && python b.py")
    assert argv[:5] == ["uv", "run", "--no-sync", "--project", str(p.workdir)]
    assert argv[5] == "--" and argv[-1] == "python a.py && python b.py"


def test_state_roundtrip(tmp_path):
    store = StateStore(tmp_path)
    st = TaskState(id="p/x y", phase="run", attempts=2)
    st.event("hello")
    store.save("p", "x y", st)
    back = store.load("p", "x y")
    assert back.phase == "run" and back.attempts == 2 and back.history[0]["msg"] == "hello"
    assert safe_name("a/b") != safe_name("a_b")
    store.path("p", "bad").parent.mkdir(parents=True, exist_ok=True)
    store.path("p", "bad").write_text("{not json")
    assert store.load("p", "bad") is None
