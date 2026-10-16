import json
import sys
from pathlib import Path

from fdgx import cli
from fdgx.loader import load

PY = Path(sys.executable).as_posix()
EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def project(name: str, run: str = "echo hi") -> str:
    return f"projects:\n  {name}:\n    env_manager: shell\n    tasks:\n      - {{name: t, run: '{run}'}}\n"


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def main_cfg(tmp_path: Path, includes: list) -> Path:
    return write(tmp_path / "fdgx.yaml", json.dumps({"scheduler": {"state_dir": "state"}, "include": includes}))


def test_includes_globs_and_relative_workdir(tmp_path):
    write(tmp_path / "a" / "fdgx.yaml", project("a"))
    write(tmp_path / "queue" / "b.yaml", project("b"))
    write(tmp_path / "queue" / "c.yaml", project("c"))
    write(tmp_path / "queue" / ".hidden.yaml", project("hidden"))
    res = load(main_cfg(tmp_path, ["a/fdgx.yaml", "queue/*.yaml", "empty/*.yaml"]))
    assert res.errors == []
    assert any("matches no files" in w for w in res.warnings)
    by_name = {p.name: p for p in res.config.projects}
    assert sorted(by_name) == ["a", "b", "c"]
    assert by_name["a"].workdir == (tmp_path / "a").resolve()
    assert by_name["b"].workdir == (tmp_path / "queue").resolve()
    assert by_name["a"].source == str((tmp_path / "a" / "fdgx.yaml").resolve())


def test_broken_include_keeps_previous_version(tmp_path):
    a = write(tmp_path / "a.yaml", project("a", "echo one"))
    write(tmp_path / "b.yaml", project("b"))
    cfg = main_cfg(tmp_path, ["a.yaml", "b.yaml"])
    first = load(cfg).config
    write(tmp_path / "b.yaml", project("b2"))
    a.write_text("projects: [this is: not valid")
    res = load(cfg, previous=first)
    assert len(res.errors) == 1 and "a.yaml" in res.errors[0]
    by_name = {p.name: p for p in res.config.projects}
    assert sorted(by_name) == ["a", "b2"]
    assert by_name["a"].tasks[0].run.run == "echo one"


def test_scheduler_key_in_include_and_missing_file(tmp_path):
    write(tmp_path / "bad.yaml", "scheduler: {}\n" + project("x"))
    write(tmp_path / "ok.yaml", project("ok"))
    res = load(main_cfg(tmp_path, ["bad.yaml", "missing.yaml", "ok.yaml"]))
    assert [p.name for p in res.config.projects] == ["ok"]
    assert any("main config file only" in e for e in res.errors)
    assert any("does not exist" in e for e in res.errors)


def test_invalid_main_keeps_previous(tmp_path):
    write(tmp_path / "a.yaml", project("a"))
    cfg = main_cfg(tmp_path, ["a.yaml"])
    first = load(cfg).config
    cfg.write_text("scheduler: {gpus: nope}\n")
    res = load(cfg, previous=first)
    assert res.config is first and res.errors


def test_name_clash(tmp_path):
    write(tmp_path / "one.yaml", project("same", "echo one"))
    write(tmp_path / "two.yaml", project("same", "echo two"))
    res = load(main_cfg(tmp_path, ["one.yaml", "two.yaml"]))
    assert [p.tasks[0].run.run for p in res.config.projects] == ["echo one"]
    assert any("already defined by" in e for e in res.errors)


def test_command_generator_and_failure(tmp_path):
    gen = write(tmp_path / "gen" / "gen.py", (
        "import json, sys\n"
        "if len(sys.argv) > 1: sys.exit(3)\n"
        "print(json.dumps({'projects': {'g': {'env_manager': 'shell', 'tasks': ["
        "{'name': f't{i}', 'run': 'echo', 'needs': [f't{i - 1}'] if i else []} for i in range(3)]}}}))\n"
    ))
    cfg = main_cfg(tmp_path, [{"command": f'"{PY}" gen.py', "dir": "gen"}])
    res = load(cfg)
    assert res.errors == []
    (g,) = res.config.projects
    assert g.workdir == gen.parent.resolve()
    assert [t.needs for t in g.tasks] == [[], ["t0"], ["t1"]]
    # same command now failing -> keeps its previous projects
    gen.write_text(gen.read_text().replace("len(sys.argv) > 1", "True"))
    res2 = load(cfg, previous=res.config)
    assert res2.errors and "exited 3" in res2.errors[0]
    assert [p.name for p in res2.config.projects] == ["g"]


def test_python_pack_equals_yaml(tmp_path, capsys):
    """The doc's M2 acceptance test: fdgx_pack.py and fdgx.yaml expand to the same task list."""
    outputs = []
    for name, inc in (("yaml", "proj_dummy/fdgx.yaml"), ("py", "proj_dummy/fdgx_pack.py")):
        cfg = write(tmp_path / name / "fdgx.yaml", json.dumps({
            "scheduler": {"state_dir": str(tmp_path / name / "state")},
            "include": [str(EXAMPLES / inc)],
        }))
        assert cli.main(["list", "-v", str(cfg)]) == 0
        outputs.append(sorted(capsys.readouterr().out.splitlines()))
    assert outputs[0] == outputs[1]
    assert len(outputs[0]) == 1 + 9  # project header + 3 trainings + 6 evals


def test_list_reports_unknown_deps_and_cycles(tmp_path, capsys):
    write(tmp_path / "p.yaml", (
        "projects:\n  p:\n    env_manager: shell\n    tasks:\n"
        "      - {name: a, run: x, needs: [b]}\n"
        "      - {name: b, run: x, needs: [a]}\n"
        "      - {name: c, run: x, needs: [nope]}\n"
    ))
    assert cli.main(["list", str(main_cfg(tmp_path, ["p.yaml"]))]) == 1
    err = capsys.readouterr().err
    assert "dependency cycle: p/a -> p/b -> p/a" in err and "unknown dependency nope" in err


def test_generators_never_sync(tmp_path):
    from fdgx.config import ConfigError, SchedulerCfg
    from fdgx.loader import generator_argv

    sc = SchedulerCfg(state_dir=tmp_path / "state", shell=["bash", "-c"])
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'x'\nversion = '0'\n")
    argv = generator_argv("python pack.py", tmp_path, sc)
    assert argv[:3] == ["uv", "run", "--no-sync"] and argv[-3:] == ["bash", "-c", "python pack.py"]
    assert generator_argv("python pack.py", tmp_path, sc, "shell") == ["bash", "-c", "python pack.py"]
    bare = tmp_path / "no_project"
    bare.mkdir()
    (tmp_path / "pyproject.toml").unlink()
    assert generator_argv("x", bare, sc) == ["bash", "-c", "x"]
    try:
        generator_argv("x", bare, sc, "uv")
        raise AssertionError("expected ConfigError")
    except ConfigError:
        pass


def test_include_env_manager_validated(tmp_path):
    res = load(main_cfg(tmp_path, [{"command": "x", "env_manager": "conda"}]))
    assert any("must be 'uv' or 'shell'" in e for e in res.errors)
