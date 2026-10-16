from fdgx.deps import BLOCKED, OK, WAITING, cycles, evaluate, resolve

READY = {"p": "ready", "q": "ready"}


def test_resolve_forms():
    known = {"p/train-a", "p/train-b", "q/data", "p/eval-a"}
    r = resolve({"p/eval-a": ["train-a", "q/data", "train-*", "nope", "missing-*"]}, known, READY)
    assert r.edges["p/eval-a"] == {"p/train-a", "p/train-b", "q/data"}
    cond, reason = r.issues["p/eval-a"]
    assert cond == BLOCKED and "unknown dependency nope" in reason


def test_pending_and_failed_projects():
    r = resolve({"p/x": ["q/data"]}, {"p/x"}, {"p": "ready", "q": "pending"})
    assert r.issues["p/x"][0] == WAITING
    r = resolve({"p/x": ["q/data"]}, {"p/x"}, {"p": "ready", "q": "failed"})
    assert r.issues["p/x"][0] == BLOCKED and "failed setup" in r.issues["p/x"][1]
    r = resolve({"p/x": ["later-*"]}, {"p/x"}, READY)  # glob matching nothing yet: wait
    assert r.issues["p/x"][0] == WAITING


def test_cycles():
    assert cycles({"a": {"b"}, "b": {"c"}, "c": {"a"}, "d": {"a"}}) == [["a", "b", "c"]]
    assert cycles({"a": {"a"}}) == [["a"]]
    assert cycles({"a": {"b"}, "b": set()}) == []


def test_evaluate_conditions():
    needs = {"p/eval": ["train"], "p/report": ["eval"], "p/loop1": ["loop2"], "p/loop2": ["loop1"]}
    phases = {"p/train": "run", "p/eval": "setup", "p/report": "setup", "p/loop1": "setup", "p/loop2": "setup"}
    c = evaluate(needs, phases, READY)
    assert c["p/eval"] == (WAITING, "waiting on p/train")
    assert c["p/report"][0] == WAITING
    assert c["p/loop1"][0] == BLOCKED and "cycle" in c["p/loop1"][1]

    phases["p/train"] = "done"
    c = evaluate(needs, phases, READY)
    assert c["p/eval"] == (OK, "")

    phases["p/train"] = "failed"
    c = evaluate(needs, phases, READY)
    assert c["p/eval"] == (BLOCKED, "dependency p/train failed")
    assert c["p/report"] == (BLOCKED, "dependency p/eval is blocked")  # transitive


def test_done_dependency_is_enough_even_if_its_own_deps_are_odd():
    c = evaluate({"p/b": ["a"], "p/a": ["gone"]}, {"p/a": "done", "p/b": "setup"}, READY)
    assert c["p/b"] == (OK, "")
