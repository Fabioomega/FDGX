import os
import time

from fdgx import control
from fdgx.state import StateStore, TaskState


def test_match_rule():
    assert control.match("p/train-a", ["train-*"])  # bare pattern: name in any project
    assert control.match("q/train-a", ["train-*"])
    assert control.match("p/train-a", ["p/*"])
    assert not control.match("q/train-a", ["p/*"])
    assert not control.match("p/eval-a", ["train-*"])
    assert control.match("p/anything", [])


def test_request_reply_round_trip(tmp_path):
    rid = control.submit(tmp_path, "hold", ["train-*"], kill=True)
    ((path, req),) = control.pending(tmp_path)
    assert (req["id"], req["op"], req["patterns"], req["kill"]) == (rid, "hold", ["train-*"], True)
    path.unlink()
    control.reply(tmp_path, rid, True, ["held p/train-a"])
    rep = control.wait_reply(tmp_path, rid, 1)
    assert rep["ok"] is True and rep["lines"] == ["held p/train-a"]
    assert control.wait_reply(tmp_path, rid, 0.2) is None  # consumed


def test_discard_stale(tmp_path):
    control.submit(tmp_path, "stop")
    rid = control.submit(tmp_path, "reload")
    control.reply(tmp_path, rid, True, [])
    old = tmp_path / "control" / "replies" / f"{rid}.json"
    os.utime(old, (time.time() - 2 * control.REPLY_TTL,) * 2)
    dropped = control.discard_stale(tmp_path)
    assert sorted(r["op"] for r in dropped) == ["reload", "stop"]
    assert control.pending(tmp_path) == [] and not old.exists()


def test_apply_offline(tmp_path):
    store = StateStore(tmp_path)
    store.save("p", "train-a", TaskState(id="p/train-a", phase="run"))
    store.save("p", "train-b", TaskState(id="p/train-b", phase="failed", failed_phase="run", attempts=3,
                                         oom_level=2))
    store.save("p", "eval-a", TaskState(id="p/eval-a", phase="done"))

    assert control.apply_offline(store, "hold", ["train-a", "eval-a"]) == ["held p/train-a"]
    assert store.load("p", "train-a").held
    assert control.apply_offline(store, "release", ["*"]) == ["released p/train-a"]
    assert not store.load("p", "train-a").held

    assert control.apply_offline(store, "retry", [], reset_oom=True) == ["retry p/train-b from run"]
    st = store.load("p", "train-b")
    assert (st.phase, st.attempts, st.oom_level) == ("run", 0, 0)

    control.apply_offline(store, "reset", ["p/eval-*"])
    assert store.load("p", "eval-a") is None
    assert control.apply_offline(store, "release", ["nope"]) == ["nothing to release matching nope"]


def test_matches_project():
    assert control.matches_project("p", [])
    assert control.matches_project("p", ["p/*"]) and control.matches_project("p", ["*"])
    assert not control.matches_project("p", ["p/train-a"]) and not control.matches_project("p", ["train-*"])
    assert not control.matches_project("p", ["q/*"])
