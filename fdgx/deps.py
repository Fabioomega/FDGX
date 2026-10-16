"""Task dependencies (`needs:`): resolution, cycle detection and the derived waiting/blocked condition."""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field

OK, WAITING, BLOCKED = "ok", "waiting", "blocked"


def is_glob(s: str) -> bool:
    return any(c in s for c in "*?[")


def qualify(entry: str, project: str) -> str:
    """`name` -> `project/name`; `other/name` stays as is."""
    return entry if "/" in entry else f"{project}/{entry}"


@dataclass
class Resolved:
    edges: dict[str, set[str]] = field(default_factory=dict)
    issues: dict[str, tuple[str, str]] = field(default_factory=dict)  # task -> (waiting|blocked, reason)


def resolve(needs: dict[str, list[str]], known: set[str], project_status: dict[str, str]) -> Resolved:
    """Turn `needs` entries into edges between known task ids.

    project_status: name -> pending | ready | failed | removed | disabled. Tasks of pending or disabled
    projects may not be known yet, so an entry pointing there means "waiting", not "unknown".
    """
    res = Resolved()
    ordered = sorted(known)
    for tid, entries in needs.items():
        project = tid.split("/", 1)[0]
        edges: set[str] = set()
        waiting: list[str] = []
        blocked: list[str] = []
        for entry in entries:
            q = qualify(entry, project)
            dep_project = q.split("/", 1)[0]
            pstat = None if is_glob(dep_project) else project_status.get(dep_project)
            if is_glob(q):
                matches = [m for m in fnmatch.filter(ordered, q) if m != tid]
                edges.update(matches)
                if not matches:
                    if pstat == "failed":
                        blocked.append(f"project {dep_project} failed setup")
                    else:
                        waiting.append(f"{entry} (no matching task yet)")
            elif q in known:
                edges.add(q)
            elif pstat == "pending":
                waiting.append(f"{entry} (project {dep_project} not ready yet)")
            elif pstat == "disabled":
                waiting.append(f"{entry} (project {dep_project} disabled)")
            elif pstat == "failed":
                blocked.append(f"{entry}: project {dep_project} failed setup")
            else:
                blocked.append(f"unknown dependency {entry}")
        res.edges[tid] = edges
        if blocked:
            res.issues[tid] = (BLOCKED, "; ".join(blocked))
        elif waiting:
            res.issues[tid] = (WAITING, "waiting on " + ", ".join(waiting))
    return res


def cycles(edges: dict[str, set[str]]) -> list[list[str]]:
    """Strongly connected components that form cycles (size > 1, or a self-loop), via Tarjan."""
    index: dict[str, int] = {}
    low: dict[str, int] = {}
    stack: list[str] = []
    on_stack: set[str] = set()
    out: list[list[str]] = []
    counter = 0

    def visit(v: str) -> None:
        nonlocal counter
        index[v] = low[v] = counter
        counter += 1
        stack.append(v)
        on_stack.add(v)
        for w in edges.get(v, ()):
            if w not in index:
                visit(w)
                low[v] = min(low[v], low[w])
            elif w in on_stack:
                low[v] = min(low[v], index[w])
        if low[v] == index[v]:
            comp = []
            while True:
                w = stack.pop()
                on_stack.discard(w)
                comp.append(w)
                if w == v:
                    break
            if len(comp) > 1 or v in edges.get(v, ()):
                out.append(sorted(comp))

    for v in sorted(edges):
        if v not in index:
            visit(v)
    return out


def evaluate(
    needs: dict[str, list[str]], phases: dict[str, str], project_status: dict[str, str]
) -> dict[str, tuple[str, str]]:
    """Condition of every task that has `needs`: (ok | waiting | blocked, reason).

    phases: task id -> phase of every live task. A dependency is satisfied only when `done`; a `failed`
    or blocked dependency blocks (transitively); anything else means waiting.
    """
    res = resolve(needs, set(phases), project_status)
    cond: dict[str, tuple[str, str]] = {}
    for comp in cycles(res.edges):
        for t in comp:
            cond[t] = (BLOCKED, "dependency cycle: " + " -> ".join(comp + [comp[0]]))

    def of(t: str) -> tuple[str, str]:
        if t in cond:
            return cond[t]
        if t not in needs:
            return (OK, "")
        cond[t] = (WAITING, "")  # provisional; cycles were handled above
        result = res.issues.get(t, (OK, ""))
        waiting_on: list[str] = []
        for d in sorted(res.edges.get(t, ())):
            phase = phases.get(d)
            if phase == "done":
                continue
            if phase == "failed":
                result = (BLOCKED, f"dependency {d} failed")
                break
            if of(d)[0] == BLOCKED:
                result = (BLOCKED, f"dependency {d} is blocked")
                break
            waiting_on.append(d)
        if result[0] == OK and waiting_on:
            result = (WAITING, "waiting on " + ", ".join(waiting_on))
        elif result[0] == WAITING and waiting_on:
            result = (WAITING, result[1] + ", " + ", ".join(waiting_on))
        cond[t] = result
        return result

    return {t: of(t) for t in needs}
