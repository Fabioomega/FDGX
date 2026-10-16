"""Pure GPU allocation: which waiting tasks start now, and on which GPUs."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Request:
    key: str
    vram_gb: float  # per GPU
    min_gpus: int
    max_gpus: int
    starving: bool = False  # waited too long: stop backfilling smaller tasks past it


def allocate(requests: list[Request], usable_gb: dict[int, float],
             slots: dict[int, int] | None = None) -> list[tuple[str, list[int]]]:
    """Greedy, in request order (callers sort by priority/FIFO).

    Each task gets as many fitting GPUs as are available, between min_gpus and max_gpus,
    preferring the GPUs with the most free memory. If a starving task does not fit, nothing
    behind it is scheduled, so GPUs drain until it can run. `slots` caps how many more jobs a GPU
    may take (GPUs not in it are uncapped).
    """
    usable = dict(usable_gb)
    slots = dict(slots or {})
    out: list[tuple[str, list[int]]] = []
    for r in requests:
        if r.max_gpus == 0:
            out.append((r.key, []))
            continue
        fit = sorted((g for g, free in usable.items() if free >= r.vram_gb and slots.get(g, 1) > 0),
                     key=lambda g: (-usable[g], g))
        if len(fit) >= max(r.min_gpus, 1):
            chosen = sorted(fit[: r.max_gpus])
            for g in chosen:
                usable[g] -= r.vram_gb
                if g in slots:
                    slots[g] -= 1
            out.append((r.key, chosen))
        elif r.starving:
            break
    return out
