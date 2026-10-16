"""Background thread that keeps the console (and therefore the instance) alive."""

from __future__ import annotations

import subprocess
import threading
import time
from typing import Callable


class Heartbeat(threading.Thread):
    """Prints status lines every `interval` seconds no matter what the main loop is doing.

    Optionally runs `keepalive_argv` every `keepalive_seconds`, for machines that judge
    "idle" by more than console output.
    """

    def __init__(
        self,
        status: Callable[[], list[str]],
        emit: Callable[[str], None],
        interval: float,
        keepalive_argv: list[str] | None = None,
        keepalive_seconds: float = 60.0,
    ):
        super().__init__(name="fdgx-heartbeat", daemon=True)
        self.status = status
        self.emit = emit
        self.interval = interval
        self.keepalive_argv = keepalive_argv
        self.keepalive_seconds = keepalive_seconds
        self._stop_event = threading.Event()
        self._keepalive: subprocess.Popen | None = None
        self._last_keepalive = 0.0

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        while not self._stop_event.wait(self.interval):
            try:
                lines = self.status()
            except Exception as e:  # the heartbeat must never die
                lines = [f"[{time.strftime('%H:%M:%S')}] fdgx alive (status error: {e!r})"]
            for line in lines:
                self.emit(line)
            self._tick_keepalive()

    def _tick_keepalive(self) -> None:
        if not self.keepalive_argv or time.time() - self._last_keepalive < self.keepalive_seconds:
            return
        if self._keepalive is not None and self._keepalive.poll() is None:
            return  # previous one still running
        self._last_keepalive = time.time()
        try:
            self._keepalive = subprocess.Popen(
                self.keepalive_argv, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        except OSError as e:
            self.emit(f"keepalive_cmd failed to start: {e}")
