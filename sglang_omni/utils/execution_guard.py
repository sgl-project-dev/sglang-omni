# SPDX-License-Identifier: Apache-2.0
"""Small process-local guards for device work shared by host threads."""

from __future__ import annotations

import contextlib
import threading
import time
from collections.abc import Iterator


class FairDeviceExecutionGuard:
    """Serialize device submissions in FIFO order across host threads.

    A condition/ticket lock is used instead of ``threading.Lock`` so a fast
    generation loop cannot repeatedly reacquire the device while a background
    encoder batch is already waiting.
    """

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._next_ticket = 0
        self._serving_ticket = 0

    @contextlib.contextmanager
    def hold(self) -> Iterator[tuple[int, int]]:
        """Yield ``(ticket, wait_ns)`` after acquiring exclusive execution."""
        wait_started_ns = time.monotonic_ns()
        with self._condition:
            ticket = self._next_ticket
            self._next_ticket += 1
            while ticket != self._serving_ticket:
                self._condition.wait()
        try:
            yield ticket, time.monotonic_ns() - wait_started_ns
        finally:
            with self._condition:
                self._serving_ticket += 1
                self._condition.notify_all()
