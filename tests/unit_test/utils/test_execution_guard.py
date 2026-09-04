# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading

from sglang_omni.utils.execution_guard import FairDeviceExecutionGuard


def test_fair_device_execution_guard_serves_waiters_in_ticket_order() -> None:
    guard = FairDeviceExecutionGuard()
    first_entered = threading.Event()
    release_first = threading.Event()
    order: list[int] = []

    def run(worker_id: int) -> None:
        with guard.hold():
            order.append(worker_id)
            if worker_id == 0:
                first_entered.set()
                assert release_first.wait(timeout=2)

    first = threading.Thread(target=run, args=(0,))
    first.start()
    assert first_entered.wait(timeout=2)

    waiters = [threading.Thread(target=run, args=(worker_id,)) for worker_id in (1, 2)]
    waiters[0].start()
    # Starting the second waiter only after the first is blocked makes ticket
    # assignment deterministic without relying on scheduler timing.
    while guard._next_ticket < 2:
        pass
    waiters[1].start()
    release_first.set()

    first.join(timeout=2)
    for waiter in waiters:
        waiter.join(timeout=2)
    assert not first.is_alive()
    assert all(not waiter.is_alive() for waiter in waiters)
    assert order == [0, 1, 2]

