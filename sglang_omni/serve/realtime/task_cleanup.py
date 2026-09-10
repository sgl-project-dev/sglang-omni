"""Bounded teardown of local asyncio plumbing, never native resource ownership."""

import asyncio


async def cancel_local_tasks(tasks, timeout):
    tasks = {
        task
        for task in tasks
        if task is not None and task is not asyncio.current_task()
    }
    pending = {task for task in tasks if not task.done()}
    # Note (Junnan Li): A reader's first cancellation can enter an iterator finally that shields
    # coordinator-owned cleanup. A second cancels only that local wait; the
    # coordinator still owns the native release/quarantine operation.
    for _ in range(2):
        if not pending:
            break
        for task in pending:
            task.cancel()
        _, pending = await asyncio.wait(pending, timeout=timeout)
    done = tasks - pending
    if done:
        await asyncio.gather(*done, return_exceptions=True)
    if pending:
        raise RuntimeError("local realtime task did not acknowledge cancellation")
