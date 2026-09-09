# SPDX-License-Identifier: Apache-2.0
import asyncio

import pytest


def test_raw_pcm_append_preserves_capacity():
    from sglang_omni.serve.realtime.audio_buffer import (
        BufferOverflow,
        RealtimeAudioBuffer,
    )

    buffer = RealtimeAudioBuffer(max_bytes=4)
    assert buffer.append_bytes(b"\0\0") == 2
    with pytest.raises(BufferOverflow):
        buffer.append_bytes(b"\0" * 4)
    assert buffer.num_bytes == 2


@pytest.mark.asyncio
async def test_runtime_created_admission_clock_clear_and_eos():
    from sglang_omni.serve.realtime.runtime import (
        Capabilities,
        RuntimeLimits,
        SessionRuntime,
    )

    class Adapter:
        async def open(self, session_id, config, emit):
            self.session_id = session_id

        async def process(self, unit, epoch):
            return unit.real_samples

        async def cancel(self):
            pass

        async def close(self):
            pass

    adapter = Adapter()
    runtime = SessionRuntime(
        "mock", Capabilities(native_unit_ms=20), lambda: adapter, RuntimeLimits()
    )
    assert runtime.state == "CREATED" and not hasattr(adapter, "session_id")
    await runtime.update({}, "open")
    assert adapter.session_id == runtime.session_id
    await runtime.append(b"\0\0" * 80, 0, 0, "a")
    await runtime.clear("clear")
    assert runtime.accepted_samples == 80
    await runtime.append(b"\0\0" * 80, 1, 5, "b")
    await runtime.end("end")
    await asyncio.wait_for(runtime.drained.wait(), 1)
    assert runtime.consumed_samples == 80 and runtime.discarded_samples == 80
    await runtime.close("test")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "policy,expected_padding", [("flush", 0), ("pad", 15), ("reject", None)]
)
async def test_tail_policy_is_executed(policy, expected_padding):
    from sglang_omni.serve.realtime.runtime import (
        Capabilities,
        ProtocolError,
        RuntimeLimits,
        SessionRuntime,
    )

    class Adapter:
        async def open(self, *args):
            pass

        async def process(self, unit, epoch):
            assert len(unit.pcm) == (640 if policy == "pad" else 160)
            return unit.real_samples

        async def close(self):
            pass

    runtime = SessionRuntime(
        "mock", Capabilities(tail_policy=policy), Adapter, RuntimeLimits()
    )
    await runtime.update({}, "open")
    await runtime.append(b"\0\0" * 80, 0, 0, "append")
    if policy == "reject":
        with pytest.raises(ProtocolError):
            await runtime.end("end")
        assert not runtime.eos
    else:
        await runtime.end("end")
        await asyncio.wait_for(runtime.drained.wait(), 1)
        assert runtime.ms(runtime.padding_samples) == expected_padding
    await runtime.close("test")


@pytest.mark.asyncio
async def test_unit_done_follows_successful_processing():
    from sglang_omni.serve.realtime.runtime import (
        Capabilities,
        RuntimeLimits,
        SessionRuntime,
    )

    class Adapter:
        async def open(self, *args):
            pass

        async def process(self, unit, epoch):
            await released.wait()
            return unit.real_samples

        async def close(self):
            pass

    released = asyncio.Event()
    runtime = SessionRuntime("mock", Capabilities(), Adapter, RuntimeLimits())
    await runtime.update({}, "open")
    await runtime.append(b"\0\0" * 320, 0, 0, "append")
    await asyncio.sleep(0)
    assert not any(
        type(env.event).__name__ == "UnitCompleted" for env, _ in runtime.output
    )
    released.set()
    await asyncio.sleep(0.01)
    assert any(type(env.event).__name__ == "UnitCompleted" for env, _ in runtime.output)
    await runtime.close("test")
