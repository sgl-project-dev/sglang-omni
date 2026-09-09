# SPDX-License-Identifier: Apache-2.0
from array import array
from dataclasses import asdict
from types import SimpleNamespace

from sglang.srt.managers.schedule_batch import Req
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.session.session_controller import SessionController

from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.proto.session import SessionRef, TimedChunk
from sglang_omni.scheduling.sglang_backend.ar_session import ARSessionBridge
from sglang_omni.scheduling.sglang_backend.request_data import SGLangARRequestData


def payload(op, rid="r", epoch=0):
    return StagePayload(
        request_id=rid,
        request=OmniRequest(
            inputs=None,
            metadata={
                "omni_session": {
                    "op": op,
                    "ref": asdict(SessionRef("s", epoch=epoch)),
                    "stages": ["ar"],
                    "chunk": asdict(TimedChunk("text", 0, 0, 0, "x")),
                    "output_limits": {"chunks": 2, "bytes": 1024},
                }
            },
        ),
        data={"relayed": True},
    )


def data(rid="r", ids=(1, 2)):
    params = SamplingParams(max_new_tokens=2, temperature=0)
    return SGLangARRequestData(
        req=Req(rid, None, array("q", ids), params, vocab_size=32)
    )


def bridge():
    cache = SimpleNamespace(
        release_session=lambda _: None, release_radix_session=lambda _: None, slots={}
    )
    scheduler = SimpleNamespace(
        session_controller=SessionController(cache),
        tree_cache=cache,
        max_running_requests=2,
        model_config=SimpleNamespace(vocab_size=32),
        _async_pending=None,
        _aborted_request_ids=set(),
        _resolve_pending_async=lambda: None,
    )
    return ARSessionBridge(scheduler, SimpleNamespace())
