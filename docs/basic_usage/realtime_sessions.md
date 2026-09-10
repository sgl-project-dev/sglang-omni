# Shared realtime sessions

`enable_realtime=True` mounts `/v1/realtime` through the shared protocol and
turn-based adapter. `create_app(..., realtime_deployment=...)` selects an explicit
adapter and capability contract for the same route.
`GET /v1/realtime/capabilities` reports deployment capabilities without opening
model state. The existing router can select this worker with `?model=...` and
continues to receive `session.created` as its first application event.

This is a PCM16 event subset with SGLang extensions, not a complete OpenAI GA
API or SDK compatibility claim. There are no HTTP session precreation, resume,
truncate, video, tool, or model implementations here.

## Deployment

For an existing Client whose completion interface supports the turn-based
conversation request path:

```python
from sglang_omni.serve.openai_api import create_app
from sglang_omni.serve.realtime.adapters import TurnBasedAdapterFactory
from sglang_omni.serve.realtime.manager import RealtimeDeployment
from sglang_omni.serve.realtime.runtime import Capabilities, RuntimeLimits

app = create_app(
    client,
    model_name="deployed-model",
    realtime_deployment=RealtimeDeployment(
        capabilities=Capabilities(interaction="turn_based"),
        adapter_factory=TurnBasedAdapterFactory(client, "deployed-model"),
        limits=RuntimeLimits(max_input_bytes=1_920_000),
    ),
)
```

This configuration uses 16 kHz mono PCM16 input and text output. Explicit
`audio.input.turn_detection=null` disables VAD; configured server/semantic VAD
uses the existing detector builder and turn computation. Semantic VAD requires
an already loaded `smart_turn_model` supplied to `TurnBasedAdapterFactory`.
Shared deployment does not automatically download/load Smart Turn. Server VAD
interruption follows the granted policy, advances the runtime epoch, and waits for
the active turn to finish and retain its history while suppressing old output. Native interaction never constructs
VAD or cancels merely because speech arrives.

`CoordinatorAdapter` instead binds a fixed stage route, request builder and
typed output converter to the public Client session methods. Its deployment
must explicitly declare the producer's atomic input consumption contract.
Native outputs are bounded and buffered until the corresponding `input_done`
receipt. Receipts describe completed pipeline units; they are consumption
receipts only under that producer contract. Cancellation fences output immediately and waits for the in-flight unit
to complete before advancing stage ownership to the new epoch. Its input receipt
is still consumed; unknown progress closes the session. The same output iterator
survives cancellation. Core KV is retained at this safe boundary.

The tests provide mock native and transcription producers plus a real
Coordinator/multiprocess mock-stage integration. This does not enable a real
native duplex model or migrate model ASR implementations.

## Wire example

Connect `/v1/realtime?model=deployed-model` and wait for `session.created`.
That identity exists before model allocation. Send:

```json
{"type":"session.update","event_id":"open-1","session":{"type":"realtime","output_modalities":["text"],"audio":{"input":{"format":{"type":"audio/pcm","rate":16000},"turn_detection":null}}}}
```

Wait for `session.updated.session.sglang.granted`, then send PCM16 bytes as base64:

```json
{"type":"input_audio_buffer.append","event_id":"audio-1","audio":"AAA=","sglang":{"seq":0,"t_start_ms":0}}
{"type":"sglang.input_audio.end","event_id":"end-1"}
```

The one sample above is accepted with `accepted_end_ms=0.0625`. EOS produces
`sglang.input_audio.ended` followed, after processing, by
`sglang.input_audio.drained` with real consumed/discarded and separate padding
milliseconds. A real speech request should contain actual recorded PCM audio.
External append boundaries may differ from native units; internal unit sequence
numbers are independent of external `sglang.seq`. Grant fixes the native cadence
and tail policy. A requested fixed external microturn is explicitly downgraded
to variable chunk lengths. Flush, pad and reject tails are implemented;
reject leaves the input open for more samples.

All commands require `event_id`. Successful commands correlate with
`client_event_id`; errors correlate with `error.event_id`. Malformed JSON,
binary/base64/PCM frames and rejected input sequence/time checks preserve the
connection. Rejected input does not advance its sample clock or sequence.
Clear/cancel never reset accepted position. Config changes are atomic; this
hot-update allowlist is `output_modalities` and `audio.input.turn_detection`.
Updates apply to subsequent units; an existing response retains its output
modality. Audio-only output carries text through output-audio-transcript events.

`response.cancel` advances epoch even with no active response, preserves pending
input, and ends affected visible responses before `sglang.response.cancelled`.
Old producer output is fenced again at the transport. Playback ACK uses
`sglang.playback.ack` with `response_id`, `item_id`, `content_index`,
`audio_end_ms`, and `sglang:{epoch:...}`. It advances monotonically only through
audio successfully sent on this socket; success emits no reply. ACK does not
change model history or KV. The single output content index is 0. Output metadata lives under `sglang`,
including epoch, unit ID, chunk sequence and input media time. Errors use
`error.param` and `sglang.fatal`.

Transcription uses `item_id` for an utterance and segment IDs for its pieces.
Segment finalization emits `.segment`; `.completed` finalizes the utterance.
Each segment freezes independently. Revisions use the SGLang extension event.

Manual commit and response.create are not granted by these adapters. EOS
finalizes remaining turn audio. `session.close`, disconnect, and the duration
limit discard pending input and wait for safe adapter cleanup without an
implicit flush. Normal `session.closed` reports zero held resources only after
cleanup acknowledgement; a cleanup timeout is fatal and never reports zero.
Input, output, response/segment ledgers, queued turn audio and history are
bounded. Overflowing output fails the affected session through reserved terminal
capacity; context exhaustion closes with reason `context_limit`.
