# #1754 状态评论存档(2026-08-29 至 2026-09-08 PT)

跟踪 issue sgl-project/sglang-omni#1754 上我发过的状态评论原文。2026-09-07 按
luojiaxuan 的要求把公开 issue 上的连续状态评论清掉(现状改为维护在 issue 正文),
删除前先把全文归档到这里,以免丢掉其中的实测数字与被证伪的方案。

两点说明:

- **第三方引擎名已中性化**:按 2026-09-07 的要求,公开仓库里不写其他框架的名字与
  成绩,原文中的引擎名一律替换为 "the reference engine",数字与结论未改。
- 这里是**原始记录**,不是现状。当前状态看 issue 正文与 `qwen3_tts_r20_vocoder_study.md`;
  下面很多条目互相更正甚至互相推翻,读的时候按时间顺序看,以最后一条为准。


---

## 2026-08-29T00:53:01Z UTC · id 5459306339

T-PR12 is now tracked by #1643, which covers the async decode and penalty handoff slice and stacks on #1640 + #1641.

@nagisa-kunhah worth syncing with @junliu-mde before you start, so the work does not overlap.



---

## 2026-08-31T04:00:45Z UTC · id 5473521252

@BruceLoveDecimal @guozhihao-224 @lijrjyan I will take T-PR5, T-PR6, T-PR8, T-PR10, T-PR15, and T-PR19 from here. If you have local progress, please share the branch or notes so I can reuse it.


---

## 2026-08-31T05:57:19Z UTC · id 5474357148

@leihehehe I’ll take T-PR9 from here. If you already have local progress, please share the branch or notes so I can reuse it.


---

## 2026-09-01T04:09:21Z UTC · id 5488753330

Completed the the reference engine H100 A/B: at 10 RPS the three-seed median p95 audible TTFA is 32.8 ms for the reference engine versus 981 ms for the comparable SGLang streaming stack, with median underrun rates of 0% versus 10.0%. The current main stack also hits a repeatable 1.7B startup SIGKILL on the #1641 / #1649 no-compile path; exact revisions, configs, 1 / 6 / 10 RPS results, and remaining quality caveats are now recorded under Closed.


---

## 2026-09-02T02:50:43Z UTC · id 5503595167

Correction on the 1.7B startup SIGKILL recorded under the the reference engine A/B entry: it is environmental, not a code regression in the #1641 / #1649 no-compile path.

The A/B ran on the shared H100 CI host, whose GitHub Actions runners guard their two-GPU lanes with an evict hook that SIGKILLs any non-CI GPU process while a CI job is active on that lane. The runner lane logs show both of yesterday's current-main startup attempts being evicted about 35 seconds after container start — exactly the pre-weight-load window:

```
2026-09-01T03:35:59Z [reaper gpus=0,1 mode=job] FOREIGN pid=3777949 container=sglang-omni-jaxan-20260901-033524-… — evicting
2026-09-01T03:39:10Z [reaper gpus=0,1 mode=job] FOREIGN pid=3819265 container=sglang-omni-jaxan-20260901-033833-… — evicting
```

A controlled re-attempt today on current main (d6670de + #1852) was killed the same way by the same hook, again with a matching evict-log entry naming the benchmark container. The pre-no-compile stack "starting normally" yesterday was luck of CI-job timing, not a code difference.

Consequences:
- No isolation or fix of #1641 / #1649 is needed for this; I am removing that action item.
- A clean current-main startup plus a refreshed SGLang-side A/B (same harness, 1 / 6 / 10 RPS) will follow once a CI-free GPU lane is available on the host; the the reference engine-side numbers stand.
- Benchmarks on that host must reserve a lane away from the CI autoscaler first; noted for future runs.


---

## 2026-09-02T03:29:24Z UTC · id 5503888651

Refreshed the SGLang side of the H100 A/B on current main d6670de plus #1852 (merge 8a8700d), same harness, dataset, model revision, serving config (mem_fraction_static 0.60, admission 16+16, decode graph buckets 1/2/4/8/12/16, vocoder ramp 2/4/8), and measurement contract as the 2026-09-01 entry. Startup is clean — weights load in 0.95 s and the pipeline is ready in ~42 s, confirming the SIGKILL correction above. the reference engine numbers are the 2026-09-01 measurements; the SGLang column supersedes the "comparable stack" column.

| open-loop RPS | the reference engine p95 audible TTFA | SGLang main+#1852 p95 audible TTFA | prior compatible stack |
|---|---|---|---|
| 1 | 26.4 ms | 162.2 ms | 174 ms |
| 6 | 28.9 ms | 201.7 ms | 219 ms |
| 10 (median of 3 seeds) | 32.8 ms | 438.0 ms | 981 ms |

At 10 RPS: success 100% on all three seeds (was one seed at 98.7%), median underrun rate 1.02% (was 10.0%), p95 end-to-end 1864.7 ms (was 2909 ms; the reference engine 844 ms). The merged Codec line (#1756/#1757), no-compile Talker/Predictor (#1641/#1649), and first-chunk ramp (#1848) account for the step change.

New: 20 RPS probes (3 seeds each side, same contract).

- SGLang saturates: successful 62.4–63.7%, effective served rate ~12.3 RPS, failures are admission rejections (min E2E 0.6 ms), p95 audible TTFA of successes 1481–1531 ms, underruns 3.3–5.0%. The ceiling is admission/scheduling (max_running 16 + max_queued 16), squarely T-PR10/T-PR13/T-PR14 territory.
- the reference engine sustains 20 RPS: three-seed median p95 audible TTFA 38.0 ms (p99 ~56 ms), 100% success on all seeds, underrun rate 0.17–0.77%. Their README's 20 RPS sub-80 ms claim is confirmed with wide margin.

Latency decomposition at 1 RPS (p50): our audible TTFA 139.4 ms = 53.7 ms TTFB + ~80 ms leading silence; the reference engine 16.3 ms with p50 leading silence 0 ms.

- Leading-silence data, per-request over every successful measurement request: SGLang n=2229 — min 70 ms, p50 80 ms, p95 90 ms; not one request became audible before 70 ms (Ryan, English, CustomVoice 1.7B, shipped-default sampling). the reference engine n=2237 — p50 0 ms, 92% within 10 ms. One 12.5 Hz codec frame is 80 ms: the model deterministically emits one silent bootstrap frame under this configuration, and the reference engine drops it. Their ttfa profile config prints `suppressed_bootstrap_chunk_schedule: [2,4,8,12]` next to `chunk_schedule: [1,2,4,8,12]`, i.e. a first-frame suppression path with an adjusted vocoder chunk plan. This re-opens the T-PR16 idea in exactly the narrow form the earlier probe did not cover: CustomVoice-only, allowlist-gated (voice x language x default sampling), evidence n=2229 for Ryan/English. Worth ~80 ms of audible TTFA at every load level.

Roadmap intelligence from the the reference engine ttfa profile execution config (printed by their server at startup):

- `required_policy: deadline_aware`, `pressing_lead_s: 1.0` — deadline-aware scheduling with a playback-lead threshold is their core serving policy (T-PR10/T-PR11).
- `talker_prefill`: exact graphs at sequence length 10 for batch 1–8 plus token buckets [32,64,128,192,384,640] — both the fixed short streaming prompt and the long CustomVoice text prompt are covered by prefill graphs (T-PR15). On our side #1581 already ships the breakable prefill machinery but Qwen3-TTS leaves it default-off; enabling it with an explicit low bucket ladder is the smallest next win.
- `codec` COLD/WARM frame and batch tables — the same lifecycle split #1846/#1855 are building.
- `chunk_schedule` starts at 1 frame (ours starts at 2) — first-chunk ramp headroom (T-PR6 follow-up).
- The whole config is a sha256-stamped packaged profile validated at startup (T-PR13's shape).

Probe: enabling the #1581 breakable prefill graphs on this stack (ladder [4..256], capture 2.79 s, 0.46 GB) and rerunning the same points gives r1 TTFB p50 53.7 -> 46.1 ms, audible TTFA p50 139.4 -> 125.6 ms; at 10 RPS the three-seed median p95 audible TTFA moves 438.0 -> 411.5 ms and the median underrun rate halves (1.02% -> 0.51%). model_info counters show 1106 graph replays concentrated in the 12-20-token buckets (CustomVoice prompt = N_text + 10/11) but 1449 prefills still eager — under load, coalesced prefill batches exceed the 256-token ladder cap, so the default-enable PR should carry the ladder to 512 like Higgs TTS does.

Next actions in order: (1) CustomVoice bootstrap-frame suppression PR (evidence above); (2) default-enable breakable prefill graphs for Qwen3-TTS with an explicit ladder to 512; (3) deadline-aware scheduling design (T-PR10) — the 10-RPS tail and the 20-RPS admission collapse are both scheduling-bound.



---

## 2026-09-02T03:49:02Z UTC · id 5504065416

Two PRs out of tonight's analysis:

- #1900 — default-enable the #1581 breakable prefill CUDA graphs with a 512-token ladder (the enablement half of T-PR15; measured TTFB p50 54 -> 46 ms at 1 RPS, three-seed median p95 audible TTFA 438 -> 412 ms at 10 RPS, capture cost 2.8 s / 0.46 GB).
- #1901 — suppress the silent bootstrap codec frame for streamed CustomVoice under a Ryan/English allowlist with a fail-closed runtime silence guard (the narrow reopening of T-PR16; worth ~80 ms of audible TTFA at every load, evidence n=2229 above). Stacked on #1852.

Remaining from the decomposition: the ~30 ms of post-#1900 TTFB needs a stage-by-stage trace before any scheduler work, and the 10-20 RPS tail is admission/scheduling (T-PR10/13/14) — next step there is a two-class urgency split (requests pre-first-audio vs continuations, with underrun promotion) rather than a full deadline policy.


---

## 2026-09-02T05:16:28Z UTC · id 5504763368

H200 evidence round for the serving roadmap (one GPU, same harness/contract as the H100 entries; this host is noisier than the H100 lane, so treat arms as mutually comparable rather than absolute):

**#1901 validated end to end** — byte-identity, fail-closed guard, and a load regression found and fixed: withholding the bootstrap frame also withholds 80 ms of client playback buffer, which at 10 RPS doubled underruns (26-32% -> 66-76%); decoding one extra frame into the suppressed first chunk (mirroring the rival's separate suppressed chunk schedule) drops underruns to 17.5-19.2% — *below* the no-suppression baseline — while keeping leading silence at 0 ms. Net at 1 RPS on this host: audible TTFA p50 150.1 -> 87.5 ms. Details and tables on the PR.

**T-PR14 negative result, quantified**: raising admission alone (max_running 16->32, max_queued 16->32, decode graphs to bs32; prefill graphs on) at 20 RPS open-loop collapses success from 54.8% (16+16) to **5.5-5.6%** — peak in-flight reaches 71-79, decode cannot keep every admitted stream realtime, and continuity failures kill nearly everything (underruns 51%, E2E p95 ~9 s). The roadmap's dependency note ("do not raise admission independently of playback/backpressure") is now measured fact.

Implication for the ordering: the high-RPS gap cannot be closed by admission or kernel tweaks — the rival's `deadline_aware` policy with `pressing_lead_s: 1.0` is what lets ~32 concurrent streams coexist by spending decode only near playback deadlines. T-PR11 (playback-lead pacing through the existing scheduling path) followed by T-PR14 retuning is the critical path; a design writeup for the pacing slice comes next rather than more config arms.


---

## 2026-09-02T07:31:37Z UTC · id 5506089682

H100 refresh with the full current stack — main d6670de + #1852 + #1901 + breakable prefill graphs (#1900's config), same harness/model/contract as every entry above. Arms ran on lane-local NUMA cores with CI isolated to its own lanes; the suppression control is the identical tree with `suppress_bootstrap_silence false`, same host and hour.

| open-loop RPS | the reference engine p95 audible TTFA | full stack p95 (p50) | control: no suppression | main-only two days ago |
|---|---|---|---|---|
| 1 | 26.4 ms | **100.5 ms (66.5)** | 170.1 ms (136.9) | 174 ms (139.4) |
| 6 | 28.9 ms | 157.9 ms (110.7) | — | 219 ms |
| 10 (median of 3 seeds) | 32.8 ms | **391.9 ms (124.8)** | 430.6 ms (181.7) | 981 ms |
| 20 (median of 3 seeds) | 38.0 ms | ~1385 ms, success ~64% | — | untested (eager stack: 63% success) |

Reading it: at 1 RPS the gap to the rival is now **2.5x at p50** (was 5.3x two days ago); #1901's isolated contribution is -72 ms p50 / -57 ms p95 (identical code, suppression toggled). At 10 RPS the full stack beats its own no-suppression control on both tail latency and underruns (median 1.0% vs 2.7%), confirming the first-chunk lead compensation holds on H100 as it did on H200. The 20 RPS point is unchanged in character — admission-bound with ~36% rejections — and stays the province of the pacing/backpressure line (#1903 and its follow-ups), not of further first-audio work.

Remaining low-load gap (~40 ms p50) is the untraced TTFB tail (T-PR19 instrumentation, then T-PR13 profile work); remaining high-load gap is scheduling (#1903 gap list).


---

## 2026-09-02T07:43:47Z UTC · id 5506220451

T-PR19 first slice delivered with existing machinery: the request-level event recorder (`POST /start_profile` + `python -m sglang_omni.profiler`) already covers admission -> preprocessing -> build -> queue -> prefill -> first emit. Steady-state medians at 1 RPS on the current full stack (H200, warmup excluded, n=56), Talker-side first codec frame end to end **27.2 ms**:

| segment | p50 |
|---|---|
| preprocessing stage | ~8.5 ms |
| request build + queue | ~2.0 ms |
| **prefill (graph-replayed) incl. first frame** | **17.5 ms** |
| first-emit plumbing | 0.4 ms |

Two probes localize the prefill cost: per-request embed composition is only ~0.25 ms (so a fixed-prefix embed cache is *not* worth building — measured, hypothesis dropped), while `model_runner.forward` on the extend batch is **9.4–17.5 ms with high variance despite 85/87 graph replays**. That is the signature of the breakable backend's QK-norm/RoPE graph break from #1581: every layer's excluded segment runs eager between graph pieces, so the "graph" is dozens of segments and the cost is CPU launch overhead, not GPU math.

Updated low-load ordering:
1. **Un-break the prefill graph** — root-cause why captured `apply_qk_norm_rope` corrupts replay (#1581 worked around it) and capture the layer whole; expected ~10-15 ms at c1, which closes most of the remaining gap to the rival's 16 ms TTFB.
2. Preprocessing stage internals (~8.5 ms for tokenize/template on the hot path) — profile before touching.
3. The two decode steps behind the compensated first chunk (~6-8 ms) are the price of #1901's load safety until deadline scheduling lands; not worth attacking separately.


---

## 2026-09-02T08:03:17Z UTC · id 5506452459

Follow-up on the prefill-forward finding — the graph break's cost and the corruption's likely locus are now both measured:

- **Upside of un-breaking**: with the #1581 QK-norm/RoPE break disabled (correctness aside), the extend forward drops from 9.4-17.5 ms to **5.1-8.1 ms** and the jitter tail disappears — ~6 ms p50 and ~10 ms p95 at c1, on top of which the breakable backend's remaining segments could be consolidated further.
- **The corruption is still real**: with capture enabled over the block, the Ryan/English first codec frame — digitally silent in 2229/2229 measured requests on the correct path — comes out at -27.7 dBFS. That invariant makes a cheap, deterministic corruption detector for any future attempt (no WER run needed).
- **The kernel is exonerated**: forcing the non-fused QK-norm + RoPE fallback under capture produces byte-identical corrupted output to the fused path (same seed, same acoustic profile). Two different implementations reading the same inputs and failing identically points at the *inputs*: the positions tensor the captured region reads is not the buffer the prefill graph runner updates at replay (the Talker collapses MRoPE [3, seq] positions to row 0 under the substitution contract — a stale or mis-bound static buffer there corrupts RoPE identically for both paths).

Next step on this thread is upstream: trace which positions buffer the breakable prefill runner updates at replay vs what the captured talker layer was bound to at capture, then either bind them or substitute before capture. Expected win once fixed: prefill 17.5 -> ~11 ms at c1 (and less variance at load), stacking with the earlier decomposition toward the rival's 16 ms total TTFB.


---

## 2026-09-02T08:09:46Z UTC · id 5506525126

**Correction to my previous comment**: the "-27.7 dBFS first frame" corruption evidence was a measurement artifact — that probe used a 160 ms analysis window at t=0, which mixes the silent bootstrap frame with the speech onset behind it. Re-measured with the proper 80 ms window, the unbroken-graph output's first frame is **-104.0 dBFS** — the silence invariant holds.

Follow-up probes on current main (H200, capture over the full QK-norm/RoPE block, no graph break):

- fresh fixed-seed request: byte-identical output with and without an explicit mrope-slot refresh; first frame digitally silent;
- repeated identical requests (radix-warm): byte-identical across repeats and to the reference;
- fused vs non-fused QK-norm/RoPE under capture: byte-identical.

So **the #1581 replay corruption does not reproduce on current main** under fresh, repeated, fused, and non-fused conditions — the guard may be protecting against a runner behavior that has since changed. Given the measured cost of the break (extend forward 9.4-17.5 ms vs 5.1-8.1 ms unbroken, plus the jitter tail), I'll put up a PR that removes the break together with an explicit mrope-positions refresh in `before_prefill` (the Talker binds the mrope slot via `is_mrope_enabled`, and TTS batches otherwise never populate it — feeding it real positions makes the binding sound regardless of which historical case #1581 hit), validated by the consistency CI, fixed-seed equivalence, and the first-frame silence invariant at scale. If the original corruption scenario is known more precisely, review input welcome there.


---

## 2026-09-02T08:22:03Z UTC · id 5506662427

Closing out today's T-PR19 probes with the last two decompositions:

- **Preprocessing stage (8.5 ms at c1)**: the compute inside `_prepare_qwen3_tts_request` is only **2.6-3.6 ms** total (embed build ~2 ms, per-row embedding hash ~0.5 ms, pad-embed ~0.3 ms — none worth attacking individually). The other ~5-6 ms is stage plumbing: payload deserialize, threaded-executor dispatch, state re-serialize, IPC hop. That is diffuse framework overhead — T-PR13 profile/pipeline territory rather than a single fix.
- **Five-PR stack end-to-end sanity** (this tree + #1907 merged, H200, single seed): c1 audible TTFA p50 **68.0 ms**, p95 133.7, silence 0, no underruns. The ~5-6 ms forward-time gain from #1907 shows up only partially end-to-end at c1 (~2 ms vs the with-break boot, within cross-boot noise) — the isolated forward measurements in the PR stand, and the launch-overhead reduction is expected to matter more under load than at c1; the H100 refresh after merges will be the arbiter.

Updated c1 budget on this host: ~5-6 ms stage plumbing + ~3 ms preprocessing compute + ~6 ms prefill non-forward (sampler/predictor/postprocess) + ~5.4 ms unbroken forward + two compensated decode steps + vocode/delivery. No remaining single >10 ms item on the low-load path; the next material gains are (a) merging the five in-flight PRs, (b) the pacing line for 10-20 RPS, (c) T-PR13-style pipeline consolidation of the diffuse ~10 ms of per-request framework overhead.


---

## 2026-09-02T11:05:27Z UTC · id 5508559312

Course correction on the high-load line, backed by a decisive experiment.

**T-PR11 playback pacing is a dead end for this workload** (full data on #1903, now draft). Matched A/B at fixed 16+16 admission, peak in-flight 34 both arms: pacing on cut r10 success 94.9%->44%, raised TTFA p50 213ms->3.5s; underruns 'dropped' to ~0 only because half the requests never became audible in window. Fundamental, not tunable: pacing reallocates deadline slack between streams, but at 10-20 RPS the system is throughput-bound and *no* stream has slack — parking any stream just makes it miss its own deadline. The design review predicted exactly this ('pacing fixes deadline allocation, not fundamental throughput').

So the 10-20 RPS gap to the rival (392ms p95 vs 32.8, and ~50-65% vs 100% success) is a **throughput** gap, not a scheduling gap. The lever is Codec/vocoder service time and batching, i.e. the existing #1846 (batched incremental Codec on arena-backed state) and #1855 (COLD/WARM incremental Codec CUDA graphs) line — not a new scheduler. Redirecting effort there.

Roadmap status update:
- **Low-load (T-PR6/9/15/16/19)**: landing — #1900 (default prefill graphs), #1901 (bootstrap-silence suppression), #1907 (un-break prefill QK/RoPE). c1 68ms p50 measured, ~2.5x from rival, no remaining single >10ms item.
- **High-load (T-PR10/11/14)**: pacing (T-PR11) refuted as a lever; refocusing on Codec throughput (#1846/#1855). T-PR14 admission retune only matters once per-stream service time drops enough that 32 concurrent streams can actually stay realtime.
- **Correctness/quality**: unchanged; WER/SIM still unscored in these perf runs.


---

## 2026-09-02T11:57:37Z UTC · id 5509148758

Second high-load lever measured and refuted, with a clean same-tree A/B.

Same code (main + #1852 + #1846), same host, same 16+16 admission, toggling only `vocoder.factory.enable_stateful_codec_decoder` at r10:

| r10 | success | TTFA p50 | underrun | missing (no first audio in window) |
|---|---|---|---|---|
| default `chunked_decode` (stateful off) | **94.2%** | 249 ms | 26.8% | 34 / 589 |
| stateful incremental Codec, arena-batched #1846 (on) | **25.8%** | 285 ms | 9.7% | 437 / 589 |

0 arena fallbacks on the stateful arm (arena engaged, 64 slots, never exhausted), so this is the batched stateful path running exactly as designed. It generated 719 s of audio in the 60 s window (~12x realtime aggregate) but delivered first audio too late for 437/589 requests. So the stateful path is not throughput-starved in aggregate — its per-stream first-audio latency under concurrency is the problem (cohort-wait + gather/scatter per step), and at r10 that collapses success to ~1/4 of the default path.

Important scoping: **#1846 is correct and does what it claims** — it improves the *stateful* path over its unbatched #1757 form. What this measures is different: the *stateful path as a whole* underperforms the default `chunked_decode` at load by ~3.6x on success. So #1846/#1855 are not the r10/r20 throughput lever; the default path is already far better there.

Net for the high-load line: both scheduled levers are now refuted with data — T-PR11 pacing (no deadline slack at load, #1903) and the stateful-Codec batching path (this). The honest current ceiling on the **default** path is r10 94% success / 27% underrun, r20 53% success, vs the rival's 100% at 38 ms. Closing that is a Codec/vocoder *service-time* problem on the default path (fewer FLOPs / faster kernels / better overlap), not any of the batching or scheduling items on the roadmap. That reframes the remaining high-load work as a kernel/pipeline investigation rather than a scheduled PR.


---

## 2026-09-02T12:26:11Z UTC · id 5509497525

Breakthrough on the high-load line — the real Codec bottleneck is decode CUDA-graph coverage, and widening it closes the r10 gap.

Profiled the **default** chunked_decode path at r10 (the path that beats the stateful one): decode splits sharply by graph coverage — 5-21 ms graphed vs 25-52 ms eager, and **72% of decodes were eager, 88% of total decode time**. The graphs only covered left_context + {ramp strides} (frames 20, 24), but streaming produces startup prefix-sum counts (2, 6, 14 = ~45%) and steady jitter counts (17-23 = ~27%). This is exactly the trick the reference engine advertises in its profile (it graphs a dense frame-count x batch table); we were graphing 2 shapes.

Fix in #1912 — derive and capture the full frame-count span (chunk-schedule prefix sums + contiguous steady band). Measured on the same tree/host, 16+16, only this change:

| point | before | after | reference |
|---|---|---|---|
| r10 success | 94.2% | **100%** | 100% |
| r10 underrun | 26.8% | **0.5-0.7%** | ~0.6% |
| r10 eager decodes | 72% | 13% | - |
| r20 underrun | 22-27% | **0.17%** | ~0.6% |

**r10 is now at parity with the reference.** r20 underruns are eliminated too; the residual r20 success (~66% at 16+16) is pure admission capacity — and because decode is no longer the bottleneck, raising admission to 32 now lifts r20 success to ~80% (was a 5.5% collapse before this change, when decode couldn't sustain the concurrency). Cost: +46 s one-time startup capture.

Roadmap correction, consolidated: the high-load gap was never the two scheduled levers (pacing #1903 refuted; stateful-Codec batching #1846 is ~3.6x slower than default) — it was decode graph coverage on the default path (#1912). Remaining: push r20 the rest of the way via the admission retune (T-PR14), now unblocked, plus mopping the last ~13% eager frame counts.



---

## 2026-09-02T12:41:45Z UTC · id 5509690037

r20 admission retune (T-PR14) measured on top of #1912 — and it defines the remaining gap precisely.

r20 admission sweep (H200, 1.7B CustomVoice, SeedTTS EN, #1912 widened graphs, max_running=max_queued):

| admission | success | underrun |
|---|---|---|
| 16 | 66% | 0.5% |
| 20 | 72% | 1.5% |
| 24 | 77% | 4% |
| 28 | **82%** | 10% |
| 32 | 80% | 15% |

Success peaks at ~82% (adm28) then *falls* at 32 while underruns keep climbing — every stream added past the knee trades success for underruns because the card cannot keep more streams realtime. Sustained generation measured ~75-80 audio-seconds/second; 20 RPS at ~5.7 s/req needs ~114. So single-card capacity at good quality is ~13-14 RPS, ~16 RPS pushing it.

**Conclusion: r20 (20 RPS) at the reference engine's quality is not reachable on one card by admission/scheduling/graph tuning — it is a raw per-frame-throughput wall.** After #1912 fixed decode-graph coverage (which matched underruns at r10 and low-concurrency r20), the residual ~1.4x throughput shortfall must come from per-frame compute reduction: Talker MLP FP8 (#1790, currently opt-in and diagnosed unsafe with the Predictor graph), further Talker/Predictor kernel fusion (beyond the merged #1641/#1649), or a lighter vocoder — the same class of optimization the reference applies.

Final scorecard vs the reference:
- **r1**: ~2.5x on audible TTFA; latency stack #1900/#1901/#1907 pending review + the ~40 ms TTFB tail (T-PR19 trace shows it is diffuse framework/pipeline overhead, T-PR13 territory).
- **r10**: **parity** — 100% success, ~0.5% underrun vs 100%/0.6% (#1912).
- **r20**: underruns matched at low concurrency; success capped ~82% by a single-card throughput wall. Recommended balanced default is admission 20 (72% success, 1.5% underrun, quality preserved); the rest is a compute-reduction problem, not a tuning one.

Actionable T-PR14 outcome: raising the shipped admission default from 16 to ~20 improves r20 success +6 pts at ~1.5% underrun without touching r10 (which never reaches the cap), pending a WER/SIM check before flipping the committed default.



---

## 2026-09-02T14:39:10Z UTC · id 5511316384

Correction to my previous comment: I named Talker MLP FP8 (#1790) as the per-frame-compute lever for r20, but **#1790's own benchmark rules it out** — its summary states the run "does not demonstrate a material end-to-end throughput improvement" (only 5.3-12.9% lower power), and it carries a correctness anomaly (FP8-only 163 s runaway audio). So FP8 is not the r20 throughput fix.

That means the r20 throughput wall has no ready-made lever in the current PR set. The correct next step is the same method that cracked r10 (profile, find the eager/expensive path, fix coverage): profile the per-frame GPU cost at r20 concurrency to locate the actual bottleneck among the Talker AR decode step, the Code Predictor, and the (now-graphed) vocoder — then attack whichever dominates. Starting that profiling now; will report where the r20 per-frame time actually goes rather than guessing at a lever.



---

## 2026-09-02T14:55:29Z UTC · id 5511556718

Definitive close on the r20 gap — profiled to root cause, exhaustively tested, conclusion is a throughput wall.

Talker decode step profiled at r20 concurrency: it is **fully graphed and fast** (3.8 ms p50 for batch ~28) up to `cuda_graph_max_bs`. So the Talker kernel is not the wall. Raising the Talker graph batch + admission to let more streams run concurrently:

| r20 | success | underrun |
|---|---|---|
| adm20 (underrun-safe knee) | 72% | 1.5% |
| adm28 | 82% | 10% |
| max_running 40 + graph_bs 40 | 90% | **52%** |
| max_running 48 + graph_bs 48 | 95% | **85%** |

Completion rate climbs toward 100%, but underruns explode — the extra concurrent streams complete only because the playback-success metric counts completion, while 52-85% of them stalled mid-playback. That is the throughput wall made explicit: the Talker+Predictor+Codec+vocoder pipeline cannot generate frames for ~40-48 concurrent streams in realtime, no matter how the graphs/admission are set.

**Conclusion (evidence-complete):** quality-preserving single-card capacity is ~13-14 RPS (~72% at 20 RPS with <2% underrun). Matching the reference's 20 RPS @ 100%/0.6% requires either less per-frame compute — and the only candidate, Talker MLP FP8 #1790, is shown by its own benchmark not to improve end-to-end throughput — or a second GPU. No scheduling, graph-coverage, or admission setting reaches it; every high-load lever on the roadmap has now been measured.

Every high-load lever, tested:
1. pacing (T-PR11 / #1903) — net-negative at load, refuted
2. stateful Codec batching (#1846) — ~3.6x slower than the default path
3. vocoder decode graph coverage (#1912) — **won r10 (parity), matched r20 underruns at safe concurrency**
4. admission retune (T-PR14) — trades success for underrun past a knee (wall)
5. Talker graph + high admission — pushes completion, explodes underruns (wall)
6. Talker FP8 (#1790) — no throughput gain by its own data

Final scorecard vs the reference: r1 ~2.5x (latency stack #1900/#1901/#1907 pending review); **r10 at parity (#1912)**; r20 quality-preserving ceiling ~13-14 RPS on one card, the remainder being a per-frame-compute / hardware problem, not a serving-logic one.



---

## 2026-09-02T17:03:07Z UTC · id 5513300371

Correction to my "single-card throughput wall" conclusion — it was wrong, and the hardware comparison proves it.

The 2026-09-01 A/B measured the reference at 20 RPS, 100% success, 0.17-0.77% underrun on **one H100 80GB**. My r20 sweeps this session (topping out at ~72% quality-preserving) ran on **one H200** — which is *faster* than an H100 (more bandwidth and compute), same 1.7B model, same revision, same tts-bench harness and success/underrun definitions. So the reference does 20 RPS at full quality on *slower* hardware than we fail to on. That is not a hardware or model-capacity wall; it is a **~2x implementation-efficiency gap**, and it is closeable in principle because the reference already closed it on this exact model.

Where the gap is, from the evidence:

1. **The reference image is literally "CUDA Graph-only".** Its startup config graphs the entire matrix — codec COLD/WARM/whole-sequence over batch {1..12,16,24,32} x frames {1..12}, talker decode + code predictor over the same batch ladder, talker prefill exact graphs. There is no eager path. Our system falls to eager constantly: I measured 72% of vocoder decodes eager before #1912, and the codec/predictor graph coverage is still partial. #1912 (widen vocoder frame graphs) is one slice of "become graph-only" and it won r10; widening the batch dimension alone did not move r20 (tested: still 89%/49-53%), so the remaining eager work is in the codec/predictor path and the full batch x frame matrix, not the vocoder alone.

2. **Single process vs our multi-stage IPC pipeline.** The reference is one HTTP/WS server; sglang-omni runs preprocessing / tts_engine / vocoder as separate processes with cross-process handoffs, plus the Python AR scheduler loop (upstream profiling put that thread near 72% of a core). At r20 concurrency this per-frame plumbing + scheduler overhead is plausibly the dominant serialization cost — the reference's in-process deadline-aware EDF avoids all of it.

So the honest revised scorecard: r10 at parity (#1912); r20 is a closeable ~2x efficiency gap whose levers are (a) full CUDA-graph coverage across codec/predictor/vocoder over the real batch x frame matrix, and (b) reducing the multi-process pipeline / scheduler overhead — architectural work the reference demonstrates is achievable on this model and this class of card, not a wall requiring a smaller model or a second GPU. I retract the earlier "13-14 RPS ceiling / needs second card" framing.



---

## 2026-09-02T20:56:43Z UTC · id 5516287892

Major correction — my r20 conclusions were contaminated by cross-host measurement, and on the correct host r20 is far closer to parity than I reported.

Every r20 sweep I posted in the last several updates ran on a *different* machine (a shared H200 box) than the reference baseline (one H100 on the CI benchmark host). Re-running the identical sweep on the **same H100 host as the reference baseline**, same harness, dataset, model revision, and success/underrun definitions:

| r20, admission | H100 (correct host) | shared H200 (contaminated) |
|---|---|---|
| 16 | 65% success / 6.1% underrun | 66% / 0.5% |
| 20 | **85% / 0.26%** | 72% / 1.5% |
| 24 | **91-93% / 0.34-1.12%** | 77% / 4% |
| 28 | **94-97% / 3.0-3.2%** | 82% / 10% |

Reference engine, same host: 100% success, 0.17-0.77% underrun.

So on the correct hardware, at admission 24 we are at **91-93% success with 0.34-1.12% underrun — underruns already in the reference's range**, and at admission 28 success reaches 94-97%. The remaining gap is ~7% of completions at matched playback quality, not the 30%+ deficit I previously reported.

**Retractions** (all three rested on the H200 numbers):
1. "Quality-preserving single-card ceiling is ~13-14 RPS" — wrong; the H100 sustains 20 RPS at 91-93% with reference-grade underruns.
2. "r20 is a single-card throughput wall requiring a second GPU or a smaller model" — wrong.
3. "We lose on faster hardware, therefore a ~2x implementation-efficiency gap" — invalid: the H200 box was heavily contended by co-tenants, so it was never a hardware comparison at all.

What stands: #1912 (widened vocoder decode-graph coverage) is what makes these numbers possible — before it, r20 underruns were 22-27% at every admission setting. The admission knee moves from 16 to ~24 once decode stops falling to eager.

Concrete T-PR14 outcome: **raise the shipped Qwen3-TTS admission default from 16 to 24**, which on the reference host takes r20 from 65% to 91-93% success while keeping underruns at reference levels, and does not affect r10 (which never reaches the cap). Pending a WER/SIM check before flipping the committed default.

Methodology note for the roadmap: benchmark arms must run on the same host as the baseline they are compared against. The shared H200 box's co-tenant load made it look 15-25 points worse at identical settings, which is what produced the false "throughput wall".



---

## 2026-09-02T21:06:56Z UTC · id 5516407206

r20 follow-up on the correct host: #1901 (bootstrap-silence suppression) is a **low-load win but a high-load regression**, and the best r20 configuration excludes it.

Same H100 host, same harness, admission swept, only difference is whether #1901 is in the stack:

| r20 | without #1901 | with #1901 |
|---|---|---|
| adm24 | **91-93% success / 0.34-1.12% underrun** | 81-82% / 10.8-12.5% |
| adm28 | 94-97% / 3.0-3.2% | 86% / 20.3% |

Reference engine on this host: 100% / 0.17-0.77%.

Cause is the lead compensation: #1901 withholds the first frame's 80 ms of audio and decodes one extra frame into the first chunk to keep the audible lead. At r10 that is free (measured earlier: it *improved* underruns there). At r20 the system runs near capacity, so the extra startup frame per stream is work the pipeline cannot spare, and the thinner playback buffer turns into underruns.

So the current best-known r20 configuration is **main + #1852 + #1912 + prefill graphs at admission 24**, giving 91-93% success at reference-grade underruns, versus 65% at the shipped admission of 16.

This does not sink #1901 — it is a clear win at r1/r10 (−72 ms audible TTFA p50, better underruns at r10). It means the feature should not be unconditionally on for high-concurrency deployments; the natural follow-ups are to scale the compensation with observed concurrency, or gate the suppression when the running-request count is near the admission cap. I will note the same on the PR.



---

## 2026-09-02T21:21:52Z UTC · id 5516599379

r20 root cause identified with direct evidence: it is **per-frame orchestration overhead**, not GPU compute and not graph coverage.

Measured on the canonical H100 host at max_running=48 (the setting that reaches 100% completion):

- **100% success**, audible TTFA p50 183 ms / p95 266 ms, peak in-flight 107, underrun 16.8%
- generated 5558 s of audio in the 60 s window = **92.6 audio-seconds/second**; 20 RPS x 5.7 s/req needs ~114, so we are at **81% of the required throughput — a 23% shortfall**, not a multiple.

Instrumenting the vocoder decode at that concurrency: **6811 decodes, 11 eager (0.2%)**. #1912's coverage holds at high concurrency, and no window larger than 24 frames ever occurs (the backlog hypothesis is disproved). Combined with the earlier per-stage profiling:

| stage | state at r20 concurrency |
|---|---|
| vocoder decode | 99.8% graphed, ~6.5 ms |
| Talker AR decode | 100% graphed, 3.8 ms p50 at batch 28 |
| Code Predictor | graphed (batch ladder tracks cuda_graph_max_bs) |
| GPU headroom vs realtime | ~17x per AR step |

Every GPU stage is graphed and fast, with large headroom against the 80 ms per-frame realtime budget, yet the pipeline cannot keep 48 streams realtime. The missing 23% is therefore in the **per-frame orchestration path**: the Python AR scheduler loop and the cross-process handoffs between preprocessing / tts_engine / vocoder stages. That is precisely the architectural difference from the reference engine, which is a single process and (by its own image description) CUDA-graph-only.

Current best measured r20 configurations on the reference host:

| config | success | audible TTFA p50 | underrun |
|---|---|---|---|
| admission 24 (shipped=16) | 92.7% | 1086 ms | 0.34% |
| max_running 48 | 100% | 183 ms | 16.8% |
| reference engine | 100% | 26 ms | 0.6% |

Neither point matches the reference on all three axes simultaneously; the trade-off curve itself is set by the orchestration ceiling. Also measured: decoupling max_queued from max_running (24 running / 64 queued) does not help — it converts rejections into ~3.2 s of queueing delay at unchanged success.

Actionable conclusion for the roadmap: the remaining r20 work is **T-PR13/T-PR19-class pipeline work** — reducing per-frame Python/IPC cost in the AR scheduler and stage handoffs (upstream profiling previously measured the AR scheduler thread near 72% of a core) — not further kernel, graph, admission, or scheduling-policy tuning, all of which are now measured out.



---

## 2026-09-02T21:30:43Z UTC · id 5516699312

Async decode (T-PR12 mechanism) measured against the r20 orchestration bottleneck — it does not help.

`enable_async_decode` is available in the stage-factory schema but was never plumbed into the Qwen3-TTS engine factory, so the knob could not be set for this model. I added the plumbing (factory arg -> builder -> `extra_scheduler_kwargs`) and measured it at the configuration that reaches 100% completion (max_running 48), on the canonical H100 host:

| r20, max_running 48 | success | audible TTFA p50 | underrun |
|---|---|---|---|
| sync decode (baseline) | 100% | 183 ms | 16.8% |
| **async decode on** | 100% | 184 ms | **18.8-25.5%** |

No improvement; underruns are slightly worse across two seeds, consistent with the extra lookahead step adding jitter rather than hiding Python overhead. (The setting was definitely applied: an unknown factory kwarg raises TypeError at startup — that is how the earlier `suppress_bootstrap_silence` mismatch surfaced — and this server started cleanly.)

That closes the last cheap lever against the orchestration ceiling. Consolidated status of every r20 lever measured on the correct host:

| lever | result |
|---|---|
| vocoder decode graph coverage (#1912) | **won r10 parity**; 99.8% graphed at r20 concurrency |
| admission retune | 65% -> 92.7% success at reference-grade underruns |
| queue depth decoupling (24 running / 64 queued) | no gain; converts rejects into 3.2 s queueing |
| Talker graph batch widening | completion up, underruns explode |
| pacing (#1903) | net negative at load |
| stateful Codec batching (#1846) | ~3.6x slower than default path |
| bootstrap-silence suppression (#1901) | r10 win, r20 regression (-10 pts success) |
| Talker MLP FP8 (#1790) | no throughput gain per its own benchmark |
| **async decode (this round)** | **no gain, slightly worse** |

The remaining r20 gap (TTFA 183 ms vs 26 ms; underrun 16.8% vs 0.6% at 100% completion) is the per-frame cost of the Python AR scheduler loop plus the multi-process stage handoffs. Closing it is architectural work — reducing per-frame Python/IPC in the scheduler and stage boundaries, in the direction of the reference's single-process, graph-only design — and is not reachable by any configuration, kernel, graph, or policy lever available today. All of them are now measured.



---

## 2026-09-02T22:15:19Z UTC · id 5517183747

Retraction and a much sharper r20 diagnosis: **the capacity is there; what is missing is deadline-aware arbitration.** My earlier "single-card throughput wall" framing was wrong.

Instrumented the AR scheduler loop at r20 (H100, concurrency 33-48, decode steps only, n=2575):

| segment | p50 | p95 |
|---|---|---|
| `get_next_batch_to_run` (scheduling decision) | 0.49 ms | 1.22 ms |
| `run_batch` (forward + sampling + code predictor) | 15.26 ms | 30.48 ms |
| `process_batch_result` | 0.17 ms | 1.93 ms |
| **total step** | **16.56 ms** | **31.95 ms** |

One decode step emits one codec frame for *every* running stream, and each stream needs a frame per 80 ms. So 33-48 concurrent streams need ~12.5 steps/s, and a step costs 16-32 ms — roughly 21% of wall clock. Adding prefill for 20 RPS of arrivals (~17.5 ms each, ~35%) puts total utilization near 56%. Python scheduling overhead is negligible (plan+post = 0.66 ms, 4% of the step).

So the machine is **not** saturated at r20, yet we measure 16.8% underruns at concurrency 48. The gap is not average throughput and not Python overhead — it is that **nothing protects the streams closest to their playback deadline** when the step time jitters (p95 is ~2x p50) or when prefill bursts interleave. Frames are produced for all streams uniformly; a stream about to underrun gets no more urgency than one seconds ahead.

That is exactly the mechanism the reference engine implements, and reading its scheduler makes the contrast concrete: its `DeadlineAwarePolicy` picks, per decision, among four stages (talker prefill / talker decode / codec / code predictor) with the order **urgent codec (deadline − reserve reached) -> startup (TTFA protection) -> pressing (within lead) -> round-robin**, batching only deadline-compatible work. It can spend a decision on codec for the streams about to starve and skip talker work for streams already ahead. Our split (talker in the AR scheduler process, codec/vocoder in another process with its own queue) has no component that can make that trade.

Consequences for the roadmap:
- r20 is **reachable by scheduling**, not blocked by hardware or per-frame compute. My prior "needs a second GPU / needs less compute per frame" statements are withdrawn.
- It also explains why T-PR11 pacing failed: pacing *removed* work from streams that were ahead, but never *prioritized* streams that were behind, and it used its own lead metric rather than the actual playback deadline.
- The concrete next step is deadline-aware arbitration inside our pipeline: surface per-request playback deadlines from the vocoder to the AR scheduler, and let the scheduler prefer streams nearest their deadline (and skip streams comfortably ahead) when choosing the decode cohort — the urgent-first ordering above, adapted to our two-process split.



---

## 2026-09-02T22:29:11Z UTC · id 5517328601

**r20 root cause found, and it is a single saturated thread — not an architectural limit.**

Chased the underruns down the pipeline with probes on the correct H100 host at 20 RPS:

1. **AR scheduler is not the bottleneck.** Decode steps run essentially back to back: gap between consecutive decode steps p50 = 0.2 ms, p95 = 36.8 ms, and **zero gaps exceeding 640 ms** (one chunk of audio). Step cost is 16.6 ms p50 (scheduling decision 0.49 ms, `run_batch` 15.3 ms, result processing 0.17 ms), i.e. Python scheduling overhead is 4% of a step. Frame production capacity is ~134 audio-seconds/second against the ~114 that 20 RPS needs.

2. **The vocoder's follow-up worker is oversubscribed.** It is a *single* thread (`_followup_worker`). Instrumented over a 60 s window it did **81.4 s of work — a 135.7% duty cycle**, with batch size pegged at the `followup_max_batch_size=8` cap and 37.2 ms per batch (p95 57.9 ms, max 152 ms). That is the direct cause of the 16-21% underruns: frames exist, but the thread that turns them into PCM cannot drain the queue.

3. **Raising the batch cap does not fix it** (tested: `followup_max_batch_size=32` + `initial_max_batch_size=32` -> batch size actually fell to 6.0, duty 131.4%, underruns 23-27%). Batch size is set by the arrival rate within the 1 ms collection window, not by the cap.

4. **Most of the per-batch cost is not GPU.** The graphed decode is ~6.5 ms of the 37 ms batch; the remaining ~30 ms is staging, the blocking wait on the CUDA completion event, waveform splitting, message construction, and the IPC put — all serialized in that one thread, which idles while the GPU works.

So the fix is concrete: **parallelize or pipeline the follow-up vocoder worker** (multiple worker threads, or overlap the CUDA-completion wait of one batch with the staging of the next). A ~1.4x throughput improvement on that thread is what the measured 135% duty cycle asks for, and it is squarely an implementation change rather than a redesign.

This supersedes my earlier framings, which I withdraw: it is not a single-card throughput wall, not per-frame compute, and not diffuse "orchestration overhead" — it is one identifiable thread at 135% duty. It also explains the shape of every failed lever: pacing, admission, graph coverage, and Talker-side changes all act on a stage that was never the constraint.



---

## 2026-09-02T22:31:20Z UTC · id 5517349326

Implementation notes for the follow-up vocoder worker fix, including two hazards that make the naive version unsafe. I inspected the threading model before writing code and am recording the constraints rather than shipping a change with a latent audio-corruption race.

**Target:** the follow-up worker measured at 135.7% duty (single thread, 37 ms per batch of 8, of which only ~6.5 ms is graphed GPU decode). Roughly 1.4x more throughput on that path is what the measurement asks for.

**Hazard 1 — CUDA graph static buffers are shared per (frames, batch) key.** `_Qwen3TTSInitialDecodeGraphs.decode()` copies codes into `self._inputs[key]`, replays, and returns `self._outputs[key]`. Replay is asynchronous, so a second thread that copies into the same static input while the first thread's replay is still in flight corrupts that decode — silently, as wrong audio rather than an error. Simply adding worker threads that share the graph holder is therefore unsafe. A lock around the launch does *not* fix it either: the buffers stay live until the GPU work completes, so the lock would have to be held across the completion wait, which serializes exactly the part we are trying to parallelize.

**Hazard 2 — pinned staging slots are thread-local.** `_thread_decode_slot()` stores the `_DecodeSlot` in `threading.local()`, and `_Qwen3TTSDecodeHandle.resolve()` releases the slot it was launched with. So a "launch on thread A, resolve on thread B" split (the obvious way to overlap the completion wait with the next launch) releases another thread's slot and breaks the ownership invariant that the retain-on-unproven-completion logic depends on.

**Viable shapes, in preference order:**
1. **Per-worker graph holders and per-worker decode streams.** Each follow-up worker owns its own `_Qwen3TTSInitialDecodeGraphs` instance, CUDA stream, and pinned slot, so hazards 1 and 2 both disappear by construction. Cost is N x graph memory and N x capture time at startup (currently ~46 s for the widened set, so worker count should stay small, 2 is enough for the measured duty). The graph-selection branch in `_launch_async` keys off stream identity (`stream is self._followup_decode_stream`) and must become a per-worker lookup.
2. **Single launcher thread plus a completion pool**, where the pool only performs work that touches no thread-local or graph state. This requires moving slot ownership out of `threading.local` into the handle first; without that refactor it violates hazard 2.

Per-stream ordering is *not* a constraint here: `followup_pending` prevents a stream from being scheduled twice concurrently, and the next chunk is only queued from `_commit_followup`, so chunks of one stream are naturally serialized regardless of worker count.

Supporting evidence that the duty cycle is the operative variable: at admission 24 (about half the concurrent streams, so roughly 68% duty) underruns are 0.34-1.12%, while at 48 (135.7% duty) they are 16.8-20.7%, with everything else held constant.



---

## 2026-09-02T22:45:35Z UTC · id 5517479308

The r20 fix is implemented and measured: #1930 parallelises the saturated follow-up vocoder worker and **halves r20 underruns**.

| 20 RPS, max_running 48 | success | underrun |
|---|---|---|
| 1 follow-up worker (before) | 100% | 16.8% / 20.7% |
| **2 workers (new default)** | **100% / 100%** | **8.6% / 9.5%** |
| 4 workers | 99.4% / 100% | 21.8% / 16.4% |

Two workers is the measured optimum, matching the 135.7% duty cycle (about 68% each); four is worse than two because the workers then contend on GPU streams and CPU with the AR scheduler thread. Each worker owns its own CUDA stream and decode-graph holder — sharing a holder would race on the graphs' static input buffers and emit corrupted audio silently, and a lock could not fix that without re-serialising the launch (details on the PR).

Updated scoreboard against the reference engine, all on its own benchmark host:

| load | ours | reference | state |
|---|---|---|---|
| 1 RPS | ~36 ms TTFB after #1928; audible TTFA ~36-40 ms with #1901 | 26.4 ms | ~1.4x |
| 10 RPS | 100% success, 0.5% underrun | 100%, 0.6% | **parity** |
| 20 RPS | 100% success, 8.6-9.5% underrun | 100%, 0.17-0.77% | success matched, underrun 10-14x |

r20 completion is now at parity and continuity is half of what it was, with the remaining gap no longer attributable to any single saturated component: the AR scheduler has headroom, decode is 99.8% graphed, and the follow-up path now runs at roughly 68% duty per worker. The next candidate is the ~30 ms of per-batch CPU work inside each follow-up batch (staging, completion wait, waveform split, message build, IPC) — pipelining that so a worker stages the next batch while the previous one's CUDA completion is still pending, which needs the pinned-slot ownership moved off `threading.local` first (hazard 2 in the notes above).



---

## 2026-09-02T23:13:56Z UTC · id 5517729443

## Status: consolidating the the reference engine-parity work into mergeable PRs

Switching from "find the next win" to "land what is already validated". Six PRs came out of this line; five are on track to merge and one is now blocked by a regression I found while clearing CI.

### Ready (CI green apart from the known `xpu-ci` outage, #1905)

| PR | Change | Measured |
|---|---|---|
| #1900 | Breakable prefill CUDA graphs on by default | prerequisite for the prefill work |
| #1901 | Suppress the silent bootstrap frame (streamed CustomVoice) | ~80 ms audible TTFA, now gated to <=24 live streams |
| #1912 | Widen vocoder decode graph coverage | r10 to parity: 100% success / 0.5% underrun (reference: 100% / 0.6%) |
| #1928 | Ship the first-audio chunk ramp on by default | TTFB 90.1 ms -> 36.0 ms |
| #1930 | Parallel follow-up vocoder workers | r20 underrun 16.8-20.7% -> 8.6-9.5% |

`xpu-ci (device_layer)` is red fleet-wide (#1905, torch+xpu image build exits 127) and is not merge-blocking — #1910 merged with it red.

### Blocked: #1907 (converted to draft)

#1907 removed the QK-norm/RoPE graph break and mirrored plain positions into `mrope_positions`. Two findings kill it in its current form:

1. **The graph-break removal is inert on `main`.** `main` does not enable the breakable prefill backend — that arrives with #1900, which #1907 is not stacked on. With no breakable-prefill context there is nothing for the removal to affect.
2. **The mrope mirroring regresses voice cloning.** `_ensure_mrope_positions` populated `mrope_positions` on every prefill, and `sglang_model.py:1113` switches `positions` to it whenever it is non-None — so every TTS prefill moved onto the mrope path with synthesised `[3, T]` values.

`TTS CI / stage 1`, `test_voice_cloning_similarity`, same runner/model/preset:

| branch | `speaker_similarity_mean` | gate 64.07 |
|---|---|---|
| #1900 | **70.80** | pass |
| #1907 | **63.41** / 63.71 | **fail** |

Reference constant is 69.006, so ~70.8 is healthy and this is a ~7-point drop, reproduced across all 3 CI retries (each regenerating audio). WER stays 0.0 — content is intact, what degrades is speaker identity, which is what mrope-routed conditioning positions would affect.

Worth recording: the comment #1907 deleted said exactly this — *"Capturing the packed QKV normalization and RoPE block corrupts Qwen3-TTS prefill replay."* The PR's hypothesis was that the corruption was really missing mrope positions. The data says no.

To revive it: rebase onto #1900 so the breakable path is actually live, scope the mrope population to graph capture/replay instead of every prefill, and require `speaker_similarity_mean` back at ~70 before re-opening.

### Still open

The ~30 ms of per-batch CPU cost in each follow-up vocoder batch (of which only ~6.5 ms is graphed GPU decode) is measured in aggregate but not yet decomposed into staging / completion-wait / waveform-split / message-build / IPC. That is the next lever on r20 and it is deliberately deferred until the five PRs above land — they all touch `streaming_vocoder.py`, so stacking a sixth change on the same file was costing more in conflict resolution than it bought.



---

## 2026-09-03T19:29:13Z UTC · id 5531013656

## Correction, and status

**Correction first.** In my previous comment I reported that #1907 caused a ~7-point voice-cloning regression. That was wrong. I compared a `moss` run against a `qwen3-tts` run without noticing the stage is parameterised by `TTS_CI_MODEL`:

| run | `TTS_CI_MODEL` | `speaker_similarity_mean` | gate |
|---|---|---|---|
| #1900 | `qwen3-tts` | 70.80 | 69.006 (`QWEN3_TTS_VC_SIMILARITY_MEAN_MIN`) |
| #1907 | `moss` | 63.71 | 64.073 (`MOSS_VC_SIMILARITY_MEAN_MIN`) |

#1907 touches only `sglang_omni/models/qwen3_tts/`, so it cannot affect MOSS-TTS. The same MOSS gate also came up 0.22 short on #1928 (62.85 vs 63.076), another Qwen3-TTS-only change, so it looks like a marginal gate rather than anything either PR did. Worth watching if it keeps landing just under.

The other half of that finding stands and is the real reason #1907 is in draft: `main` does not enable the breakable prefill backend, so removing the QK-norm/RoPE graph break is inert there.

## Merged

**#1912** is in as `8dda280c`. Widening the vocoder decode graph coverage takes r10 from 94% success / 27% underrun to 100% / 0.5%, matching the reference engine's 100% / 0.6%. Reviewer caught a real bug before merge: I had saturated every window at `left_context + steady_stride`, which silently dropped any stride wider than the steady one back to eager. Each stride now saturates at its own `left_context + stride`.

## Open

| PR | State |
|---|---|
| #1930 parallel follow-up workers | approved, rebased, r20 re-measurement outstanding |
| #1928 first-chunk ramp default | rebased, needs review |
| #1900 breakable prefill default | rebased, needs review, otherwise green |
| #1901 bootstrap silence suppression | rebased, needs review |
| #1907 prefill graph break | draft, see correction above |

All five are rebased onto current `main` with no merge commits. #1930's PR body still carries r20 numbers measured before batch collection was serialised, so those get re-measured before it merges.



---

## 2026-09-08T02:25:57Z UTC · id 5578169739

## Correction

The "~30 ms of per-batch CPU cost in each follow-up vocoder batch" I recorded on 2026-09-03 is wrong. A four-stage probe on main `91e9c309` (r20, 1173 requests per arm, one H100 80GB) decomposes `_run_followup_batch` as:

| stage | mean | p95 | share |
|---|---|---|---|
| plan (build under lock) | 0.69 ms | 1.95 ms | 4.3% |
| launch (staging + graph replay) | 1.43 ms | 3.16 ms | 9.0% |
| resolve (wait on the completion event) | 13.59 ms | 27.88 ms | 85.4% |
| commit (waveform split, message, IPC) | 0.21 ms | 0.74 ms | 1.3% |

CPU is 2.3 ms per group in total. The r20 lever is vocoder GPU capacity, not host overhead, and the chunk ramp is what consumes it: `(1,2,4)` into a steady 8 puts every stream through windows 1/3/7/15/23 before the steady 24, and 64% of decode calls are those transients, taking 57% of vocoder time. Dropping the ramp entirely gives 0.00% underrun at a TTFA p50 cost of 155 to 329 ms, which is direct evidence the system was just over its capacity threshold rather than CPU bound.

## Merged since that comment

| PR | merged (PT) | commit | |
|---|---|---|---|
| #1852 | 09-03 12:38 | `21254063` | T-PR5, codec streaming for CustomVoice and VoiceDesign |
| #1930 | 09-04 12:37 | `af76a23b` | parallel follow-up vocoder workers |
| #1900 | 09-04 15:31 | `ef7ae016` | breakable prefill CUDA graphs on by default |
| #1928 | 09-04 18:44 | `db9d5a8f` | T-PR6, first-audio chunk ramp on by default |
| #1901 | 09-04 21:59 | `91e9c309` | bootstrap silence suppression |
| #1907 | 09-07 19:47 | `9d148b22` | T-PR15, QK-norm and RoPE captured in the prefill graph |
| #1997 | 09-07 19:48 | `836cdd77` | T-PR8 and T-PR9, stateful incremental codec decoding by default |

#1907 turned out not to be blocked after all. The "inert on main" finding held only while main lacked the breakable prefill backend, and #1900 landed that on 09-04. The other half of it is handled too: the mrope mirror is scoped by the prefill runner's own `can_run_graph` verdict rather than by a null runner, since SGLang hands out an `EagerRunner` instead of `None` when prefill graphs are off.

#1997 supersedes #1846 (@BruceLoveDecimal) and #1855 (@leihehehe), both merged in with their review findings addressed and both credited as co-authors on the squash. @BruceLoveDecimal @leihehehe, please close them once you are satisfied with how it landed.

## Open

**#1998** is the only one left in this chain, rebased on the new main. #1907 landing is what makes its default safe: `eager_on_graph` only fires inside `BreakableCUDAGraphCapture`, so under the full backend the QK-norm and RoPE block would have been captured after all and would have read the bound `mrope_positions` slot that a TTS batch never populates. With #1907 in, the graph break is gone and the slot is refreshed for every replayed batch, so there is no unguarded RoPE left in the captured region. Its output parity figures were measured before #1907, so the seeded 74% column gets rerun on the rebased tree before merge.

## Roadmap body updated

- T-PR8 and T-PR9 moved to Landed as #1997, with the #1846 and #1855 lineage recorded. T-PR9 had still read "no PR yet".
- T-PR5 moved to Landed as #1852. The entry still pointed at #1847, which was closed in its favour.
- T-PR15 now records #1900 and #1907 as landed and #1998 as open. It had still read "no PR yet".
- T-PR18 no longer says review is blocked. #1794 merged on 09-02.
- T-PR6's Landed entry adds #1928, which shipped the ramp on by default.
- T-PR19's observability half is recorded as landing with #1997.

## Still open

The the reference engine gap splits in two and only one half is closed. Underrun at r20 went from 17.3% to 0.37% against the reference engine's 0.5%, so load degradation is handled. Audible TTFA p50 is 68 ms against the reference engine's 26 ms, still 2.6x, and the r20 budget puts the two largest remaining items on the Talker side rather than in any kernel: admission waits a whole decode step at 13.9 ms, and prefill to first frame is another 13.9 ms. The scheduler loop processes new requests once per iteration and blocks on the previous step's GPU event first, which is what makes admission cost a full step. Nothing in flight covers that. T-PR19's bounded PCM delivery slice also still has no PR.
