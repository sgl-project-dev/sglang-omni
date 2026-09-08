# 外审:Qwen3-TTS follow-up 解码窗口对齐

- 日期:2026-09-05 PT
- 模型档位:最新旗舰 + 极高推理档(思考 34s),临时会话
- 送审对象:把一个 collect batch 内所有 plan 右侧补齐到批内最宽窗口,使一批恒等于一次解码

## 我送审前的判断

vocoder 在 r20 下饱和(占用率 76%,underrun 16-19%),根因是 chunk ramp 让每条流
经过 1/3/7/15/23 五个各不相同的窗口宽度,而 decode batch 只能装同形状的行,
导致实测每组只有 1.83 行。补齐可行,因为实测"补零"与"补真实未来帧"给出完全
相同的误差,且该误差与改变 batch size 造成的误差同量级。

## 它的意见全文

I would not reject the idea. I'd reject calling it proven yet.

### (a) The three weakest points

1. **Your causality evidence is suggestive, but the experiment is confounded by
   tensor shape/kernel selection.** The strongest observation is actually "zero
   future == true future": that strongly implies future values are not affecting
   emitted samples. Good. But padding changes the tensor shape, and you already
   know shape/batch changes alone move the result: batch 1->2 gives relative L2
   ~0.009 versus ~0.012 for padding. So most of the observed delta may simply be
   bf16/kernel-path variation. A reviewer will ask for a controlled experiment
   where shape, batch size, row position, and kernel path are held constant, and
   only the valid-length/padded-region condition changes. Also report tails, not
   just median: p95/p99/max by window size and by chunk boundary.

2. **Waveform L2 is almost the wrong quality metric for this failure mode.** It
   can punish harmless sub-sample/phase differences while missing the thing users
   will actually notice: a click, timbre twitch, or prosody discontinuity exactly
   at an emitted-chunk seam. A 0.012 median says very little about perceptual
   equivalence. The scary case is not "the whole waveform moved slightly." It is
   "one 5-20 ms region at the 7->15 or 23->24 transition occasionally pops."

3. **"One collect = one decode" is not automatically a throughput win.** This is
   the biggest systems objection. Your own measurements say that at batch 8, 3
   frames costs 7.6 ms but 24 frames costs 23.3 ms, a 3.1x increase. Small windows
   are launch-bound; large batched windows are not. If one mature 24-frame stream
   causes several ramp rows to be padded to 24, you may trade shape fragmentation
   for a lot of real compute. Optimize GPU time per emitted frame / deadline met,
   not number of decode launches.

### (b) Quality evidence I would require

Tighter numerical experiment: identical codec tokens, exact-shape baseline vs
padding, controlling batch size, row index/order, repeated execution. Cover
widths 1, 3, 7, 15, 23, 24 and every larger padding target. Report p99/max, not
just median.

Audible-defect metrics: multi-resolution STFT/log-mel error, seam-local analysis
over +-20-50 ms around every emitted boundary, waveform/derivative
discontinuities, spectral jumps, F0 jumps, energy jumps. Cross-correlate before
L2 so a tiny temporal shift does not dominate.

Semantic integrity: ASR WER/CER baseline vs padded, speaker-embedding similarity,
F0/energy/prosody. PESQ/STOI as supporting evidence only.

Blind paired listening test with a pre-registered non-inferiority margin, long
utterances, and the production case mix (mixed ramp/mature streams, batch 1/2/4/8,
row positions, speakers, languages).

### (c) What I would do instead

Make the scheduler cost-aware rather than "always pad to max". Build a measured
table `T(B, W)` and merge two shape groups only when
`T(B1+B2, max(W1,W2)) < T(B1,W1) + T(B2,W2)`, respecting each stream's deadline.

Simpler version worth benchmarking: three canonical buckets 8 / 16 / 24, mapping
1,3,7 -> 8; 15 -> 16; 23,24 -> 24. That destroys most of the six-way fragmentation
without forcing every transient to pay the 24-frame cost.

Benchmark four policies at 20 req/s and sweep load upward: current exact-shape
batching, {8,16,24} bucketing, max-width-per-collection, fixed-24 padding. Compare
p50/p95/p99 vocoder queue delay, TTFA, underrun rate, GPU occupied time, total
decode GPU-ms/sec, effective batch size, emitted audio seconds per GPU-second.

## 我的处置

**采纳 (a)3 并改设计。** 它这条是对的,而且用的是我自己的数据:一行成熟的
24 帧流会把同批的 ramp 行全部拖到 24 帧,B=8 时 23.3ms 对 7.6ms。原方案
"永远补到批内最宽"在这种混合批上可能比现状还慢。改为把补齐策略做成可选的
四档,用实测选,而不是先验断言:`exact`(现状)、`bucket`(8/16/24)、
`max`(批内最宽)、`fixed24`。判据取每秒解码 GPU 毫秒与 underrun,不取
launch 次数。

**采纳 (a)1 的受控实验要求。** 现有实验里补齐同时改变了张量形状,与 kernel
选择混淆。补一个把 batch size、行位置、执行次数全部固定,只改"有效长度 /
补齐区"的对照,并报 p95/p99/max,按窗口宽度和 chunk 边界分层。

**采纳 (b) 的接缝分析与频域指标。** 加多分辨率 STFT / log-mel 误差,以及
每个发射边界 ±20-50ms 的接缝局部不连续度(波形一阶差分、能量跳变)。

**无法执行的部分,如实记录:** ASR WER 需要 Deepgram key,本机没有;盲测
听音需要人类被试。这两项标注为未完成,不当作已通过。
