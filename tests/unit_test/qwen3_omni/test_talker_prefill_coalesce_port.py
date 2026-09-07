# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
import json
import unittest
from types import SimpleNamespace as NS

import pytest
from sglang.srt.utils import hf_transformers_utils

import sglang_omni.models.qwen3_omni.bootstrap as qwen_bootstrap
import sglang_omni.models.qwen3_omni.request_builders as qwen_request_builders
import sglang_omni.models.qwen3_omni.stages as qwen_stages
import sglang_omni.models.qwen3_omni.talker_model_runner as talker_model_runner_module
import sglang_omni.models.qwen3_omni.talker_scheduler as talker_scheduler_module
import sglang_omni.scheduling.bootstrap as scheduling_bootstrap_module
import sglang_omni.scheduling.omni_scheduler as omni_scheduler_module
import sglang_omni.scheduling.sglang_backend as sglang_backend_module
from sglang_omni.platforms import current_platform


def _mode(extend=True):
    return NS(is_extend=lambda: extend)


def _scheduler_types(monkeypatch: pytest.MonkeyPatch):
    clock = NS(now=100.0)
    calls = []

    def upstream(scheduler, running):
        calls.append(tuple(scheduler.waiting_queue))
        admitted = scheduler.waiting_queue[: scheduler.admit_limit]
        del scheduler.waiting_queue[: len(admitted)]
        return NS(
            batch_to_run=NS(reqs=admitted, forward_mode=_mode()),
            running_batch=running,
        )

    monkeypatch.setattr(omni_scheduler_module.time, "perf_counter", lambda: clock.now)
    monkeypatch.setattr(
        omni_scheduler_module._Upstream,
        "get_new_batch_prefill",
        upstream,
    )
    monkeypatch.setattr(
        omni_scheduler_module.OmniScheduler,
        "_admin_model_info",
        lambda self: {"success": True, "data": {"existing": 7}},
    )

    def make(target=4, wait=40.0, idle=False):
        scheduler = object.__new__(talker_scheduler_module.QwenTalkerScheduler)
        scheduler.prefill_coalesce_requests = target
        scheduler.prefill_coalesce_wait_s = wait / 1000
        scheduler.prefill_coalesce_when_idle = idle
        scheduler.prefill_coalesce_requires_pending_builds = False
        scheduler.prefill_coalesce_after_builds_during_decode = False
        scheduler.chunked_req = None
        scheduler.waiting_queue = []
        scheduler.admit_limit = 99
        return scheduler

    return make, clock, calls


class GateTests(unittest.TestCase):
    def setUp(self):
        self.monkeypatch = pytest.MonkeyPatch()
        self.addCleanup(self.monkeypatch.undo)
        self.make, self.clock, self.calls = _scheduler_types(self.monkeypatch)
        self.running = NS(is_empty=lambda: False)

    def test_disabled_has_no_hold(self):
        scheduler = self.make(target=0)
        scheduler.waiting_queue = [NS(_coalesce_enqueue_t=100.0)]
        self.assertEqual(
            len(scheduler.get_new_batch_prefill(self.running).batch_to_run.reqs),
            1,
        )

    def test_target_releases_same_order(self):
        scheduler = self.make()
        rows = [NS(rid=str(i), _coalesce_enqueue_t=100.0) for i in range(4)]
        scheduler.waiting_queue = rows.copy()
        self.assertEqual(
            scheduler.get_new_batch_prefill(self.running).batch_to_run.reqs,
            rows,
        )
        self.assertEqual(scheduler._prefill_batch_histogram, {4: 1})

    def test_timeout_does_not_restart_and_fifo_is_preserved(self):
        scheduler = self.make()
        first = NS(rid="a")
        second = NS(rid="b", _coalesce_enqueue_t=100.01)
        scheduler.waiting_queue = [first, second]
        self.assertIsNone(scheduler.get_new_batch_prefill(self.running).batch_to_run)
        self.clock.now = 100.039
        self.assertIsNone(scheduler.get_new_batch_prefill(self.running).batch_to_run)
        self.assertEqual(first._coalesce_enqueue_t, 100.0)
        self.clock.now = 100.041
        self.assertEqual(
            scheduler.get_new_batch_prefill(self.running).batch_to_run.reqs,
            [first, second],
        )
        self.assertEqual(scheduler._prefill_batch_histogram, {2: 1})

    def test_partial_admission_preserves_leftover_deadline(self):
        scheduler = self.make()
        rows = [NS(rid=str(i), _coalesce_enqueue_t=99.0) for i in range(3)]
        scheduler.waiting_queue = rows.copy()
        scheduler.admit_limit = 1
        self.assertEqual(
            scheduler.get_new_batch_prefill(self.running).batch_to_run.reqs,
            rows[:1],
        )
        self.assertEqual(
            scheduler.get_new_batch_prefill(self.running).batch_to_run.reqs,
            rows[1:2],
        )
        self.assertEqual(scheduler.waiting_queue[0]._coalesce_enqueue_t, 99.0)

    def test_aborted_oldest_does_not_release_newcomer_early(self):
        scheduler = self.make()
        old = NS(_coalesce_enqueue_t=99.0)
        new = NS(_coalesce_enqueue_t=100.0)
        scheduler.waiting_queue = [old, new]
        scheduler.waiting_queue.remove(old)
        self.assertIsNone(scheduler.get_new_batch_prefill(self.running).batch_to_run)

    def test_idle_and_chunked_prefill_bypass(self):
        for running, chunked in [
            (None, None),
            (NS(is_empty=lambda: True), None),
            (self.running, object()),
        ]:
            with self.subTest(running=running, chunked=chunked):
                scheduler = self.make()
                scheduler.chunked_req = chunked
                scheduler.waiting_queue = [NS(_coalesce_enqueue_t=100.0)]
                self.assertIsNotNone(
                    scheduler.get_new_batch_prefill(running).batch_to_run
                )

    def test_counter_ignores_decode_and_empty_plans(self):
        scheduler = self.make()
        plan = NS(batch_to_run=NS(reqs=[1], forward_mode=_mode(False)))
        self.monkeypatch.setattr(
            omni_scheduler_module.OmniScheduler,
            "get_new_batch_prefill",
            lambda self, running: plan,
        )
        self.assertIs(scheduler.get_new_batch_prefill(None), plan)
        self.assertFalse(hasattr(scheduler, "_prefill_batch_histogram"))
        plan.batch_to_run = None
        self.assertIs(scheduler.get_new_batch_prefill(None), plan)
        self.assertFalse(hasattr(scheduler, "_prefill_batch_histogram"))

    def test_admin_policy_is_detached_and_serializable(self):
        scheduler = self.make()
        scheduler._prefill_batch_histogram = {1: 3, 4: 2}
        reply = scheduler._admin_model_info()
        self.assertEqual(reply["data"]["existing"], 7)
        info = reply["data"]["talker_prefill_batching"]
        self.assertEqual(
            info,
            {
                "target_requests": 4,
                "max_wait_ms": 40.0,
                "when_idle": False,
                "batch_histogram": {"1": 3, "4": 2},
            },
        )
        self.assertEqual(json.loads(json.dumps(info)), info)
        import msgpack

        self.assertEqual(
            msgpack.unpackb(msgpack.packb(info), strict_map_key=True),
            info,
        )
        info["batch_histogram"].clear()
        self.assertEqual(scheduler._prefill_batch_histogram, {1: 3, 4: 2})


class FactoryTests(unittest.TestCase):
    def test_complete_stage_and_bootstrap_forward_to_scheduler_only(self):
        for target in [0, 4]:
            with self.subTest(target=target):
                self._run_factory(target)

    def test_lookahead_with_cache_and_batching_binds_only_the_talker_abort_predicate(
        self,
    ):
        for enabled in [False, True]:
            with self.subTest(enabled=enabled):
                self._run_factory(4, lookahead=enabled)

    def test_scratch_option_binds_before_capture_on_identical_initialization_path(self):
        for skip_scratch in [False, True]:
            for graph_enabled in [False, True]:
                with self.subTest(
                    skip_scratch=skip_scratch,
                    graph_enabled=graph_enabled,
                ):
                    self._run_factory(
                        4,
                        lookahead=True,
                        skip_scratch=skip_scratch,
                        graph_enabled=graph_enabled,
                    )

    def _run_factory(
        self,
        target,
        lookahead=None,
        skip_scratch=None,
        graph_enabled=False,
    ):
        observed = {}
        config = NS(
            model_path="test-model",
            hf_config=NS(
                thinker_config=NS(
                    audio_token_id=1,
                    image_token_id=2,
                    video_token_id=3,
                ),
                talker_config=NS(
                    text_config=NS(vocab_size=4096),
                    accept_hidden_layer=4,
                    codec_bos_id=5,
                    codec_eos_token_id=6,
                    codec_nothink_id=7,
                    codec_think_bos_id=8,
                    codec_think_eos_id=9,
                    codec_pad_id=10,
                    speaker_id={},
                ),
                tts_bos_token_id=11,
                tts_eos_token_id=12,
                tts_pad_token_id=13,
                im_start_token_id=14,
                im_end_token_id=15,
                system_token_id=16,
                user_token_id=17,
                assistant_token_id=18,
            ),
        )
        phases = []
        model = NS()

        def configure_scratch(*, skip_unused):
            phases.append("configure")
            observed["skip_scratch"] = skip_unused

        model.configure_predictor_scratch_writes = configure_scratch
        worker = NS(
            model_runner=NS(
                model_config=config,
                model=model,
                sampler=object(),
            )
        )

        def infrastructure(*args, **kwargs):
            phases.append("infrastructure")
            self.assertIs(
                kwargs["defer_cuda_graph_capture"],
                graph_enabled,
            )
            kwargs["model_post_load_hook"](model)
            if not graph_enabled:
                init_graphs(worker)
            return worker, None, None, None, config

        def init_graphs(value):
            self.assertIs(value, worker)
            self.assertIn("skip_scratch", observed)
            self.assertEqual(hasattr(model, "_sampler"), graph_enabled)
            phases.append("init")

        def prefill_builder(**kwargs):
            observed["builder"] = kwargs
            return NS(
                append_text_chunk=lambda *args, **kwargs: None,
                mark_thinker_done=lambda *args, **kwargs: None,
            )

        class FakeScheduler:
            def __init__(self, **kwargs):
                observed["scheduler"] = kwargs
                self.outbox = object()
                observed["scheduler_instance"] = self

            def is_request_aborted(self, request_id):
                return request_id == "cancelled"

            def bind_model_runner(self, runner):
                observed["runner"] = runner

        def model_runner(*args, **kwargs):
            return NS(args=args, **kwargs)

        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr(
                hf_transformers_utils,
                "get_tokenizer",
                lambda *args, **kwargs: object(),
            )
            monkeypatch.setattr(
                current_platform,
                "enable_talker_graph",
                lambda: True,
            )
            monkeypatch.setattr(
                qwen_request_builders,
                "TalkerPrefillBuilder",
                prefill_builder,
            )
            monkeypatch.setattr(
                talker_model_runner_module,
                "QwenTalkerModelRunner",
                model_runner,
            )
            monkeypatch.setattr(
                talker_scheduler_module,
                "QwenTalkerScheduler",
                FakeScheduler,
            )
            monkeypatch.setattr(
                talker_scheduler_module,
                "configure_talker_server_args",
                lambda *args, **kwargs: graph_enabled,
            )
            monkeypatch.setattr(
                scheduling_bootstrap_module,
                "create_sglang_infrastructure",
                infrastructure,
            )
            monkeypatch.setattr(
                scheduling_bootstrap_module,
                "init_sglang_cuda_graphs",
                init_graphs,
            )
            monkeypatch.setattr(
                sglang_backend_module,
                "SGLangOutputProcessor",
                lambda **kwargs: NS(**kwargs),
            )
            monkeypatch.setattr(
                qwen_stages,
                "build_generation_batch_overrides",
                lambda **kwargs: {},
            )
            monkeypatch.setattr(
                qwen_stages,
                "_apply_colocated_ar_memory_contract",
                lambda *args, **kwargs: None,
            )
            monkeypatch.setattr(
                qwen_stages,
                "build_sglang_server_args",
                lambda *args, **kwargs: NS(mem_fraction_static=0.123),
            )
            monkeypatch.setattr(
                qwen_stages,
                "validate_generation_batch_policy",
                lambda **kwargs: None,
            )
            monkeypatch.setattr(qwen_stages, "avail_gpu_mem", lambda *args: 1)
            monkeypatch.setattr(
                qwen_stages,
                "get_process_gpu_memory_bytes",
                lambda *args: 0,
            )

            options = (
                {}
                if target == 0
                else {
                    "prefill_coalesce_requests": 4,
                    "prefill_coalesce_wait_ms": 40.0,
                    "prefill_coalesce_when_idle": False,
                }
            )
            for factory in [
                qwen_stages.create_talker_ar_executor_from_config,
                qwen_bootstrap.create_talker_scheduler,
            ]:
                self.assertIs(
                    inspect.signature(factory)
                    .parameters["code_predictor_skip_scratch_writes"]
                    .default,
                    False,
                )
                self.assertIs(
                    inspect.signature(factory)
                    .parameters["enable_async_decode"]
                    .default,
                    False,
                )
                self.assertEqual(
                    inspect.signature(factory)
                    .parameters["async_decode_min_batch_size"]
                    .default,
                    2,
                )
                self.assertEqual(
                    inspect.signature(factory)
                    .parameters["assistant_projection_cache_size"]
                    .default,
                    0,
                )
            self.assertEqual(
                inspect.signature(qwen_request_builders.make_talker_scheduler_adapters)
                .parameters["assistant_projection_cache_size"]
                .default,
                0,
            )
            if lookahead is not None:
                options.update(
                    enable_async_decode=lookahead,
                    async_decode_min_batch_size=3,
                )
            if skip_scratch is not None:
                options["code_predictor_skip_scratch_writes"] = skip_scratch
            qwen_stages.create_talker_ar_executor_from_config(
                "test-model",
                assistant_projection_cache_size=4096,
                **options,
            )

        self.assertEqual(phases, ["infrastructure", "configure", "init"])
        self.assertIs(observed["skip_scratch"], bool(skip_scratch))
        self.assertIs(
            observed["scheduler"]["enable_async_decode"],
            bool(lookahead),
        )
        self.assertEqual(
            observed["scheduler"]["async_decode_min_batch_size"],
            2 if lookahead is None else 3,
        )
        predicate = observed["runner"].request_is_aborted
        if lookahead:
            self.assertIs(
                predicate.__self__,
                observed["scheduler_instance"],
            )
            self.assertTrue(predicate("cancelled"))
            self.assertFalse(predicate("live"))
        else:
            self.assertIsNone(predicate)
        self.assertEqual(
            observed["scheduler"]["prefill_coalesce_requests"],
            target,
        )
        self.assertEqual(
            observed["scheduler"]["prefill_coalesce_wait_ms"],
            40.0,
        )
        self.assertFalse(observed["scheduler"]["prefill_coalesce_when_idle"])
        self.assertEqual(
            observed["builder"]["assistant_projection_cache_size"],
            4096,
        )
        self.assertFalse(
            any(key.startswith("prefill_coalesce") for key in observed["builder"])
        )
        self.assertNotIn(
            "prefill_coalesce_requests",
            observed["runner"].__dict__,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
