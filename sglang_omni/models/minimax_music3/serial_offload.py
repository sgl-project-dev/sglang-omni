# SPDX-License-Identifier: Apache-2.0
"""Serial GPU residency for the MiniMax Music 3 AR and DIT/DAV stages.

``--stage-offload-components ar,dit`` colocates both stages in one process
and lets only one of them hold GPU weights at a time: AR generates a whole
request, hands off to DIT/DAV, and stays off the GPU until DIT/DAV is finished
with that request.

The handoff has one request owner. A duplicated, out-of-order, or late terminal
event cannot wake AR while DIT/DAV is decoding another request.

A handoff that never ends stops admission for good. That is reported rather
than force-corrected:
waking AR while DIT/DAV still holds the GPU is how this configuration runs out
of memory, so a stuck server is preferable to a crashed one, and the log names
the requests still outstanding.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from typing import Any

import torch

logger = logging.getLogger(__name__)

STALL_REPORT_SECONDS = 900.0
_GIB = 1024.0**3


def _owned_tensors(
    modules: dict[str, Any],
) -> list[tuple[tuple[str, str], torch.Tensor]]:
    """Every tensor owned by a stage's modules, tied tensors listed once.

    ``named_parameters``/``named_buffers`` de-duplicate by default, so a tied
    weight appears under one name per module. De-duplicate once more across the
    group so modules that share a tensor still get one canonical host copy.
    Residency mutates the tensor object in place, so every name that shares it
    follows along and the tying survives the round trip.
    """
    owned: list[tuple[tuple[str, str], torch.Tensor]] = []
    seen: set[int] = set()
    for module_name, module in modules.items():
        for tensor_name, tensor in (
            *module.named_parameters(),
            *module.named_buffers(),
        ):
            if id(tensor) in seen:
                continue
            seen.add(id(tensor))
            owned.append(((module_name, tensor_name), tensor))
    return owned


def _gpu_memory_note(device: torch.device) -> str:
    """Device occupancy, for attributing what a handoff actually frees.

    ``torch_*`` covers only this process's caching allocator, while ``free``
    is the whole device, so a gap between them is memory held outside torch's
    allocator. Reporting both is what distinguishes weights coming off the GPU
    from the pools that stay put -- notably the SGLang KV cache, which
    ``mem_fraction_static`` reserves for the life of the process and which no
    weight offload releases.
    """
    if device.type != "cuda":
        return "cpu"
    free, total = torch.cuda.mem_get_info(device)
    return (
        f"free={free / _GIB:.2f}/{total / _GIB:.2f}GiB "
        f"torch_allocated={torch.cuda.memory_allocated(device) / _GIB:.2f}GiB "
        f"torch_reserved={torch.cuda.memory_reserved(device) / _GIB:.2f}GiB"
    )


class StageResidency:
    """GPU residency for a group of stage modules whose weights never change.

    Inference never writes these weights, so the host copy is canonical: it is
    filled once and every wake refills the GPU from it. Sleeping then only
    drops the GPU replica -- no device-to-host copy and no fresh host
    allocation per request, which is what ``module.to()`` in both directions
    was paying for on every single handoff.

    Reusing one host copy also stops the sleep path handing the caching
    allocator a differently sized block each round trip. That is what lets a
    long-running server fragment its way into an OOM after tens of requests
    rather than failing on the first one.
    """

    def __init__(
        self,
        modules: dict[str, Any],
        device: torch.device,
        *,
        resident: bool = True,
        label: str = "stage",
    ):
        if not modules:
            raise ValueError("StageResidency requires at least one module")
        self._modules = dict(modules)
        self._device = torch.device(device)
        self._resident = bool(resident)
        self._label = label
        self._host: dict[tuple[str, str], torch.Tensor] = {}
        if not self._resident:
            # Built on the host: retain its existing storage through a distinct
            # tensor handle. Keeping the Parameter itself here would alias the
            # object whose ``.data`` wake replaces, moving the supposed host
            # copy onto the GPU along with the module.
            self._host = {
                name: tensor.detach() for name, tensor in _owned_tensors(self._modules)
            }
        weight_bytes = sum(
            tensor.numel() * tensor.element_size()
            for _, tensor in _owned_tensors(self._modules)
        )
        logger.info(
            f"MiniMax Music 3 residency {label}: {weight_bytes / _GIB:.2f}GiB of "
            f"weights, starts {'resident' if self._resident else 'offloaded'} "
            f"({_gpu_memory_note(self._device)})"
        )

    @property
    def resident(self) -> bool:
        return self._resident

    def sleep(self) -> None:
        """Drop the GPU replica; a cheap no-op once already asleep."""
        if not self._resident:
            return
        if self._device.type == "cuda":
            torch.cuda.synchronize(self._device)
        owned = _owned_tensors(self._modules)
        if not self._host:
            self._host = {
                name: tensor.detach().to("cpu", copy=True) for name, tensor in owned
            }
        for name, tensor in owned:
            tensor.data = self._host[name]
        self._resident = False
        logger.info(
            f"MiniMax Music 3 residency {self._label} -> host "
            f"({_gpu_memory_note(self._device)})"
        )

    def wake(self) -> None:
        """Refill the GPU replica from the host copy; no-op once resident."""
        if self._resident:
            return
        for name, tensor in _owned_tensors(self._modules):
            tensor.data = self._host[name].to(
                self._device, copy=True, non_blocking=True
            )
        if self._device.type == "cuda":
            torch.cuda.synchronize(self._device)
        self._resident = True
        logger.info(
            f"MiniMax Music 3 residency {self._label} -> gpu "
            f"({_gpu_memory_note(self._device)})"
        )


class SerialOffloadCoordinator:
    """Process-wide ownership and optional CUDA residency coordinator."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._enabled = False
        self._ar: StageResidency | None = None
        self._owner: str | None = None
        self._phase = "disabled"
        self._failure: BaseException | None = None
        self._paused_at: float | None = None
        self._stall_reported = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    def enable(self) -> None:
        """Enable request ownership for a backend without CUDA residency."""
        with self._condition:
            if self._enabled:
                return
            self._enabled = True
            self._phase = "idle"

    def register_ar(self, model: Any, device: torch.device) -> None:
        residency = StageResidency({"ar": model}, device, label="ar")
        with self._condition:
            if self._enabled:
                raise RuntimeError("MiniMax Music 3 serial offload is already enabled")
            self._failure = None
            self._owner = None
            self._ar = residency
            self._enabled = True
            self._phase = "idle"
        logger.info(
            f"MiniMax Music 3 serial offload enabled device={device}; AR "
            "starts GPU-resident, DIT/DAV starts offloaded"
        )

    def try_acquire_ar(self, request_id: str) -> bool:
        """Atomically claim the complete AR-to-acoustic request lifecycle."""
        if not self._enabled:
            return True
        with self._condition:
            self._raise_if_failed_locked()
            if self._owner == request_id and self._phase == "ar":
                return True
            if self._owner is None and self._phase == "idle":
                self._owner = request_id
                self._phase = "ar"
                return True
            self._report_stall_locked()
            return False

    def acquire_ar(
        self,
        request_id: str,
        *,
        should_abort: Callable[[], bool] | None = None,
    ) -> None:
        """Wait for and claim AR ownership without holding the lock during work."""
        if not self._enabled:
            return
        with self._condition:
            while not self.try_acquire_ar(request_id):
                if should_abort is not None and should_abort():
                    raise InterruptedError(
                        "MiniMax Music 3 serial offload admission aborted"
                    )
                self._condition.wait(timeout=0.1)

    def begin_dit_handoff(self, request_id: str) -> None:
        """Hand the GPU to DIT/DAV for *request_id* and take AR off it."""
        if not self._enabled:
            return
        with self._condition:
            self._raise_if_failed_locked()
            if self._owner != request_id:
                raise RuntimeError(
                    "MiniMax Music 3 serial offload handoff does not own request "
                    f"{request_id!r} (owner={self._owner!r})"
                )
            if self._phase == "acoustic":
                return
            if self._phase != "ar":
                raise RuntimeError(
                    f"MiniMax Music 3 cannot hand off from phase {self._phase!r}"
                )
            self._phase = "ar_unloading"

        try:
            if self._ar is not None:
                self._ar.sleep()
        except BaseException as exc:
            self._fail_closed(exc)
            raise

        with self._condition:
            self._phase = "acoustic"
            self._paused_at = time.monotonic()
            self._stall_reported = False
            self._condition.notify_all()
        logger.info(
            f"MiniMax Music 3 serial offload: AR -> CPU (DIT/DAV's turn, "
            f"request={request_id})"
        )

    def require_acoustic(self, request_id: str) -> None:
        """Reject compute that is not owned by the active acoustic request."""
        if not self._enabled:
            return
        with self._condition:
            self._raise_if_failed_locked()
            if self._owner != request_id or self._phase != "acoustic":
                raise RuntimeError(
                    "MiniMax Music 3 acoustic request does not own residency: "
                    f"request={request_id!r} owner={self._owner!r} "
                    f"phase={self._phase!r}"
                )

    def end_dit_handoff(
        self,
        request_id: str,
        *,
        release_acoustic: Callable[[], None] | None = None,
    ) -> None:
        """Retire *request_id* and return residency to AR.

        Unknown and duplicate terminal events are no-ops. Acoustic release and
        AR wake run without the ownership mutex held.
        """
        if not self._enabled:
            return
        with self._condition:
            self._raise_if_failed_locked()
            if self._owner != request_id or self._phase != "acoustic":
                return
            self._phase = "acoustic_unloading"

        try:
            if release_acoustic is not None:
                release_acoustic()
            if self._ar is not None:
                self._ar.wake()
        except BaseException as exc:
            self._fail_closed(exc)
            raise

        with self._condition:
            self._owner = None
            self._phase = "idle"
            self._paused_at = None
            self._stall_reported = False
            self._condition.notify_all()
        logger.info(
            f"MiniMax Music 3 serial offload: AR -> GPU (AR's turn, "
            f"request={request_id})"
        )

    def cancel_ar(self, request_id: str) -> None:
        """Release an AR owner after its compute has stopped, before handoff."""
        if not self._enabled:
            return
        with self._condition:
            if self._owner != request_id or self._phase != "ar":
                return
            self._owner = None
            self._phase = "idle"
            self._condition.notify_all()

    def _fail_closed(self, exc: BaseException) -> None:
        with self._condition:
            self._failure = exc
            self._phase = "failed"
            self._condition.notify_all()

    def fail_closed(self, exc: BaseException) -> None:
        """Prevent later admission after an unsafe backend cleanup failure."""
        if self._enabled:
            self._fail_closed(exc)

    def _raise_if_failed_locked(self) -> None:
        if self._failure is not None:
            raise RuntimeError(
                "MiniMax Music 3 serial offload is unavailable after a residency "
                "transition failed"
            ) from self._failure

    def _report_stall_locked(self) -> None:
        """Name the owner once AR has been parked too long."""
        if self._paused_at is None or self._stall_reported:
            return
        elapsed = time.monotonic() - self._paused_at
        if elapsed < STALL_REPORT_SECONDS:
            return
        self._stall_reported = True
        logger.error(
            f"MiniMax Music 3 serial offload: AR has been off the GPU for "
            f"{elapsed:.0f}s and is still waiting on "
            f"{self._owner!r}; AR admits nothing until DIT/DAV "
            "retires them, so this server needs a restart if the requests "
            "are gone"
        )


_COORDINATOR = SerialOffloadCoordinator()


def get_coordinator() -> SerialOffloadCoordinator:
    return _COORDINATOR


def reset_coordinator() -> None:
    """Reset process-local state after a failed build or during test teardown."""
    global _COORDINATOR
    _COORDINATOR = SerialOffloadCoordinator()


__all__ = [
    "STALL_REPORT_SECONDS",
    "SerialOffloadCoordinator",
    "StageResidency",
    "get_coordinator",
    "reset_coordinator",
]
