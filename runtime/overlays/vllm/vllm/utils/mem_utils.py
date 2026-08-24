# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib
import gc
import time
from collections.abc import Generator
from dataclasses import dataclass, field
from functools import cache

import psutil
import torch
import torch.types

from vllm.logger import init_logger
from vllm.platforms import current_platform

from .mem_constants import GiB_bytes, KiB_bytes, MiB_bytes

logger = init_logger(__name__)


def format_kib(b: int) -> str:
    return f"{round(b / KiB_bytes, 2)}"


def format_mib(b: int) -> str:
    return f"{round(b / MiB_bytes, 2)}"


def format_gib(b: int) -> str:
    return f"{round(b / GiB_bytes, 2)}"


@cache
def get_max_shared_memory_bytes(gpu: int = 0) -> int:
    """Returns the maximum shared memory per thread block in bytes."""
    from vllm import _custom_ops as ops

    max_shared_mem = ops.get_max_shared_memory_per_block_device_attribute(gpu)
    # value 0 will cause MAX_SEQ_LEN become negative and test_attention.py
    # will fail
    assert max_shared_mem > 0, "max_shared_mem cannot be zero"
    return int(max_shared_mem)


def get_cpu_memory() -> int:
    """Returns the total CPU memory of the node in bytes."""
    return psutil.virtual_memory().total


_UMA_PRESSURE_THRESHOLD = 0.8
_UMA_MIN_RELEASE_BYTES = 512 * MiB_bytes


def release_device_memory_under_pressure(device: torch.device) -> bool:
    """On integrated (UMA) GPUs, release caching-allocator memory back to the
    OS when system memory pressure is high. The OS may start thrashing before
    an allocation failure would trigger PyTorch's own cache release.

    Returns:
        True if memory was released.
    """
    if device.type != "cuda" or not current_platform.is_integrated_gpu(device.index):
        return False

    releasable = torch.accelerator.memory_reserved(
        device
    ) - torch.accelerator.memory_allocated(device)
    if releasable < _UMA_MIN_RELEASE_BYTES:
        return False

    # cudaMemGetInfo underreports free memory on UMA, see MemorySnapshot.measure
    mem = psutil.virtual_memory()
    if mem.available > (1 - _UMA_PRESSURE_THRESHOLD) * mem.total:
        return False

    torch.accelerator.synchronize(device)
    torch.accelerator.empty_cache()
    logger.debug(
        "Released %sGiB of cached device memory under memory pressure",
        format_gib(releasable),
    )
    return True


class DeviceMemoryProfiler:
    def __init__(self, device: torch.types.Device | None = None):
        self.device = device

    def current_memory_usage(self) -> float:
        # Return the memory usage in bytes.
        gc.collect()
        return current_platform.get_current_memory_usage(self.device)

    def __enter__(self):
        self.initial_memory = self.current_memory_usage()
        # This allows us to call methods of the context manager if needed
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.final_memory = self.current_memory_usage()
        self.consumed_memory = self.final_memory - self.initial_memory

        # Force garbage collection
        gc.collect()


@dataclass
class MemorySnapshot:
    """Memory snapshot."""

    torch_peak: int = 0
    torch_allocated: int = 0
    free_memory: int = 0
    total_memory: int = 0
    cuda_memory: int = 0
    torch_memory: int = 0
    non_torch_memory: int = 0
    timestamp: float = 0.0

    device: torch.types.Device = None
    auto_measure: bool = True

    def __post_init__(self) -> None:
        if self.device is None:
            device_fn = current_platform.current_device
            assert device_fn is not None
            self.device_ = torch.device(device_fn())
        else:
            self.device_ = torch.device(self.device)

        if self.auto_measure:
            self.measure()

    def measure(self) -> None:
        device = self.device_

        # we measure the torch peak memory usage via allocated_bytes,
        # rather than `torch.accelerator.memory_reserved()` .
        # After `torch.accelerator.reset_peak_memory_stats()`,
        # `torch.accelerator.memory_reserved()` will keep growing, and only shrink
        # when we call `torch.accelerator.empty_cache()` or OOM happens.
        stats = torch.accelerator.memory_stats(device)
        self.torch_peak = stats.get("allocated_bytes.all.peak", 0)
        self.torch_allocated = stats.get("allocated_bytes.all.current", 0)

        self.free_memory, self.total_memory = torch.accelerator.get_memory_info(device)
        if current_platform.is_integrated_gpu(device.index):
            # On UMA (Unified Memory Architecture) platforms where CPU and
            # GPU share physical memory (e.g. GH200, DGX Spark, Jetson Orin),
            # cudaMemGetInfo underreports free memory because it does not
            # account for reclaimable OS memory (page cache, buffers).
            # Use psutil to get the true available memory.
            # https://docs.nvidia.com/cuda/cuda-for-tegra-appnote/#estimating-total-allocatable-device-memory-on-an-integrated-gpu-device
            self.free_memory = psutil.virtual_memory().available

        self.cuda_memory = self.total_memory - self.free_memory

        # torch.accelerator.memory_reserved() is how many bytes
        # PyTorch gets from cuda (by calling cudaMalloc, etc.)
        # this is used to measure the non-torch memory usage
        self.torch_memory = torch.accelerator.memory_reserved(device)

        self.non_torch_memory = self.cuda_memory - self.torch_memory
        self.timestamp = time.time()

    def __sub__(self, other: "MemorySnapshot") -> "MemorySnapshot":
        if self.device_ != other.device_:
            raise ValueError(
                "The two snapshots should be from the same device! "
                f"Found: {self.device_} vs. {other.device_}"
            )

        return MemorySnapshot(
            torch_peak=self.torch_peak - other.torch_peak,
            torch_allocated=self.torch_allocated - other.torch_allocated,
            free_memory=self.free_memory - other.free_memory,
            total_memory=self.total_memory - other.total_memory,
            cuda_memory=self.cuda_memory - other.cuda_memory,
            torch_memory=self.torch_memory - other.torch_memory,
            non_torch_memory=self.non_torch_memory - other.non_torch_memory,
            timestamp=self.timestamp - other.timestamp,
            device=self.device_,
            auto_measure=False,
        )

    def __repr__(self) -> str:
        return (
            f"torch_peak={format_gib(self.torch_peak)}GiB, "
            f"torch_allocated={format_gib(self.torch_allocated)}GiB, "
            f"free_memory={format_gib(self.free_memory)}GiB, "
            f"total_memory={format_gib(self.total_memory)}GiB, "
            f"{current_platform.device_name}_memory={format_gib(self.cuda_memory)}GiB, "
            f"torch_memory={format_gib(self.torch_memory)}GiB, "
            f"non_torch_memory={format_gib(self.non_torch_memory)}GiB, "
            f"timestamp={self.timestamp}, "
            f"auto_measure={self.auto_measure}"
        )


@dataclass
class MemoryProfilingResult:
    """Memory profiling result. All numbers are in bytes."""

    non_kv_cache_memory: int = 0
    torch_peak_increase: int = 0
    non_torch_increase: int = 0
    total_consumed: int = 0
    transient_peak_headroom: int = 0
    weights_memory: int = 0
    before_create: MemorySnapshot = field(default_factory=MemorySnapshot)
    profile_time: float = 0.0

    def __post_init__(self) -> None:
        device = self.before_create.device_

        self.before_profile = MemorySnapshot(device=device, auto_measure=False)
        self.after_profile = MemorySnapshot(device=device, auto_measure=False)

    def __repr__(self) -> str:
        return (
            f"Memory profiling takes {self.profile_time:.2f} seconds. "
            f"Total non KV cache memory: "
            f"{format_gib(self.non_kv_cache_memory)}GiB; "
            f"torch peak memory increase: "
            f"{format_gib(self.torch_peak_increase)}GiB; "
            f"total consumed (from mem_get_info): "
            f"{format_gib(self.total_consumed)}GiB; "
            f"weights memory: {format_gib(self.weights_memory)}GiB."
        )


@contextlib.contextmanager
def memory_profiling(
    baseline_snapshot: MemorySnapshot,
    weights_memory: int = 0,
) -> Generator[MemoryProfilingResult, None, None]:
    """
    Memory profiling context manager.

    baseline_snapshot: the memory snapshot before the current vLLM instance.
    weights_memory: memory used by PyTorch when loading the model weights.
        Note that, before loading the model weights, we also initialize the device
        and distributed environment, which may consume some memory. This part is not
        included in the weights_memory because PyTorch does not control it.

    The memory in one GPU can be classified into 3 categories:
    1. memory used by anything other than the current vLLM instance.
    2. memory used by torch in the current vLLM instance.
    3. memory used in the current vLLM instance, but not by torch.

    A quantitive example:

    Before creating the current vLLM instance:
        category 1: 1 GiB
        category 2: 0 GiB
        category 3: 0 GiB

    After creating the current vLLM instance and loading the model,
    (i.e. before profiling):
        category 1: 1 GiB
        category 2: 2 GiB (model weights take 2 GiB)
        category 3: 0.5 GiB (memory used by NCCL)

    During profiling (peak):
        category 1: 1 GiB
        category 2: 4 GiB (peak activation tensors take 2 GiB)
        category 3: 1 GiB (memory used by NCCL + buffers for some attention backends)

    After profiling:
        category 1: 1 GiB
        category 2: 3 GiB (after garbage-collecting activation tensors)
        category 3: 1 GiB (memory used by NCCL + buffers for some attention backends)

    In this case, non-kv cache takes 5 GiB in total, including:
    a. 2 GiB used by the model weights (category 2)
    b. 2 GiB reserved for the peak activation tensors (category 2)
    c. 1 GiB used by non-torch components (category 3)

    The memory used for loading weights (a.) is directly given from the
    argument `weights_memory`.

    The increase of `torch.accelerator.memory_stats()["allocated_bytes.all.peak"]`
    during profiling gives (b.).

    The increase of `non_torch_memory` from creating the current vLLM instance
    until after profiling to get (c.).
    """
    gc.collect()
    torch.accelerator.empty_cache()
    torch.accelerator.reset_peak_memory_stats(baseline_snapshot.device_)

    result = MemoryProfilingResult(
        before_create=baseline_snapshot,
        # the part of memory used for holding the model weights
        weights_memory=weights_memory,
    )

    result.before_profile.measure()

    yield result

    gc.collect()
    torch.accelerator.empty_cache()

    result.after_profile.measure()

    diff_profile = result.after_profile - result.before_profile
    diff_from_create = result.after_profile - result.before_create
    result.torch_peak_increase = diff_profile.torch_peak
    result.non_torch_increase = diff_from_create.non_torch_memory
    result.profile_time = diff_profile.timestamp

    # Measure total consumption via mem_get_info() instead of
    # memory_reserved(), which goes negative when pluggable allocators
    # (e.g. cumem) bypass PyTorch's tracking.
    result.total_consumed = (
        result.before_create.free_memory - result.after_profile.free_memory
    )

    # total_consumed already covers persistent torch allocations; add only the
    # transient peak headroom to avoid double-counting.
    result.transient_peak_headroom = (
        result.after_profile.torch_peak - result.after_profile.torch_allocated
    )
    result.non_kv_cache_memory = result.total_consumed + result.transient_peak_headroom


class UnifiedMemoryBudgetError(RuntimeError):
    """Fail-closed allocator-budget failure on an integrated-GPU platform.

    Raised instead of allowing the PyTorch allocator to grow into the
    protected host reserve or past the phase budget. The original cause is
    chained where an allocator OOM was the trigger.
    """


@dataclass(frozen=True)
class UnifiedMemoryAllocatorReceipt:
    """Exact arithmetic receipt for one allocator-ceiling application.

    All byte fields are exact integers. `requested_ceiling_bytes` is the
    ideal ceiling E; `applied_ceiling_bytes` is the representable ceiling
    after float-fraction rounding, never larger than E.
    """

    phase: str  # "profile" | "kv" | "graphs" | "auto"
    integrated_gpu: bool
    total_cuda_bytes: int
    total_host_bytes: int
    pool_total_bytes: int
    available_bytes: int
    reserve_bytes: int
    current_allocated_bytes: int
    current_reserved_bytes: int
    current_baseline_bytes: int
    phase_budget_bytes: int
    requested_ceiling_bytes: int
    applied_fraction: float
    applied_ceiling_bytes: int


def compute_unified_memory_allocator_ceiling(
    *,
    phase: str,
    total_cuda_bytes: int,
    total_host_bytes: int,
    available_bytes: int,
    reserve_bytes: int,
    current_allocated_bytes: int,
    current_reserved_bytes: int,
    phase_budget_bytes: int | None = None,
) -> UnifiedMemoryAllocatorReceipt:
    """Compute the fail-closed PyTorch allocator ceiling for an integrated GPU.

    Exact checked-integer formula (vLLM PR #49760 busy-host correction):

        T = min(CUDA-visible total bytes, host MemTotal bytes)
        A = current host MemAvailable bytes
        R = configured absolute reserve bytes (already validated 0 < R < T)
        L = current torch allocated bytes
        S = current torch reserved bytes
        C = max(L, S)                      # existing allocator baseline
        B = phase absolute budget bytes    # None => T - R
        I = max(0, A - R)                  # safe incremental growth
        P = T - R                          # absolute shared-pool process ceiling
        E = min(P, C + I, B)               # requested absolute ceiling

    Invariants enforced here (all fail closed before torch is touched):
        R in (0, T); L <= C; S <= C; C <= E <= P; E - C <= max(0, A - R);
        E <= B when B is set. Any non-finite or negative value fails.
    """
    if phase not in ("load", "profile", "kv", "graphs", "auto"):
        raise UnifiedMemoryBudgetError(f"unknown allocator phase: {phase!r}")

    def _check_nonneg(name: str, value: int) -> int:
        if not isinstance(value, int) or isinstance(value, bool):
            raise UnifiedMemoryBudgetError(f"{name} must be an integer")
        if value < 0:
            raise UnifiedMemoryBudgetError(f"{name} must not be negative")
        if value > 2**63 - 1:
            raise UnifiedMemoryBudgetError(
                f"{name} exceeds signed 64-bit range")
        return value

    total_cuda_bytes = _check_nonneg("total_cuda_bytes", total_cuda_bytes)
    total_host_bytes = _check_nonneg("total_host_bytes", total_host_bytes)
    available_bytes = _check_nonneg("available_bytes", available_bytes)
    reserve_bytes = _check_nonneg("reserve_bytes", reserve_bytes)
    current_allocated_bytes = _check_nonneg("current_allocated_bytes",
                                            current_allocated_bytes)
    current_reserved_bytes = _check_nonneg("current_reserved_bytes",
                                           current_reserved_bytes)

    pool_total = min(total_cuda_bytes, total_host_bytes)
    if pool_total == 0:
        raise UnifiedMemoryBudgetError("pool total is zero; cannot size allocator")

    if not (0 < reserve_bytes < pool_total):
        raise UnifiedMemoryBudgetError(
            f"reserve {reserve_bytes} must be in (0, pool_total {pool_total})"
        )

    if available_bytes <= reserve_bytes:
        raise UnifiedMemoryBudgetError(
            "available host memory does not exceed the protected reserve; "
            "refusing to proceed"
        )

    baseline = max(current_allocated_bytes, current_reserved_bytes)
    if current_allocated_bytes > baseline or current_reserved_bytes > baseline:
        raise UnifiedMemoryBudgetError("allocator baseline arithmetic inconsistency")

    # Safe incremental growth: never grow into the protected reserve.
    incremental = max(0, available_bytes - reserve_bytes)
    # Absolute shared-pool process ceiling.
    pool_ceiling = pool_total - reserve_bytes
    if pool_ceiling <= 0:
        raise UnifiedMemoryBudgetError(
            "pool ceiling is non-positive after reserve; cannot apply ceiling"
        )

    if phase_budget_bytes is None:
        budget = pool_ceiling
    else:
        budget = _check_nonneg("phase_budget_bytes", phase_budget_bytes)

    # R17 b3: an EXPLICIT operator budget may raise the ceiling above the
    # conservative MemAvailable-derived allowance (which systematically
    # underestimates allocatable memory on GB10 UMA due to reclaimable page
    # cache), but never above pool_ceiling = total - reserve. Unset budget
    # preserves the stock conservative formula exactly.
    if phase_budget_bytes is None:
        requested = min(pool_ceiling, baseline + incremental, budget)
    else:
        requested = min(pool_ceiling, max(baseline + incremental, budget))

    if baseline > pool_ceiling:
        raise UnifiedMemoryBudgetError(
            "current allocator baseline exceeds total-minus-reserve; "
            "cannot cap below loaded residency"
        )
    if requested < baseline:
        raise UnifiedMemoryBudgetError(
            "requested ceiling fell below current allocator baseline"
        )
    if phase_budget_bytes is None and requested > budget:
        raise UnifiedMemoryBudgetError("requested ceiling exceeds phase budget")
    if phase_budget_bytes is None and requested - baseline > incremental:
        raise UnifiedMemoryBudgetError(
            "ceiling growth exceeds available-minus-reserve allowance"
        )

    return UnifiedMemoryAllocatorReceipt(
        phase=phase,
        integrated_gpu=True,
        total_cuda_bytes=total_cuda_bytes,
        total_host_bytes=total_host_bytes,
        pool_total_bytes=pool_total,
        available_bytes=available_bytes,
        reserve_bytes=reserve_bytes,
        current_allocated_bytes=current_allocated_bytes,
        current_reserved_bytes=current_reserved_bytes,
        current_baseline_bytes=baseline,
        phase_budget_bytes=budget,
        requested_ceiling_bytes=requested,
        applied_fraction=0.0,
        applied_ceiling_bytes=0,
    )


def apply_unified_memory_allocator_ceiling(
    receipt: UnifiedMemoryAllocatorReceipt,
    *,
    set_fraction: object,
) -> tuple[float, int]:
    """Convert the integer ceiling E into a representable float fraction.

    `set_fraction` is the callable used to apply the fraction
    (torch.cuda.set_per_process_memory_fraction in production; a fake in
    tests). Returns (applied_fraction, applied_bytes) with the guarantee
    `floor(fraction * pool_total) <= E` and `applied_bytes >= baseline`.
    """
    import math

    pool_total = receipt.pool_total_bytes
    requested = receipt.requested_ceiling_bytes
    baseline = receipt.current_baseline_bytes

    fraction = requested / pool_total
    # Round down until the representable product cannot exceed E.
    while math.floor(fraction * pool_total) > requested:
        fraction = math.nextafter(fraction, 0.0)
    applied_bytes = math.floor(fraction * pool_total)
    if applied_bytes < baseline:
        raise UnifiedMemoryBudgetError(
            "representable cap fell below current residency"
        )
    set_fraction(fraction)
    return fraction, applied_bytes
