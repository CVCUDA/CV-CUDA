# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Deterministic reproductions of cross-library stream contract violations in
the cvcuda Python bindings.

Covered bugs:

1. ``test_cuda_export_populates_stream_field`` — CAI (CUDA Array Interface)
   export must emit v3 with a populated ``stream`` key.  Pre-fix cvcuda
   emits v2 with no stream field.  Contract assertion, no timing
   dependency.

2. ``test_dlpack_export_honors_consumer_stream`` — ``__dlpack__(stream=X)``
   must synchronize cvcuda's writer stream into the consumer's stream X
   before returning the capsule (DLPack v1 spec).  Pre-fix cvcuda
   accepts the kwarg but ignores it.  Deterministic via a GPU-side
   busy-wait kernel on the producer stream: post-fix, consumer stream
   transitively waits for the kernel to drain; pre-fix, it does not.

Both tests finish in well under a second on a single-GPU host.  The
corresponding input-side race (cvcuda ignoring the producer's CAI
stream) is timing-sensitive and covered separately by
``test_multi_stream.py::test_wait_stream_fanout`` on multi-GPU CI.
"""

import time

import cupy
import numpy as np


# ----------------------------------------------------------------------
# GPU-side busy-wait kernel.  Used to hold a stream in a measurable
# non-idle state without relying on host callbacks (which deadlock
# against pytest's GIL).  clock64() reads the SM clock; cycle counts
# are chosen to be safely above the assertion threshold on modern GPUs
# (~1 GHz SM clock: 2e8 cycles ≈ 200 ms).  Even on very fast GPUs the
# wait will stay comfortably above the 40 ms threshold the tests check.
# ----------------------------------------------------------------------
_DELAY_KERNEL = cupy.RawKernel(
    r"""
extern "C" __global__ void cvcuda_test_busy_wait(unsigned long long cycles) {
    unsigned long long start = clock64();
    while (clock64() - start < cycles) { /* spin */ }
}
""",
    "cvcuda_test_busy_wait",
)


def _enqueue_gpu_delay(stream_handle: int, cycles: int = 200_000_000) -> None:
    """Enqueue a busy-wait on the given CUDA stream.

    No host callbacks: the spin runs on a GPU SM, so pytest's GIL is
    never involved.  The stream does not complete work after this point
    until the kernel retires (~100-200 ms on recent GPUs).
    """
    ext = cupy.cuda.ExternalStream(stream_handle)
    with ext:
        _DELAY_KERNEL((1,), (1,), (cycles,))


def test_cuda_export_populates_stream_field():
    """Tensor.cuda() must emit CAI v3 with the writer stream in `stream`.

    Pre-fix cvcuda emits CAI v2 with no ``stream`` field, so both
    assertions trip deterministically.  Post-fix the version bumps to 3
    and the stream key reports the cvcuda stream the data was last
    written on.
    """
    import cvcuda

    host = np.ones((1, 8, 8, 3), dtype=np.uint8) * 99
    src = cupy.asarray(host)
    inp = cvcuda.as_tensor(src, "NHWC")

    s = cvcuda.Stream()
    with s:
        out = cvcuda.flip(inp, -1, stream=s)

    cai = out.cuda().__cuda_array_interface__

    s.sync()

    assert cai.get("version") == 3, (
        f"expected CAI version 3 on export, got {cai.get('version')!r} "
        "(pre-fix cvcuda emits v2 with no stream field)"
    )
    assert (
        "stream" in cai
    ), f"expected 'stream' key in exported CAI dict, got keys {sorted(cai)!r}"
    assert cai["stream"] == int(
        s.handle
    ), f"expected stream={int(s.handle)} (the writer cvcuda.Stream), got {cai['stream']!r}"


def test_dlpack_export_honors_consumer_stream():
    """``__dlpack__(stream=X)`` must synchronize cvcuda's writer stream into X.

    Setup: run a cvcuda op on a cvcuda.Stream S1, then enqueue a
    ~100-200 ms GPU busy-wait on the same stream, so S1 has substantial
    pending work at the moment the consumer asks for a DLPack capsule.

    Act: call ``out.cuda().__dlpack__(stream=S2.ptr)`` on a different
    cupy stream S2, then synchronize S2 while timing it.

    Contract (DLPack v1): the producer must arrange for work queued
    on S2 after this call to wait until the tensor is ready on S1.
    The cheapest compliant implementation is ``cudaEventRecord`` on S1
    + ``cudaStreamWaitEvent`` on S2 before returning the capsule.

    Expected:

    * Pre-fix: cvcuda ignores ``stream=``, no wait is queued on S2,
      ``S2.synchronize()`` returns immediately — ``elapsed`` well under
      the delay kernel's runtime.  Assertion trips.
    * Post-fix: cvcuda queues the wait; ``S2.synchronize()`` blocks
      until S1 drains — ``elapsed`` is at least the delay-kernel
      runtime.  Assertion passes.

    The 40 ms threshold gives plenty of headroom both ways: the spin
    kernel is sized for ~100-200 ms, and pre-fix elapsed is typically
    under 2 ms (pure Python/capsule overhead).
    """
    import cvcuda

    N, H, W, C = 1, 8, 8, 3
    host = np.ones((N, H, W, C), dtype=np.uint8) * 42
    src = cupy.asarray(host)
    inp = cvcuda.as_tensor(src, "NHWC")

    cvcuda_stream = cvcuda.Stream()
    with cvcuda_stream:
        out = cvcuda.flip(inp, -1, stream=cvcuda_stream)

    # Hold S1 non-idle for ~100-200 ms via GPU busy-wait.
    _enqueue_gpu_delay(int(cvcuda_stream.handle))

    consumer_stream = cupy.cuda.Stream(non_blocking=True)

    # Request the DLPack capsule on the consumer's stream.  Post-fix
    # this call inserts cudaStreamWaitEvent(consumer_stream, <evt on
    # cvcuda_stream>) so consumer_stream transitively waits for the
    # delay kernel.
    t0 = time.perf_counter()
    cap = out.cuda().__dlpack__(stream=int(consumer_stream.ptr))
    consumer_stream.synchronize()
    elapsed = time.perf_counter() - t0

    # Cleanup: drain the producer stream so the spin kernel doesn't
    # linger into the next test.
    cvcuda_stream.sync()

    # Keep the capsule reference alive until after measurement so the
    # DLManagedTensor deleter is not invoked mid-timing.
    del cap

    assert elapsed > 0.040, (
        f"consumer stream synchronize took only {elapsed * 1000:.2f} ms; "
        "expected > 40 ms because the producer stream has a ~100-200 ms "
        "busy-wait kernel queued.  This indicates __dlpack__(stream=...) "
        "did NOT insert a cross-stream wait — the producer's pending "
        "work is invisible to the consumer (DLPack v1 stream contract "
        "violation)."
    )
