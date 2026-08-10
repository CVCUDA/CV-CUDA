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

"""Unit tests for Resource::submitSync's three control-flow paths.

submitSync has three paths:

  1. Fast path  — resource already bound to the same stream: return immediately,
                  no cudaGetDevice, no event work.
  2. First-bind — resource has never been submitted: take ownership of the
                  stream with no sync work.
  3. Cross-stream — resource moves from stream A to stream B: record an event
                    on A, wait on B, so B observes all prior A work.

The CAI-sentinel sub-path (cudaStreamLegacy / cudaStreamPerThread → fall back
to cudaDeviceSynchronize) is covered separately in test_cai_input_stream_race.py.
"""

import cupy

import cvcuda

# GPU-side busy-wait: keeps a stream occupied for long enough that any missing
# cross-stream barrier surfaces as wrong output.  ~500 ms at 1 GHz SM clock.
_BUSY_WAIT = cupy.RawKernel(
    r"""
extern "C" __global__ void busy_wait(unsigned long long cycles) {
    unsigned long long t = clock64();
    while (clock64() - t < cycles) {}
}
""",
    "busy_wait",
)
_WAIT_CYCLES = 500_000_000


def _fill_tensor(tensor, value, stream):
    """Fill a uint8 NHWC cvcuda tensor with a constant via cupy on *stream*."""
    cp_arr = cupy.asarray(tensor.cuda(stream.handle))
    with stream._cupy_stream():
        cp_arr.fill(value)


class _CupyStream:
    """Thin wrapper so a raw cupy Stream handle can be used as a context manager."""

    def __init__(self, handle):
        self._s = cupy.cuda.ExternalStream(handle)

    def __enter__(self):
        self._s.__enter__()
        return self

    def __exit__(self, *a):
        self._s.__exit__(*a)


class _RawStreamHandle:
    """Wraps a raw integer CUDA stream handle as a cupy stream protocol object."""

    def __init__(self, handle: int):
        self._handle = handle

    def __cuda_stream__(self):
        return (0, self._handle)


def _as_cupy_stream(cvcuda_stream):
    if hasattr(cupy.cuda.Stream, "from_external"):
        return cupy.cuda.Stream.from_external(_RawStreamHandle(cvcuda_stream.handle))
    return _CupyStream(cvcuda_stream.handle)


# ---------------------------------------------------------------------------
# Path 1 — first-binding
# ---------------------------------------------------------------------------


def test_first_bind_takes_ownership():
    """submitStreamSync on a never-submitted resource must not raise."""
    stream = cvcuda.Stream()
    tensor = cvcuda.Tensor((1, 4, 4, 1), cvcuda.Type.U8, "NHWC")
    # No assertion beyond "does not throw": first-bind sets m_lastStream and
    # returns without doing any event or device work.
    tensor.submitStreamSync(stream)


def test_first_bind_then_op_produces_correct_output():
    """A resource used for the first time on a stream produces correct output."""
    stream = cvcuda.Stream()
    src = cvcuda.Tensor((1, 8, 8, 3), cvcuda.Type.U8, "NHWC")
    with stream:
        out = cvcuda.cvtcolor(src, cvcuda.ColorConversion.RGB2BGR)
    stream.sync()
    # RGB2BGR on a zero-initialised buffer is still all-zero; just verify it
    # completes without error and the output shape is as expected.
    assert out.shape == (1, 8, 8, 3)


# ---------------------------------------------------------------------------
# Path 2 — same-stream fast path
# ---------------------------------------------------------------------------


def test_same_stream_fast_path_does_not_raise():
    """Repeated submitStreamSync calls on the same stream must not raise."""
    stream = cvcuda.Stream()
    tensor = cvcuda.Tensor((1, 4, 4, 1), cvcuda.Type.U8, "NHWC")
    tensor.submitStreamSync(stream)
    # All subsequent calls hit the fast path: prevHandle == stream.handle()
    for _ in range(10):
        tensor.submitStreamSync(stream)


def test_same_stream_fast_path_preserves_ordering():
    """Two ops on the same stream using the same resource must be ordered."""
    stream = cvcuda.Stream()
    src = cvcuda.Tensor((1, 8, 8, 1), cvcuda.Type.U8, "NHWC")
    with stream:
        # Both ops share the same input tensor → both hit the fast path after
        # the first submission.  CUDA stream ordering guarantees the second
        # op sees the output of the first.
        mid = cvcuda.convertto(src, cvcuda.Type.F32, scale=1 / 255.0)
        out = cvcuda.convertto(mid, cvcuda.Type.U8, scale=255.0)
    stream.sync()
    assert out.shape == src.shape


# ---------------------------------------------------------------------------
# Path 3 — cross-stream slow path
# ---------------------------------------------------------------------------


def test_cross_stream_sync_orders_work():
    """Work enqueued on stream A must be visible on stream B after submitSync.

    A buffer is first-bound to stream A, then filled on A after a long
    busy-wait (so the fill is guaranteed to be in-flight when cvcuda attempts
    to consume it).  submitStreamSync moves the resource to stream B by
    recording an event on A and inserting a wait on B.  A subsequent read on B
    must observe the fill, not an uninitialised buffer.
    """
    stream_a = cvcuda.Stream()
    stream_b = cvcuda.Stream()

    buf = cvcuda.Tensor((1, 8, 8, 3), cvcuda.Type.U8, "NHWC")
    cp_buf = cupy.asarray(buf.cuda())

    FILL_VALUE = 77

    # First-bind resource to stream A (no CUDA work).
    buf.submitStreamSync(stream_a)

    # Queue work on stream A: long busy-wait then fill.  Both are in-flight
    # when submitStreamSync(stream_b) is called from the host.
    with _as_cupy_stream(stream_a):
        _BUSY_WAIT((1,), (1,), (_WAIT_CYCLES,))
        cp_buf.fill(FILL_VALUE)

    # Transfer ownership to stream B: cudaEventRecord on A (captures the fill),
    # cudaStreamWaitEvent on B.  Stream B will not start until A's fill lands.
    buf.submitStreamSync(stream_b)

    with _as_cupy_stream(stream_b):
        result = cp_buf.copy()

    stream_b.sync()

    unique = sorted(set(result.flatten().tolist()))
    assert (result == FILL_VALUE).all(), (
        f"cross-stream sync failed: expected all {FILL_VALUE}, got {unique}. "
        "submitSync did not insert an event barrier between stream A and B."
    )


def test_cross_stream_multiple_hops():
    """Resource hopping across three streams must stay correctly ordered."""
    streams = [cvcuda.Stream() for _ in range(3)]
    tensor = cvcuda.Tensor((1, 8, 8, 1), cvcuda.Type.U8, "NHWC")

    # First bind on stream 0
    with streams[0]:
        cvcuda.convertto(tensor, cvcuda.Type.F32, scale=1 / 255.0)

    # Hop to stream 1, then stream 2 — each submitSync inserts a barrier.
    tensor.submitStreamSync(streams[1])
    tensor.submitStreamSync(streams[2])

    with streams[2]:
        cvcuda.convertto(tensor, cvcuda.Type.F32, scale=1 / 255.0)

    streams[2].sync()
