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

"""Regression guard for an input-side CAI v3 stream race.

When a producer (e.g., cupy) launches a fill kernel on a non-blocking CUDA
stream but advertises ``__cuda_array_interface__["stream"] == 1`` (legacy
default per the CAI v3 spec), an event-based barrier on the legacy default
stream does NOT capture the producer's work — non-blocking streams have
no implicit synchronization with the legacy default.  A subsequent cvcuda
op on a distinct cvcuda stream would then read the buffer before the fill
has retired.

cvcuda's ``priv::Resource::submitSync`` falls back to ``cudaDeviceSynchronize``
when ``prevHandle`` is one of the CAI default-stream sentinels (``cudaStreamLegacy``
or ``cudaStreamPerThread``), which catches both this case and PyTorch's
CAI v2 (no stream field).  Combined with ResourceGuard's ``run()`` pattern
that inserts barriers BEFORE the consumer kernel queues, the race is
closed in production cvcuda.  This test pins both halves of the fix.
"""

import gc

import cupy
import pytest

import cvcuda

# GPU-side busy-wait kernel (idiom borrowed from test_interop_cai_stream.py).
# Runs on an SM and does not require host callbacks, so pytest's GIL is not
# involved.  ~500 ms on a 1 GHz SM clock — comfortably above any host-side
# event-record/event-wait latency we might race against.
_DELAY_KERNEL = cupy.RawKernel(
    r"""
extern "C" __global__ void cvcuda_test_busy_wait(unsigned long long cycles) {
    unsigned long long start = clock64();
    while (clock64() - start < cycles) { /* spin */ }
}
""",
    "cvcuda_test_busy_wait",
)


def _warm_up_varshape_autocontrast(shape, consumer_stream):
    """Prime AutoContrast while keeping its warm-up resources alive."""
    size = (shape[1], shape[0])
    warm_src = cvcuda.ImageBatchVarShape(1)
    warm_src.pushback(cvcuda.Image(size, cvcuda.Format.RGB8))
    warm_dst = cvcuda.ImageBatchVarShape(1)
    warm_dst.pushback(cvcuda.Image(size, cvcuda.Format.RGB8))
    with consumer_stream:
        cvcuda.autocontrast_into(warm_dst, warm_src, stream=consumer_stream)
    consumer_stream.sync()
    return size, (warm_src, warm_dst)


def _use_then_release_external_image_wrapper(
    shape, consumer_stream, other_batch, *, external_is_output
):
    """Give an external wrapper stream ownership, then make it reusable."""
    buffer = cupy.zeros(shape, dtype=cupy.uint8)
    image = cvcuda.as_image(buffer)
    wrapper_id = image.id
    batch = cvcuda.ImageBatchVarShape(1)
    batch.pushback(image)

    src_batch, dst_batch = (
        (other_batch, batch) if external_is_output else (batch, other_batch)
    )
    with consumer_stream:
        cvcuda.autocontrast_into(dst_batch, src_batch, stream=consumer_stream)
    consumer_stream.sync()

    batch.clear()
    del batch, image, buffer
    gc.collect()
    cvcuda.internal.syncAuxStream()
    return wrapper_id


@pytest.mark.parametrize("iterations", [3])
def test_cvtcolor_honors_cai_stream_when_producer_uses_nonblocking_stream(iterations):
    """Producer fills on a non-blocking cupy stream; consumer (cvcuda) must
    observe the fill, not race against it."""
    producer_stream = cupy.cuda.Stream(non_blocking=True)
    cycles = 500_000_000  # ~500 ms

    for i in range(iterations):
        cvcuda.clear_cache()
        consumer_stream = cvcuda.Stream()

        with producer_stream:
            # Push a long busy-wait, then the fill.  Both are queued on
            # producer_stream (non-blocking).  The fill's actual completion
            # is far enough out that any timing-flaky barrier in cvcuda
            # surfaces as wrong output.
            _DELAY_KERNEL((1,), (1,), (cycles,))
            src = cupy.full((1, 8, 8, 3), fill_value=100, dtype=cupy.uint8)

        src_nv = cvcuda.as_tensor(src, "NHWC")
        with consumer_stream:
            out = cvcuda.cvtcolor(
                src_nv,
                cvcuda.ColorConversion.BGR2GRAY,
                stream=consumer_stream,
            )
        consumer_stream.sync()

        result = cupy.asarray(out.cuda()).get()
        # BGR2GRAY of (100, 100, 100) is 100. If the cvcuda kernel ran
        # before the producer's fill completed, the buffer is uninitialized
        # / partially filled and the output is a mix of values.
        unique = sorted(set(result.flatten().tolist()))
        assert (result == 100).all(), (
            f"iter {i}: cvcuda read producer buffer before fill completed. "
            f"Output values: {unique}. This is the input-side CAI v3 stream "
            f"race: cupy advertises stream=1 (legacy default) but launched "
            f"on a non-blocking stream, so cvcuda's event-based barrier on "
            f"the legacy default missed the producer's work."
        )


@pytest.mark.parametrize("iterations", [3])
def test_autocontrast_honors_cai_stream_when_producer_uses_nonblocking_stream(
    iterations,
):
    """AutoContrast must insert ResourceGuard barriers before its kernels."""
    producer_stream = cupy.cuda.Stream(non_blocking=True)
    consumer_stream = cvcuda.Stream()
    cycles = 500_000_000
    shape = (1, 128, 128, 3)

    # Prime the cached operator and its internal min/max workspace. Otherwise,
    # its first cudaMalloc can serialize the device and accidentally hide the race.
    warm_src = cvcuda.Tensor(shape, cvcuda.Type.U8, "NHWC")
    warm_dst = cvcuda.Tensor(shape, cvcuda.Type.U8, "NHWC")
    with consumer_stream:
        cvcuda.autocontrast_into(warm_dst, warm_src, stream=consumer_stream)
    consumer_stream.sync()

    for i in range(iterations):
        src = cupy.zeros(shape, dtype=cupy.uint8)
        dst = cvcuda.Tensor(shape, cvcuda.Type.U8, "NHWC")
        cupy.cuda.get_current_stream().synchronize()

        with producer_stream:
            _DELAY_KERNEL((1,), (1,), (cycles,))
            src.fill(100)

        src_nv = cvcuda.as_tensor(src, "NHWC")
        with consumer_stream:
            cvcuda.autocontrast_into(dst, src_nv, stream=consumer_stream)
        consumer_stream.sync()

        result = cupy.asarray(dst.cuda()).get()
        unique = sorted(set(result.flatten().tolist()))
        assert (result == 100).all(), (
            f"iter {i}: AutoContrast read its producer-owned input before the "
            f"fill completed. Output values: {unique}. ResourceGuard barriers "
            "must be submitted before the operator kernels."
        )


def test_autocontrast_varshape_honors_cai_stream_when_producer_uses_nonblocking_stream():
    """VarShape AutoContrast must guard image inputs before its kernels."""
    cvcuda.clear_cache()

    producer_stream = cupy.cuda.Stream(non_blocking=True)
    consumer_stream = cvcuda.Stream()
    cycles = 500_000_000
    shape = (128, 128, 3)

    # Prime the VarShape overload and its internal min/max workspace. Otherwise,
    # its first cudaMalloc can serialize the device and accidentally hide the race.
    size, _warm_batches = _warm_up_varshape_autocontrast(shape, consumer_stream)

    # Give one external wrapper prior ownership on the consumer stream, then
    # release it completely so the producer-owned input below must rebind it.
    first_dst_image = cvcuda.Image(size, cvcuda.Format.RGB8)
    first_dst_batch = cvcuda.ImageBatchVarShape(1)
    first_dst_batch.pushback(first_dst_image)
    first_id = _use_then_release_external_image_wrapper(
        shape,
        consumer_stream,
        first_dst_batch,
        external_is_output=False,
    )
    first_dst_batch.clear()
    del first_dst_batch, first_dst_image

    src = cupy.zeros(shape, dtype=cupy.uint8)
    dst_image = cvcuda.Image(size, cvcuda.Format.RGB8)
    dst_batch = cvcuda.ImageBatchVarShape(1)
    dst_batch.pushback(dst_image)
    cupy.cuda.get_current_stream().synchronize()

    with producer_stream:
        _DELAY_KERNEL((1,), (1,), (cycles,))
        src.fill(100)

    src_image = cvcuda.as_image(src)
    assert src_image.id == first_id, (
        "The producer-owned input did not rebind the completed cached Image "
        f"wrapper: first id={first_id}, rebound id={src_image.id}"
    )
    src_batch = cvcuda.ImageBatchVarShape(1)
    src_batch.pushback(src_image)
    with consumer_stream:
        cvcuda.autocontrast_into(dst_batch, src_batch, stream=consumer_stream)
    consumer_stream.sync()

    result = cupy.asarray(dst_image.cuda()).get()
    unique = sorted(set(result.flatten().tolist()))
    assert (result == 100).all(), (
        "VarShape AutoContrast read its producer-owned input before the fill "
        f"completed. Output values: {unique}. ResourceGuard barriers must be "
        "submitted before the operator kernels."
    )


def test_autocontrast_varshape_honors_cai_stream_for_producer_owned_output():
    """VarShape AutoContrast must guard image outputs before its kernels."""
    cvcuda.clear_cache()

    producer_stream = cupy.cuda.Stream(non_blocking=True)
    consumer_stream = cvcuda.Stream()
    cycles = 500_000_000
    shape = (128, 128, 3)

    # Prime the VarShape overload and its internal min/max workspace. Otherwise,
    # its first cudaMalloc can serialize the device and accidentally hide the race.
    _size, _warm_batches = _warm_up_varshape_autocontrast(shape, consumer_stream)

    src = cupy.full(shape, fill_value=100, dtype=cupy.uint8)
    cupy.cuda.get_current_stream().synchronize()
    src_image = cvcuda.as_image(src)
    src_image.submitStreamSync(consumer_stream)
    src_batch = cvcuda.ImageBatchVarShape(1)
    src_batch.pushback(src_image)

    # Give one external wrapper prior ownership on the consumer stream, then
    # release it completely so the producer-owned output below must rebind it.
    first_id = _use_then_release_external_image_wrapper(
        shape,
        consumer_stream,
        src_batch,
        external_is_output=True,
    )

    dst = cupy.zeros(shape, dtype=cupy.uint8)
    cupy.cuda.get_current_stream().synchronize()
    with producer_stream:
        _DELAY_KERNEL((1,), (1,), (cycles,))
        dst.fill(7)

    dst_image = cvcuda.as_image(dst)
    assert dst_image.id == first_id, (
        "The producer-owned output did not rebind the completed cached Image "
        f"wrapper: first id={first_id}, rebound id={dst_image.id}"
    )
    dst_batch = cvcuda.ImageBatchVarShape(1)
    dst_batch.pushback(dst_image)
    with consumer_stream:
        cvcuda.autocontrast_into(dst_batch, src_batch, stream=consumer_stream)
    consumer_stream.sync()
    producer_stream.synchronize()

    result = dst.get()
    unique = sorted(set(result.flatten().tolist()))
    assert (result == 100).all(), (
        "A producer-owned VarShape output overwrote AutoContrast after "
        f"submission. Output values: {unique}. ResourceGuard barriers must "
        "include each output image before the operator kernels."
    )


def test_cleared_batch_releases_external_image_wrapper_for_rebind():
    """A cleared batch must release its external Image wrapper for reuse.

    The stream-race regressions need to prove that ``as_image`` rebound an
    existing wrapper rather than created a fresh one. A cached
    ``ImageBatchVarShape`` retains its pushed Images, so the deterministic
    release sequence must clear the batch before dropping Python handles and
    synchronizing the auxiliary callback stream.
    """
    cvcuda.clear_cache()

    consumer_stream = cvcuda.Stream()
    size = (128, 128)
    shape = (size[1], size[0], 3)

    src_batch = cvcuda.ImageBatchVarShape(1)
    src_batch.pushback(cvcuda.Image(size, cvcuda.Format.RGB8))

    first_id = _use_then_release_external_image_wrapper(
        shape,
        consumer_stream,
        src_batch,
        external_is_output=True,
    )

    second_buffer = cupy.zeros(shape, dtype=cupy.uint8)
    second_image = cvcuda.as_image(second_buffer)
    second_id = second_image.id

    try:
        assert second_id == first_id, (
            "The completed external Image wrapper remained held after both "
            f"streams synchronized: first id={first_id}, second id={second_id}"
        )
    finally:
        del second_image, second_buffer
        cvcuda.clear_cache()
