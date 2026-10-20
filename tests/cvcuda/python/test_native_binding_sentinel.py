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

from contextlib import contextmanager
import gc
import weakref

import cupy
import numpy as np
import pytest

import cvcuda
from cvcuda_test_utils import as_cupy_stream


_DELAY_KERNEL = cupy.RawKernel(
    r"""
extern "C" __global__ void cvcuda_native_binding_delay(
    unsigned long long cycles)
{
    unsigned long long start = clock64();
    while (clock64() - start < cycles) { }
}
""",
    "cvcuda_native_binding_delay",
)

_GATE_KERNEL = cupy.RawKernel(
    r"""
extern "C" __global__ void cvcuda_native_binding_gate(
    unsigned int *release, unsigned long long timeoutCycles)
{
    unsigned long long start = clock64();
    while (atomicAdd(release, 0U) == 0U
           && clock64() - start < timeoutCycles) { }
}
""",
    "cvcuda_native_binding_gate",
)

_SET_GATE_KERNEL = cupy.RawKernel(
    r"""
extern "C" __global__ void cvcuda_native_binding_set_gate(
    unsigned int *release, unsigned int value)
{
    atomicExch(release, value);
}
""",
    "cvcuda_native_binding_set_gate",
)

# A timeout makes a broken lifetime path fail instead of hanging if it calls a
# synchronizing CUDA allocator before the test can release the gate.
_GATE_TIMEOUT_CYCLES = np.uint64(10_000_000_000)


@contextmanager
def _park_stream(external_stream):
    release = cupy.empty(1, dtype=cupy.uint32)
    control_stream = cupy.cuda.Stream(non_blocking=True)
    _SET_GATE_KERNEL((1,), (1,), (release, np.uint32(0)), stream=control_stream)
    control_stream.synchronize()
    _GATE_KERNEL(
        (1,),
        (1,),
        (release, _GATE_TIMEOUT_CYCLES),
        stream=external_stream,
    )
    assert not external_stream.done
    try:
        yield
    finally:
        _SET_GATE_KERNEL((1,), (1,), (release, np.uint32(1)), stream=control_stream)
        control_stream.synchronize()
        external_stream.synchronize()


def _native_into_case(name, stream):
    src = cvcuda.Tensor((1, 8, 8, 3), np.uint8, "NHWC")
    stream_kw = {} if stream is None else {"stream": stream}

    if name == "reformat":
        dst = cvcuda.Tensor((1, 3, 8, 8), np.uint8, "NCHW")
        result = cvcuda.reformat_into(dst, src, **stream_kw)
    elif name == "resize":
        dst = cvcuda.Tensor((1, 4, 4, 3), np.uint8, "NHWC")
        result = cvcuda.resize_into(dst, src, cvcuda.Interp.NEAREST, **stream_kw)
    else:
        dst = cvcuda.Tensor((1, 8, 8, 3), np.float32, "NHWC")
        result = cvcuda.convertto_into(dst, src, **stream_kw)

    return src, dst, result


@pytest.mark.parametrize("name", ["reformat", "resize", "convertto"])
@pytest.mark.parametrize("explicit_stream", [False, True])
def test_native_into_preserves_identity(name, explicit_stream):
    stream = cvcuda.Stream() if explicit_stream else None

    _src, dst, result = _native_into_case(name, stream)

    assert result is dst
    (stream or cvcuda.Stream.current).sync()


def test_direct_native_resources_use_common_guard_failure_path():
    cvcuda._test.fail_submit_sync_only(True)
    try:
        with pytest.raises(
            ValueError, match="injected Resources_SubmitSyncOnly failure"
        ):
            _native_into_case("convertto", cvcuda.Stream())
    finally:
        cvcuda._test.fail_submit_sync_only(False)


def test_typed_operator_submission_lease_covers_cross_stream_enqueue():
    # The C++ probe forces thread A to stop after operator submitSync and
    # starts thread B with the same cached state on another stream. It fails
    # unless B remains outside submitSync until A performs its consumer enqueue.
    cvcuda.clear_cache()
    cache_items = cvcuda.cache_size()
    cvcuda._test.typed_operator_submission_lease()
    # The probe creates two streams plus one operator metadata item. Completion
    # leasing may create multiple native states, but none becomes another cache
    # entry.
    assert cvcuda.cache_size() == cache_items + 3


def test_native_resourceguard_none_is_lifetime_only():
    """NONE skips sync/serialization while retaining its owner to retirement."""
    cvcuda._test.native_resourceguard_modes()


def _findhomography_case():
    shape = (1, 16, 2)
    return (
        cvcuda.Tensor(shape, np.float32, "NWC"),
        cvcuda.Tensor(shape, np.float32, "NWC"),
        cvcuda.Tensor((1, 3, 3), np.float32, "NHW"),
    )


def _submit_findhomography(case, stream, operator=None):
    src, dst, models = case
    if operator is None:
        cvcuda.findhomography_into(models, src, dst, stream=stream)
    else:
        cvcuda.findhomography_into_with_op(models, src, dst, operator, stream=stream)


def test_findhomography_workspace_state_is_completion_leased():
    """Independent cache acquisitions must not share in-flight workspace."""
    cvcuda.clear_cache()
    first = cvcuda.get_findhomography_operator(1, 16)
    second = cvcuda.get_findhomography_operator(1, 16)

    # Warm the solver and, when completion leasing is enabled, preallocate two
    # states before the delayed submission. A cudaMalloc during the overlap
    # could otherwise synchronize the device and hide cache serialization.
    warm_stream = cvcuda.Stream()
    _submit_findhomography(_findhomography_case(), warm_stream, first)
    _submit_findhomography(_findhomography_case(), warm_stream, second)
    warm_stream.sync()
    del first, second
    gc.collect()

    stream_a = cvcuda.Stream()
    stream_b = cvcuda.Stream()
    external_a = as_cupy_stream(stream_a.handle)
    case_a = _findhomography_case()
    case_b = _findhomography_case()

    _DELAY_KERNEL(
        (1,),
        (1,),
        (np.uint64(500_000_000),),
        stream=external_a,
    )
    assert not external_a.done

    _submit_findhomography(case_a, stream_a)
    _submit_findhomography(case_b, stream_b)
    stream_b.sync()
    submission_a_still_pending = not external_a.done
    stream_a.sync()

    assert submission_a_still_pending, (
        "the second cache acquisition waited for the first stream, proving "
        "that both submissions reused one FindHomography workspace state"
    )


def test_prefetched_findhomography_operator_tracks_workspace_across_streams():
    """A caller-held operator must still order its workspace submissions."""
    cvcuda.clear_cache()
    operator = cvcuda.get_findhomography_operator(1, 16)

    warm_stream = cvcuda.Stream()
    _submit_findhomography(_findhomography_case(), warm_stream, operator)
    warm_stream.sync()

    stream_a = cvcuda.Stream()
    stream_b = cvcuda.Stream()
    external_a = as_cupy_stream(stream_a.handle)
    case_a = _findhomography_case()
    case_b = _findhomography_case()

    _DELAY_KERNEL(
        (1,),
        (1,),
        (np.uint64(500_000_000),),
        stream=external_a,
    )
    assert not external_a.done

    _submit_findhomography(case_a, stream_a, operator)
    _submit_findhomography(case_b, stream_b, operator)
    stream_b.sync()
    submission_a_completed = external_a.done
    stream_a.sync()

    assert submission_a_completed, (
        "the caller-held FindHomography operator was not tracked as a mutable "
        "resource, so its workspace was submitted concurrently on two streams"
    )


def test_mutable_operator_state_is_not_reused_before_gpu_retirement():
    """Pinned BndBox command staging must outlive its asynchronous copy."""
    cvcuda.clear_cache()
    stream = cvcuda.Stream()
    external_stream = as_cupy_stream(stream.handle)
    shape = (1, 16, 16, 4)
    src_buffer = cupy.zeros(shape, dtype=cupy.uint8)
    src = cvcuda.as_tensor(src_buffer, "NHWC")
    warm_dst = cvcuda.Tensor(shape, np.uint8, "NHWC")

    def boxes(count):
        return cvcuda.BndBoxesI(
            boxes=[
                [
                    cvcuda.BndBoxI(
                        box=(index, index, 3, 3),
                        thickness=1,
                        borderColor=(255, 255, 0),
                        fillColor=(0, 128, 255, 128),
                    )
                    for index in range(count)
                ]
            ]
        )

    three_boxes = boxes(3)
    one_box = boxes(1)

    # Prime the operator and capture independent reference results before
    # forcing the overlap.
    cvcuda.bndbox_into(warm_dst, src, three_boxes, stream=stream)
    stream.sync()
    expected_three = cupy.asarray(warm_dst.cuda()).get()
    cvcuda.bndbox_into(warm_dst, src, one_box, stream=stream)
    stream.sync()
    expected_one = cupy.asarray(warm_dst.cuda()).get()

    three_dst = cvcuda.Tensor(shape, np.uint8, "NHWC")
    one_dst = cvcuda.Tensor(shape, np.uint8, "NHWC")

    _DELAY_KERNEL(
        (1,),
        (1,),
        (np.uint64(300_000_000),),
        stream=external_stream,
    )
    assert not external_stream.done

    # The first call queues an asynchronous copy from three-command pinned
    # host staging. The second call must use another operator state until that
    # copy and its kernel retire; reusing the staging rewrites byte offsets and
    # makes the first kernel dereference an unaligned RectangleCommand.
    cvcuda.bndbox_into(three_dst, src, three_boxes, stream=stream)
    cvcuda.bndbox_into(one_dst, src, one_box, stream=stream)
    stream.sync()

    np.testing.assert_array_equal(cupy.asarray(three_dst.cuda()).get(), expected_three)
    np.testing.assert_array_equal(cupy.asarray(one_dst.cuda()).get(), expected_one)


def test_osd_elements_owner_is_retained_until_gpu_retirement():
    """Device-backed OSD element storage must outlive asynchronous kernels."""
    cvcuda.clear_cache()
    stream = cvcuda.Stream()
    external_stream = as_cupy_stream(stream.handle)
    shape = (1, 64, 64, 4)
    src = cvcuda.Tensor(shape, np.uint8, "NHWC")
    dst = cvcuda.Tensor(shape, np.uint8, "NHWC")

    def elements():
        return cvcuda.Elements(
            [
                [
                    cvcuda.PolyLine(
                        points=np.array([[1, 1], [20, 1], [10, 20]], dtype=np.int32),
                        thickness=1,
                        isClosed=True,
                        borderColor=(255, 0, 0),
                        fillColor=(0, 255, 0, 255),
                    )
                ]
            ]
        )

    # Prime operator and command-buffer allocation before forcing overlap.
    warm_elements = elements()
    cvcuda.osd_into(dst, src, warm_elements, stream=stream)
    stream.sync()
    cvcuda.internal.drain_pending_releases()

    submitted_elements = elements()
    submitted_ref = weakref.ref(submitted_elements)
    with _park_stream(external_stream):
        cvcuda.osd_into(dst, src, submitted_elements, stream=stream)
        del submitted_elements
        gc.collect()

        # PolyLine owns a CUDA allocation whose raw pointer is copied into the
        # OSD command stream. Dropping Elements here used to run cudaFree,
        # synchronizing the parked stream and leaving ownership implicit.
        assert not external_stream.done

    stream.sync()
    cvcuda.internal.drain_pending_releases()
    gc.collect()
    assert submitted_ref() is None


def _submit_constant_case(name, src, stream):
    if name == "reformat":
        dst = cvcuda.Tensor((1, 3, 64, 64), np.uint8, "NCHW")
        cvcuda.reformat_into(dst, src, stream=stream)
    elif name == "resize":
        dst = cvcuda.Tensor((1, 32, 32, 3), np.uint8, "NHWC")
        cvcuda.resize_into(dst, src, cvcuda.Interp.NEAREST, stream=stream)
    elif name == "flip":
        dst = cvcuda.Tensor((1, 64, 64, 3), np.uint8, "NHWC")
        cvcuda.flip_into(dst, src, 0, stream=stream)
    else:
        dst = cvcuda.Tensor((1, 64, 64, 3), np.float32, "NHWC")
        cvcuda.convertto_into(dst, src, stream=stream)
    return dst


@pytest.mark.parametrize("name", ["reformat", "resize", "convertto", "flip"])
def test_native_guard_orders_external_producer_before_submission(name):
    producer = cupy.cuda.Stream(non_blocking=True)
    consumer = cvcuda.Stream()

    # Prime one-time operator and allocator work so it cannot serialize the
    # delayed producer and hide a missing pre-submit barrier.
    warm_src = cvcuda.Tensor((1, 64, 64, 3), np.uint8, "NHWC")
    _submit_constant_case(name, warm_src, consumer)
    consumer.sync()

    src_buffer = cupy.zeros((1, 64, 64, 3), dtype=cupy.uint8)
    cupy.cuda.get_current_stream().synchronize()
    with producer:
        _DELAY_KERNEL((1,), (1,), (np.uint64(250_000_000),))
        src_buffer.fill(37)

    src = cvcuda.as_tensor(src_buffer, "NHWC")
    dst = _submit_constant_case(name, src, consumer)
    consumer.sync()

    result = cupy.asarray(dst.cuda()).get()
    assert (result == 37).all()


def test_native_guard_holds_tensor_cache_entries_until_gpu_completion():
    cvcuda.clear_cache()
    stream = cvcuda.Stream()
    external_stream = as_cupy_stream(stream.handle)

    shape = (1, 8, 8, 3)
    src = cvcuda.Tensor(shape, np.uint8, "NHWC")
    dst = cvcuda.Tensor(shape, np.uint8, "NHWC")
    held_ids = {src.id, dst.id}

    # Prime one-time operator and CUDA dispatch work before inserting the
    # delay. The first submission can serialize the stream while loading CUDA
    # code, completing the delay before the lifetime assertion below.
    cvcuda.convertto_into(dst, src, stream=stream)
    stream.sync()
    cvcuda.internal.drain_pending_releases()

    with _park_stream(external_stream):
        result = cvcuda.convertto_into(dst, src, stream=stream)
        assert result is dst

        del result, dst, src
        gc.collect()
        assert not external_stream.done

        # If the native guard dropped either holder at function return,
        # Tensor's cache would immediately hand that in-flight wrapper back.
        replacement0 = cvcuda.Tensor(shape, np.uint8, "NHWC")
        replacement1 = cvcuda.Tensor(shape, np.uint8, "NHWC")
        assert replacement0.id not in held_ids
        assert replacement1.id not in held_ids

    stream.sync()
    cvcuda.internal.drain_pending_releases()


@pytest.mark.parametrize("name", ["convertto", "flip"])
def test_operator_cache_does_not_grow_with_queued_submissions(name):
    cvcuda.clear_cache()
    stream = cvcuda.Stream()
    external_stream = as_cupy_stream(stream.handle)
    shape = (1, 8, 8, 3)
    src = cvcuda.Tensor(shape, np.uint8, "NHWC")
    dst = cvcuda.Tensor(shape, np.uint8, "NHWC")

    if name == "convertto":
        cvcuda.convertto_into(dst, src, stream=stream)
    else:
        cvcuda.flip_into(dst, src, 0, stream=stream)
    stream.sync()
    cvcuda.internal.drain_pending_releases()
    warmed_cache_size = cvcuda.cache_size()

    _DELAY_KERNEL(
        (1,),
        (1,),
        (np.uint64(500_000_000),),
        stream=external_stream,
    )
    assert not external_stream.done
    for _ in range(64):
        if name == "convertto":
            cvcuda.convertto_into(dst, src, stream=stream)
        else:
            cvcuda.flip_into(dst, src, 0, stream=stream)

    # Async holds retain the native operator state, while its separate cache
    # metadata remains reusable. Coupling the cache item to the in-flight
    # resource would manufacture one operator per queued call.
    assert cvcuda.cache_size() == warmed_cache_size

    stream.sync()
    cvcuda.internal.drain_pending_releases()


def test_tensor_and_varshape_resize_operator_cache_can_be_cleared():
    cvcuda.clear_cache()
    stream = cvcuda.Stream()

    tensor_src = cvcuda.Tensor((1, 8, 8, 3), np.uint8, "NHWC")
    tensor_dst = cvcuda.Tensor((1, 4, 4, 3), np.uint8, "NHWC")

    batch_src = cvcuda.ImageBatchVarShape(1)
    batch_src.pushback(cvcuda.Image((8, 8), cvcuda.Format.RGB8))
    batch_dst = cvcuda.ImageBatchVarShape(1)
    batch_dst.pushback(cvcuda.Image((4, 4), cvcuda.Format.RGB8))
    cache_size_without_operator = cvcuda.cache_size()

    cvcuda.resize_into(tensor_dst, tensor_src, cvcuda.Interp.NEAREST, stream=stream)
    cache_size_with_tensor_operator = cvcuda.cache_size()
    assert cache_size_with_tensor_operator == cache_size_without_operator + 1

    cvcuda.resize_into(batch_dst, batch_src, cvcuda.Interp.NEAREST, stream=stream)
    assert cvcuda.cache_size() == cache_size_with_tensor_operator
    stream.sync()

    # Tensor and VarShape share the same typed operator state/cache item, and
    # that shared entry follows cache teardown.
    assert cvcuda.cache_size() > 0
    cvcuda.clear_cache()
    assert cvcuda.cache_size() == 0
