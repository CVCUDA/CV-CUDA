# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import ctypes

import numpy as np
import pytest as t

import cvcuda
import cupy


def test_stream_gcbag_vs_streamsync_race_condition():
    inputImage = cupy.asarray(
        np.random.randint(0, 256, (100, 1500, 1500, 3), dtype=np.uint8)
    )
    cvcudaInputTensor = cvcuda.as_tensor(inputImage, "NHWC")
    inputmap = cupy.asarray(
        np.random.randint(0, 256, (100, 1500, 1500, 2), dtype=np.uint8).astype(
            np.float32
        )
    )
    cvcudaInputMap = cvcuda.as_tensor(inputmap, "NHWC")

    cvcuda_stream = cvcuda.Stream()
    with cvcuda_stream:
        cvcudaResizeTensor = cvcuda.remap(cvcudaInputTensor, cvcudaInputMap)
    del cvcudaResizeTensor


def test_current_stream():
    assert cvcuda.Stream.current is cvcuda.Stream.default
    assert type(cvcuda.Stream.current) is cvcuda.Stream


def test_user_stream():
    with cvcuda.Stream():
        assert cvcuda.Stream.current is not cvcuda.Stream.default
    stream = cvcuda.Stream()
    with stream:
        assert stream is cvcuda.Stream.current
        assert stream is not cvcuda.Stream.default
    assert stream is not cvcuda.Stream.default
    assert stream is not cvcuda.Stream.current


def test_nested_streams():
    stream1 = cvcuda.Stream()
    stream2 = cvcuda.Stream()
    assert stream1 is not stream2
    with stream1:
        with stream2:
            assert stream2 is cvcuda.Stream.current
            assert stream1 is not cvcuda.Stream.current
        assert stream2 is not cvcuda.Stream.current
        assert stream1 is cvcuda.Stream.current


def test_wrap_stream_voidp():
    stream = cupy.cuda.Stream()

    extStream = ctypes.c_void_p(stream.ptr)

    cvcudaStream = cvcuda.as_stream(extStream)

    assert extStream.value == cvcudaStream.handle


def test_wrap_stream_int():
    stream = cupy.cuda.Stream()

    extStream = int(stream.ptr)

    cvcudaStream = cvcuda.as_stream(extStream)

    assert extStream == cvcudaStream.handle


def test_stream_conv_to_int():
    stream = cvcuda.Stream()

    assert stream.handle == int(stream)


class MockStream:
    def __init__(self, cuda_stream=None):
        if cuda_stream:
            self.m_stream = cupy.cuda.ExternalStream(cuda_stream)
        else:
            self.m_stream = cupy.cuda.Stream()

    def cuda_stream(self):
        return self.m_stream.ptr

    def stream(self):
        return self.m_stream


@t.mark.parametrize(
    "stream_type",
    [
        MockStream,
    ],
)
def test_wrap_stream_external(stream_type):
    extstream = stream_type()

    # Keep the underlying cupy stream alive across the del below.
    # cupy.cuda.Stream eagerly destroys the CUDA stream in __del__,
    # so we must prevent GC from reclaiming it.
    underlying = extstream.stream()

    stream = cvcuda.as_stream(underlying.ptr)

    assert extstream.cuda_stream() == stream.handle

    del extstream

    extstream = stream_type(stream.handle)
    stream = cvcuda.as_stream(extstream.stream().ptr)

    assert extstream.cuda_stream() == stream.handle

    del underlying


def test_as_stream_cupy_object():
    """cvcuda.as_stream() must accept a cupy.cuda.Stream object directly, not just
    an integer handle.  Without a dedicated type_caster this raises TypeError."""
    stream = cupy.cuda.Stream()
    cvcuda_stream = cvcuda.as_stream(stream)
    assert cvcuda_stream.handle == stream.ptr


def test_as_stream_cupy_object_keeps_stream_alive():
    """When wrapping a cupy stream *by object*, cvcuda must keep the stream alive
    for as long as the cvcuda wrapper exists.
    If the wrapper stores only the integer (m_wrappedObj = int), the cupy stream
    is destroyed the moment the caller drops their reference, leaving a dead handle."""
    import gc

    cupy_stream = cupy.cuda.Stream()
    handle = cupy_stream.ptr

    cvcuda_stream = cvcuda.as_stream(cupy_stream)

    # Drop caller's reference to the cupy stream.
    del cupy_stream
    gc.collect()

    # cvcuda_stream must still hold the cupy stream alive via m_wrappedObj.
    # If the stream was destroyed, streamSynchronize will raise.
    cupy.cuda.runtime.streamSynchronize(cvcuda_stream.handle)
    assert cvcuda_stream.handle == handle


def test_as_stream_cupy_stream_switch():
    """A resource submitted on a cupy stream (via as_stream(cupy_stream)) can be
    safely used on a different stream even after the caller drops their cupy reference.

    as_stream(cupy_stream) keeps the cupy stream alive via m_wrappedObj, so the
    CUDA handle remains valid when submitSync synchronizes against it.
    The chain is: out_nv -> Resource -> m_lastStream -> cvcuda Stream -> cupy_stream.
    """
    import gc

    src = cupy.full((1, 4, 4, 3), fill_value=100, dtype=cupy.uint8)
    src_nv = cvcuda.as_tensor(src, "NHWC")

    # Wrap by object (not .ptr) so cvcuda holds a strong ref to the cupy stream.
    cupy_stream = cupy.cuda.Stream()
    cvcuda_stream = cvcuda.as_stream(cupy_stream)
    with cvcuda_stream:
        out_nv = cvcuda.cvtcolor(
            src_nv, cvcuda.ColorConversion.BGR2GRAY, stream=cvcuda_stream
        )
    cupy_stream.synchronize()

    # Drop caller's references.  The cupy stream stays alive via the ref chain above.
    del cupy_stream
    del cvcuda_stream
    gc.collect()

    # Use out_nv on a fresh native stream.  submitSync synchronizes against the
    # still-valid cupy stream handle held in m_lastStream.
    stream2 = cvcuda.Stream()
    with stream2:
        out2 = cvcuda.cvtcolor(out_nv, cvcuda.ColorConversion.GRAY2BGR, stream=stream2)
    stream2.sync()

    result = cupy.asarray(out2.cuda())
    assert result.shape == (1, 4, 4, 3)


def test_stream_default_is_zero():
    assert cvcuda.Stream.default.handle == 0


def test_stream_size_in_bytes():
    """
    Checks if the computation of the Stream size in bytes is correct
    """
    stream = cvcuda.Stream()
    assert cvcuda.internal.nbytes_in_cache(stream) == 0
