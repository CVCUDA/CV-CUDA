# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import numpy as np
import pytest as t

import cvcuda
import cupy


def test_multiple_streams():
    stream1 = cvcuda.Stream()  # create a new stream
    stream2 = cvcuda.Stream()  # create a new stream
    stream3 = cvcuda.Stream()  # create a new stream
    assert stream1 is not stream2
    assert stream1 is not stream3
    assert stream2 is not stream3
    assert cvcuda.Stream.current is cvcuda.Stream.default
    assert cvcuda.Stream.current is not stream1
    assert cvcuda.Stream.current is not stream2
    assert cvcuda.Stream.current is not stream3


def test_stream_context():
    stream1 = cvcuda.Stream()  # create a new stream
    stream2 = cvcuda.Stream()  # create a new stream
    with stream1:
        assert cvcuda.Stream.current is stream1
    with stream2:
        assert cvcuda.Stream.current is stream2
    assert cvcuda.Stream.current is cvcuda.Stream.default


def test_stream_context_nested():
    stream1 = cvcuda.Stream()  # create a new stream
    stream2 = cvcuda.Stream()  # create a new stream
    with stream1:
        assert cvcuda.Stream.current is stream1
        with stream2:
            assert cvcuda.Stream.current is stream2
        assert cvcuda.Stream.current is stream1
    assert cvcuda.Stream.current is cvcuda.Stream.default
    with stream2:
        assert cvcuda.Stream.current is stream2
    assert cvcuda.Stream.current is cvcuda.Stream.default


class _CatchThisException(Exception):
    """A test specific Exception to check that we raise the correct Exception."""


def test_stream_context_exception():
    stream1 = cvcuda.Stream()  # create a new stream
    stream2 = cvcuda.Stream()  # create a new stream
    with t.raises(_CatchThisException):
        with stream1:
            assert cvcuda.Stream.current is stream1
            with stream2:
                assert cvcuda.Stream.current is stream2
                raise _CatchThisException()
            assert cvcuda.Stream.current is stream1
        assert cvcuda.Stream.current is cvcuda.Stream.default
    with stream2:
        assert cvcuda.Stream.current is stream2
    assert cvcuda.Stream.current is cvcuda.Stream.default


def test_operator_stream():
    stream1 = cvcuda.Stream()  # create a new stream
    stream2 = cvcuda.Stream()  # create a new stream
    stream3 = cvcuda.Stream()  # create a new stream
    assert stream1 is not stream2
    assert stream1 is not stream3
    assert stream2 is not stream3
    assert cvcuda.Stream.current is cvcuda.Stream.default
    assert cvcuda.Stream.current is not stream1
    assert cvcuda.Stream.current is not stream2
    assert cvcuda.Stream.current is not stream3
    with stream1:
        assert cvcuda.Stream.current is stream1
        img = cupy.asarray(np.zeros((10, 10, 3), dtype=np.uint8))
        img = cvcuda.as_tensor(img, "HWC")
        cvcuda.cvtcolor(img, cvcuda.ColorConversion.BGR2GRAY)
        assert cvcuda.Stream.current is stream1
    with stream2:
        assert cvcuda.Stream.current is stream2
        img = cupy.asarray(np.zeros((10, 10, 3), dtype=np.uint8))
        img = cvcuda.as_tensor(img, "HWC")
        cvcuda.cvtcolor(img, cvcuda.ColorConversion.BGR2GRAY)
        assert cvcuda.Stream.current is stream2
    with stream3:
        assert cvcuda.Stream.current is stream3
        img = cupy.asarray(np.zeros((10, 10, 3), dtype=np.uint8))
        img = cvcuda.as_tensor(img, "HWC")
        cvcuda.cvtcolor(img, cvcuda.ColorConversion.BGR2GRAY)
        assert cvcuda.Stream.current is stream3
    assert cvcuda.Stream.current is cvcuda.Stream.default
    stream1.sync()
    stream2.sync()
    stream3.sync()


def test_operator_changing_stream():

    N = 10
    H = 1080
    W = 1080
    C = 3
    Loop = 50
    streams = [cvcuda.Stream() for _ in range(4)]  # create a list of streams

    inputTensor = cupy.asarray(np.random.randint(0, 256, (N, H, W, C), dtype=np.uint8))
    outputTensor = cupy.asarray(np.random.randint(0, 256, (N, H, W, C), dtype=np.uint8))
    # Perform deep copy
    inputTensor_copy = inputTensor.copy()

    inTensor = cvcuda.as_tensor(inputTensor, "NHWC")
    outTensor = cvcuda.as_tensor(outputTensor, "NHWC")

    prev_stream = None
    for _ in range(Loop):
        for stream in streams:
            if prev_stream is not None:
                stream.wait_stream(prev_stream)
            cvcuda.flip_into(outTensor, inTensor, -1, stream=stream)  # output x flipped
            cvcuda.flip_into(inTensor, outTensor, -1, stream=stream)  # output y flipped
            prev_stream = stream

    streams[-1].sync()
    final_out = cupy.asarray(inTensor.cuda()).get()
    assert np.all(final_out == inputTensor_copy.get())


def test_operator_changing_stream_loaded():

    N = 10
    H = 1080
    W = 1080
    C = 3
    Loop = 50
    stream1 = cvcuda.Stream()
    stream2 = cvcuda.Stream()

    inputTensor = cupy.asarray(np.random.randint(0, 256, (N, H, W, C), dtype=np.uint8))
    inputTensorTmp = cupy.asarray(
        np.random.randint(0, 256, (N, H, W, C), dtype=np.uint8)
    )
    outputTensor = cupy.asarray(np.random.randint(0, 256, (N, H, W, C), dtype=np.uint8))
    # Perform deep copy
    inputTensor_copy = inputTensor.copy()

    inTensor = cvcuda.as_tensor(inputTensor, "NHWC")
    inTensorTmp = cvcuda.as_tensor(inputTensorTmp, "NHWC")
    outTensor = cvcuda.as_tensor(outputTensor, "NHWC")

    for _ in range(Loop):
        # put a bunch of work on stream 1
        for _ in range(Loop * 2):
            cvcuda.flip(inTensorTmp, 0, stream=stream1)
        # put a bunch of work on stream 1 this will happen after the above work on stream 1
        cvcuda.flip_into(
            inTensorTmp, inTensor, -1, stream=stream1
        )  # output x/y flipped
        stream2.wait_stream(stream1)
        cvcuda.flip_into(
            outTensor, inTensorTmp, -1, stream=stream2
        )  # output y/y flipped

    stream2.sync()
    final_out = cupy.asarray(outTensor.cuda()).get()
    assert np.all(final_out == inputTensor_copy.get())


def test_wait_stream_self():
    stream = cvcuda.Stream()
    # Waiting on yourself must be a no-op, not a deadlock.
    stream.wait_stream(stream)


def test_wait_stream_fanout():
    N, H, W, C = 2, 64, 64, 3
    stream1 = cvcuda.Stream()
    stream2 = cvcuda.Stream()
    stream3 = cvcuda.Stream()

    inputTensor = cupy.asarray(np.random.randint(0, 256, (N, H, W, C), dtype=np.uint8))
    scratchTensor = cupy.asarray(np.zeros((N, H, W, C), dtype=np.uint8))
    outputTensor2 = cupy.asarray(np.zeros((N, H, W, C), dtype=np.uint8))
    outputTensor3 = cupy.asarray(np.zeros((N, H, W, C), dtype=np.uint8))
    inputTensor_copy = inputTensor.copy()

    inTensor = cvcuda.as_tensor(inputTensor, "NHWC")
    scratch = cvcuda.as_tensor(scratchTensor, "NHWC")
    outTensor2 = cvcuda.as_tensor(outputTensor2, "NHWC")
    outTensor3 = cvcuda.as_tensor(outputTensor3, "NHWC")

    # stream1 writes a flipped version into scratch
    cvcuda.flip_into(scratch, inTensor, -1, stream=stream1)
    # stream2 and stream3 each wait for stream1, then double-flip back to the original
    stream2.wait_stream(stream1)
    stream3.wait_stream(stream1)
    cvcuda.flip_into(outTensor2, scratch, -1, stream=stream2)
    cvcuda.flip_into(outTensor3, scratch, -1, stream=stream3)

    stream2.sync()
    stream3.sync()

    assert np.all(cupy.asarray(outTensor2.cuda()).get() == inputTensor_copy.get())
    assert np.all(cupy.asarray(outTensor3.cuda()).get() == inputTensor_copy.get())
