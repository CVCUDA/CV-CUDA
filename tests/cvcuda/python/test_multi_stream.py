# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import torch
import cvcuda
import pytest as t


def test_multiple_streams():
    stream1 = cvcuda.cuda.Stream()  # create a new stream
    stream2 = cvcuda.cuda.Stream()  # create a new stream
    stream3 = cvcuda.cuda.Stream()  # create a new stream
    assert stream1 is not stream2
    assert stream1 is not stream3
    assert cvcuda.cuda.Stream.current is cvcuda.cuda.Stream.default
    assert cvcuda.cuda.Stream.current is not stream1
    assert cvcuda.cuda.Stream.current is not stream2
    assert cvcuda.cuda.Stream.current is not stream3


def test_stream_context():
    stream1 = cvcuda.cuda.Stream()  # create a new stream
    stream2 = cvcuda.cuda.Stream()  # create a new stream
    with stream1:
        assert cvcuda.cuda.Stream.current is stream1
    with stream2:
        assert cvcuda.cuda.Stream.current is stream2
    assert cvcuda.cuda.Stream.current is cvcuda.cuda.Stream.default


def test_stream_context_nested():
    stream1 = cvcuda.cuda.Stream()  # create a new stream
    stream2 = cvcuda.cuda.Stream()  # create a new stream
    with stream1:
        assert cvcuda.cuda.Stream.current is stream1
        with stream2:
            assert cvcuda.cuda.Stream.current is stream2
        assert cvcuda.cuda.Stream.current is stream1
    assert cvcuda.cuda.Stream.current is cvcuda.cuda.Stream.default
    with stream2:
        assert cvcuda.cuda.Stream.current is stream2
    assert cvcuda.cuda.Stream.current is cvcuda.cuda.Stream.default


def test_stream_context_exception():
    stream1 = cvcuda.cuda.Stream()  # create a new stream
    stream2 = cvcuda.cuda.Stream()  # create a new stream
    with t.raises(Exception):
        with stream1:
            assert cvcuda.cuda.Stream.current is stream1
            with stream2:
                assert cvcuda.cuda.Stream.current is stream2
                raise Exception()
            assert cvcuda.cuda.Stream.current is stream1
        assert cvcuda.cuda.Stream.current is cvcuda.cuda.Stream.default
    with stream2:
        assert cvcuda.cuda.Stream.current is stream2
    assert cvcuda.cuda.Stream.current is cvcuda.cuda.Stream.default


def test_operator_stream():
    stream1 = cvcuda.cuda.Stream()  # create a new stream
    stream2 = cvcuda.cuda.Stream()  # create a new stream
    stream3 = cvcuda.cuda.Stream()  # create a new stream
    assert stream1 is not stream2
    assert stream1 is not stream3
    assert cvcuda.cuda.Stream.current is cvcuda.cuda.Stream.default
    assert cvcuda.cuda.Stream.current is not stream1
    assert cvcuda.cuda.Stream.current is not stream2
    assert cvcuda.cuda.Stream.current is not stream3
    with stream1:
        assert cvcuda.cuda.Stream.current is stream1
        img = torch.zeros(10, 10, 3, dtype=torch.uint8, device="cuda")
        img = cvcuda.as_tensor(img, "HWC")
        cvcuda.cvtcolor(img, cvcuda.ColorConversion.BGR2GRAY)
        assert cvcuda.cuda.Stream.current is stream1
    with stream2:
        assert cvcuda.cuda.Stream.current is stream2
        img = torch.zeros(10, 10, 3, dtype=torch.uint8, device="cuda")
        img = cvcuda.as_tensor(img, "HWC")
        cvcuda.cvtcolor(img, cvcuda.ColorConversion.BGR2GRAY)
        assert cvcuda.cuda.Stream.current is stream2
    with stream3:
        assert cvcuda.cuda.Stream.current is stream3
        img = torch.zeros(10, 10, 3, dtype=torch.uint8, device="cuda")
        img = cvcuda.as_tensor(img, "HWC")
        cvcuda.cvtcolor(img, cvcuda.ColorConversion.BGR2GRAY)
        assert cvcuda.cuda.Stream.current is stream3
    assert cvcuda.cuda.Stream.current is cvcuda.cuda.Stream.default


def test_operator_changing_stream():

    N = 10
    H = 1080
    W = 1080
    C = 3
    Loop = 50
    streams = [cvcuda.cuda.Stream() for _ in range(4)]  # create a list of streams

    inputTensor = torch.randint(0, 256, (N, H, W, C), dtype=torch.uint8).cuda()
    outputTensor = torch.randint(0, 256, (N, H, W, C), dtype=torch.uint8).cuda()
    # Perform deep copy
    inputTensor_copy = inputTensor.clone()

    inTensor = cvcuda.as_tensor(inputTensor.data, "NHWC")
    outTensor = cvcuda.as_tensor(outputTensor.data, "NHWC")

    for _ in range(Loop):
        for stream in streams:
            cvcuda.flip_into(outTensor, inTensor, -1, stream=stream)  # output x flipped
            cvcuda.flip_into(inTensor, outTensor, -1, stream=stream)  # output y flipped

    final_out = torch.as_tensor(inTensor.cuda()).cpu()
    assert torch.equal(final_out, inputTensor_copy.cpu())


def test_operator_changing_stream_loaded():

    N = 10
    H = 1080
    W = 1080
    C = 3
    Loop = 50
    stream1 = cvcuda.cuda.Stream()
    stream2 = cvcuda.cuda.Stream()

    inputTensor = torch.randint(0, 256, (N, H, W, C), dtype=torch.uint8).cuda()
    inputTensorTmp = torch.randint(0, 256, (N, H, W, C), dtype=torch.uint8).cuda()
    outputTensor = torch.randint(0, 256, (N, H, W, C), dtype=torch.uint8).cuda()
    # Perform deep copy
    inputTensor_copy = inputTensor.clone()

    inTensor = cvcuda.as_tensor(inputTensor.data, "NHWC")
    inTensorTmp = cvcuda.as_tensor(inputTensorTmp.data, "NHWC")
    outTensor = cvcuda.as_tensor(outputTensor.data, "NHWC")

    for _ in range(Loop):
        # put a bunch of work on stream 1
        for _ in range(Loop * 2):
            cvcuda.flip(inTensorTmp, 0, stream=stream1)
        # put a bunch of work on stream 1 this will happen after the above work on stream 1
        cvcuda.flip_into(
            inTensorTmp, inTensor, -1, stream=stream1
        )  # output x/y flipped
        cvcuda.flip_into(
            outTensor, inTensorTmp, -1, stream=stream2
        )  # output y/y flipped

    final_out = torch.as_tensor(outTensor.cuda()).cpu()
    assert torch.equal(final_out, inputTensor_copy.cpu())
