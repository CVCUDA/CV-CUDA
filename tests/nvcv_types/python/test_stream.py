# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import nvcv
import torch
import numba
from numba import cuda
import pytest as t


assert cuda.is_available()


def test_current_stream():
    assert nvcv.cuda.Stream.current is nvcv.cuda.Stream.default
    assert type(nvcv.cuda.Stream.current) == nvcv.cuda.Stream


def test_user_stream():
    with nvcv.cuda.Stream():
        assert nvcv.cuda.Stream.current is not nvcv.cuda.Stream.default
    stream = nvcv.cuda.Stream()
    with stream:
        assert stream is nvcv.cuda.Stream.current
        assert stream is not nvcv.cuda.Stream.default
    assert stream is not nvcv.cuda.Stream.default
    assert stream is not nvcv.cuda.Stream.current


def test_nested_streams():
    stream1 = nvcv.cuda.Stream()
    stream2 = nvcv.cuda.Stream()
    assert stream1 is not stream2
    with stream1:
        with stream2:
            assert stream2 is nvcv.cuda.Stream.current
            assert stream1 is not nvcv.cuda.Stream.current
        assert stream2 is not nvcv.cuda.Stream.current
        assert stream1 is nvcv.cuda.Stream.current


def test_wrap_stream_voidp():
    numbaStream = numba.cuda.stream()
    nvcvStream = nvcv.cuda.as_stream(numbaStream.handle)

    assert numbaStream.handle.value == nvcvStream.handle


def test_wrap_stream_int():
    numbaStream = numba.cuda.stream()
    nvcvStream = nvcv.cuda.as_stream(numbaStream.handle.value)

    assert numbaStream.handle.value == nvcvStream.handle


def test_stream_conv_to_int():
    stream = nvcv.cuda.Stream()

    assert stream.handle == int(stream)


class NumbaStream:
    def __init__(self, cuda_stream=None):
        if cuda_stream:
            self.m_stream = numba.cuda.external_stream(cuda_stream)
        else:
            self.m_stream = numba.cuda.stream()

    def cuda_stream(self):
        return self.m_stream.handle.value

    def stream(self):
        return self.m_stream


class TorchStream:
    def __init__(self, cuda_stream=None):
        if cuda_stream:
            self.m_stream = torch.cuda.ExternalStream(cuda_stream)
        else:
            self.m_stream = torch.cuda.Stream()

    def cuda_stream(self):
        return self.m_stream.cuda_stream

    def stream(self):
        return self.m_stream


@t.mark.parametrize(
    "stream_type",
    [
        NumbaStream,
        TorchStream,
    ],
)
def test_wrap_stream_external(stream_type):
    extstream = stream_type()

    stream = nvcv.cuda.as_stream(extstream.stream())

    assert extstream.cuda_stream() == stream.handle

    # stream must hold a ref to the external stream, the wrapped cudaStream
    # must not have been deleted
    del extstream

    extstream = stream_type(stream.handle)
    stream = nvcv.cuda.as_stream(extstream.stream())

    assert extstream.cuda_stream() == stream.handle
