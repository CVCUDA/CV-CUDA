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

import nvcv
import torch
import ctypes
import pytest as t


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
    stream = torch.cuda.Stream()

    extStream = ctypes.c_void_p(stream.cuda_stream)

    nvcvStream = nvcv.cuda.as_stream(extStream)

    assert extStream.value == nvcvStream.handle


def test_wrap_stream_int():
    stream = torch.cuda.Stream()

    extStream = int(stream.cuda_stream)

    nvcvStream = nvcv.cuda.as_stream(extStream)

    assert extStream == nvcvStream.handle


def test_stream_conv_to_int():
    stream = nvcv.cuda.Stream()

    assert stream.handle == int(stream)


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


def test_stream_default_is_zero():
    assert nvcv.cuda.Stream.default.handle == 0
