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

import cvcuda
import pytest as t
import numpy as np
import cvcuda_util as util
import threading
import queue


RNG = np.random.default_rng(0)


@t.mark.parametrize(
    "input_args,out_shape,interp",
    [
        (
            ((5, 16, 23, 4), np.uint8, "NHWC"),
            (5, 132, 15, 4),
            cvcuda.Interp.LINEAR,
        ),
        (
            ((16, 23, 4), np.uint8, "HWC"),
            (132, 15, 4),
            cvcuda.Interp.CUBIC,
        ),
        (((16, 23, 1), np.uint8, "HWC"), (132, 15, 1), None),
    ],
)
def test_op_resize(input_args, out_shape, interp):
    input = cvcuda.Tensor(*input_args)

    if interp is None:
        out = cvcuda.resize(input, out_shape)
    else:
        out = cvcuda.resize(input, out_shape, interp)
    assert out.layout == input.layout
    assert out.shape == out_shape
    assert out.dtype == input.dtype

    out = cvcuda.Tensor(out_shape, input.dtype, input.layout)
    if interp is None:
        tmp = cvcuda.resize_into(out, input)
    else:
        tmp = cvcuda.resize_into(out, input, interp)
    assert tmp is out
    assert out.layout == input.layout
    assert out.shape == out_shape
    assert out.dtype == input.dtype

    stream = cvcuda.Stream()
    if interp is None:
        out = cvcuda.resize(src=input, shape=out_shape, stream=stream)
    else:
        out = cvcuda.resize(src=input, shape=out_shape, interp=interp, stream=stream)
    assert out.layout == input.layout
    assert out.shape == out_shape
    assert out.dtype == input.dtype

    if interp is None:
        tmp = cvcuda.resize_into(src=input, dst=out, stream=stream)
    else:
        tmp = cvcuda.resize_into(src=input, dst=out, interp=interp, stream=stream)
    assert tmp is out
    assert out.layout == input.layout
    assert out.shape == out_shape
    assert out.dtype == input.dtype


@t.mark.parametrize(
    "inSize, outSize, interp",
    [((123, 321), (321, 123), cvcuda.Interp.LINEAR), ((123, 321), (321, 123), None)],
)
def test_op_resizevarshape(inSize, outSize, interp):

    RNG = np.random.default_rng(0)

    input = util.create_image_batch(
        10, cvcuda.Format.RGBA8, size=inSize, max_random=256, rng=RNG
    )

    base_output = util.create_image_batch(
        10, cvcuda.Format.RGBA8, size=outSize, max_random=256, rng=RNG
    )

    sizes = []
    for image in base_output:
        sizes.append([image.width, image.height])

    if interp is None:
        out = cvcuda.resize(input, sizes)
    else:
        out = cvcuda.resize(src=input, sizes=sizes, interp=interp)

    assert len(out) == len(input)
    assert out.capacity == input.capacity
    assert out.uniqueformat == input.uniqueformat
    assert out.maxsize == outSize

    stream = cvcuda.cuda.Stream()
    if interp is None:
        tmp = cvcuda.resize_into(src=input, dst=base_output, stream=stream)
    else:
        tmp = cvcuda.resize_into(
            src=input, dst=base_output, interp=interp, stream=stream
        )
    assert tmp is base_output
    assert len(base_output) == len(input)
    assert out.capacity == input.capacity
    assert out.uniqueformat == input.uniqueformat
    assert out.maxsize == outSize


def test_op_resize_multithread():
    src_dtype, src_layout = np.uint8, "NHWC"
    src_shape = (2, 720, 1280, 3)
    dst_shape = (2, 72, 128, 3)
    dst_queue = queue.Queue()
    num_threads = 16
    threads = []

    def thread_run(dst_queue, src, dst_shape):
        import cvcuda

        dst = cvcuda.resize(src, dst_shape)
        dst_queue.put(dst)

    for i in range(num_threads):
        src = cvcuda.Tensor(
            (src_shape[0], src_shape[1] + i, src_shape[2] + i, src_shape[3]),
            src_dtype,
            src_layout,
        )
        thread = threading.Thread(target=thread_run, args=(dst_queue, src, dst_shape))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    while not dst_queue.empty():
        dst = dst_queue.get()
        assert dst.layout == src_layout
        assert dst.dtype == src_dtype
        assert dst.shape == dst_shape


def test_op_resize_user_stream_with_tensor():
    stream = cvcuda.Stream()
    src_shape = (5, 1080, 1920, 4)
    dst_shape = (5, 108, 192, 4)
    dtype, layout = np.uint8, "NHWC"
    src = cvcuda.Tensor(src_shape, dtype, layout)

    with stream:
        dst = cvcuda.resize(src, dst_shape, cvcuda.Interp.NEAREST)
        assert dst.layout == layout
        assert dst.shape == dst_shape
        assert dst.dtype == dtype


@t.mark.parametrize("batch_size", [5])
def test_op_resize_user_stream_with_image_batch(batch_size):
    stream = cvcuda.Stream()
    src_shape = (batch_size, 1080, 1920, 4)
    dst_shape = (batch_size, 108, 192, 4)
    dst_sizes = [(dst_shape[2], dst_shape[1]) for _ in range(src_shape[0])]
    src = util.create_image_batch(
        src_shape[0],
        cvcuda.Format.RGBA8,
        max_size=(dst_shape[2], dst_shape[1]),
        rng=RNG,
    )

    with stream:
        dst = cvcuda.resize(src, dst_sizes, cvcuda.Interp.NEAREST)
        assert len(dst) == len(src)
        assert dst.uniqueformat == dst.uniqueformat
        assert dst.maxsize == dst_sizes[0]
