#!/usr/bin/env python3
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

"""CV-CUDA Erase operator benchmark - Python equivalent of BenchErase.cpp"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cvcuda  # noqa: E402
from python_bench_utils import (  # noqa: E402
    get_input_kind,
    parse_shape,
    get_dtype,
    get_dtype_size,
    get_num_channels,
    get_format_from_dtype,
    create_tensor,
    create_image_batch_varshape,
    create_stream_cache,
    run_benchmark,
)


def _fill_i32_tensor(tensor, values, device_id):
    """Fill vector-typed int32 parameter tensors via their CUDA buffer."""

    import cupy as cp

    with cp.cuda.Device(device_id):
        cuda_buffer = tensor.cuda()
        ptr = cuda_buffer.__cuda_array_interface__["data"][0]
        data = cp.asarray(values, dtype=cp.int32)
        mem = cp.cuda.UnownedMemory(ptr, data.nbytes, cuda_buffer)
        view = cp.ndarray(
            data.shape, dtype=cp.int32, memptr=cp.cuda.MemoryPointer(mem, 0)
        )
        view[...] = data
        cp.cuda.get_current_stream().synchronize()


def erase(state):
    """Erase operator benchmark matching C++ BenchErase.cpp"""

    shape = parse_shape(state.get_string("shape"))
    dtype_str = state.get_string("InOutDataType")
    dtype = get_dtype(dtype_str)
    num_channels = get_num_channels(dtype_str)
    random_mode = state.get_string("randomMode")
    input_kind = get_input_kind(state.get_string("inputKind"))
    num_erase = state.get_int64("numErase")
    device_id = state.get_device()

    N, H, W = shape

    region_mode = random_mode == "torchvision"
    random = False if region_mode else {"random": True, "constant": False}[random_mode]
    seed = 0
    try:
        layout = state.get_string("layout")
    except Exception:
        layout = "NHWC"

    if layout not in ("NHWC", "NCHW", "NCHW_FAKE"):
        state.skip("Erase benchmark supports only NHWC, NCHW, and NCHW_FAKE layouts")
        return None

    is_planar = layout == "NCHW"
    is_fake_planar = layout == "NCHW_FAKE"
    if is_fake_planar and input_kind == "VarShape":
        state.skip("Fake-planar (NCHW_FAKE) erase benchmark is tensor-only")
        return None
    if region_mode and input_kind == "VarShape":
        state.skip("Torchvision Erase region benchmark is tensor-only")
        return None
    if not region_mode and is_planar and num_channels == 2:
        state.skip("Planar Erase benchmark does not support 2-channel layouts")
        return None
    if is_planar and input_kind == "VarShape" and num_channels == 1:
        state.skip("Single-channel varshape Erase has no distinct planar image layout")
        return None
    if is_planar and input_kind == "VarShape" and dtype_str == "uchar4":
        state.skip("RGBA8p varshape is unsupported by the Python image API")
        return None

    dtype_size = get_dtype_size(dtype_str)
    image_bytes = N * H * W * dtype_size
    region_height = max(H // 4, 1)
    region_width = max(W // 4, 1)
    param_bytes = (
        num_channels * region_height * region_width * 4
        if region_mode
        else num_erase * (8 + 12 + num_channels * 4 + 4)
    )
    if is_fake_planar:
        state.add_global_memory_reads(3 * image_bytes + param_bytes)
        state.add_global_memory_writes(3 * image_bytes)
    else:
        state.add_global_memory_reads(image_bytes + param_bytes)
        state.add_global_memory_writes(image_bytes)

    anchor = cvcuda.Tensor((num_erase,), cvcuda.Type._2S32, "N")
    erasing = cvcuda.Tensor((num_erase,), cvcuda.Type._3S32, "N")
    erase_mask = (1 << num_channels) - 1
    _fill_i32_tensor(anchor, [[0, 0]] * num_erase, device_id)
    _fill_i32_tensor(erasing, [[10, 10, erase_mask]] * num_erase, device_id)

    values = create_tensor(
        (num_erase * num_channels,),
        cvcuda.Type.F32,
        device_id,
        layout="N",
        fill_mode=1.0,
    )
    imgIdx = create_tensor(
        (num_erase,), cvcuda.Type.S32, device_id, layout="N", fill_mode=0
    )
    region_values = create_tensor(
        (num_channels, region_height, region_width),
        cvcuda.Type.F32,
        device_id,
        layout="CHW",
        fill_mode=1.0,
    )

    get_stream = create_stream_cache()

    if is_fake_planar:
        src = create_tensor(
            (N, num_channels, H, W),
            dtype,
            device_id,
            layout="NCHW",
            fill_mode="checkerboard",
        )
        inter_src = create_tensor(
            (N, H, W, num_channels), dtype, device_id, layout="NHWC", fill_mode=0
        )
        inter_dst = create_tensor(
            (N, H, W, num_channels), dtype, device_id, layout="NHWC", fill_mode=0
        )
        dst = create_tensor(
            (N, num_channels, H, W), dtype, device_id, layout="NCHW", fill_mode=0
        )

        def run_fake(launch):
            stream = get_stream(launch)
            cvcuda.reformat_into(inter_src, src, stream=stream)
            if region_mode:
                cvcuda.erase_into(
                    inter_dst,
                    inter_src,
                    0,
                    0,
                    region_height,
                    region_width,
                    region_values,
                    stream=stream,
                )
            else:
                cvcuda.erase_into(
                    inter_dst,
                    inter_src,
                    anchor,
                    erasing,
                    values,
                    imgIdx,
                    random=random,
                    seed=seed,
                    stream=stream,
                )
            cvcuda.reformat_into(dst, inter_dst, stream=stream)

        return run_fake

    if input_kind == "Tensor":  # Tensor mode
        tensor_shape = (N, num_channels, H, W) if is_planar else (N, H, W, num_channels)
        src = create_tensor(
            tensor_shape,
            dtype,
            device_id,
            layout=layout,
            fill_mode="checkerboard",
        )
        dst = create_tensor(
            tensor_shape,
            dtype,
            device_id,
            layout=layout,
            fill_mode=0,
        )
    else:  # ImageBatchVarShape mode
        img_format = get_format_from_dtype(dtype_str, num_channels, planar=is_planar)
        src = create_image_batch_varshape(
            (N, H, W, num_channels),
            0,
            img_format,
            dtype,
            device_id,
            fill_mode="checkerboard",
        )
        dst = create_image_batch_varshape(
            (N, H, W, num_channels),
            0,
            img_format,
            dtype,
            device_id,
            fill_mode=0,
        )

    def run(launch):
        stream = get_stream(launch)
        if region_mode:
            cvcuda.erase_into(
                dst,
                src,
                0,
                0,
                region_height,
                region_width,
                region_values,
                stream=stream,
            )
        else:
            cvcuda.erase_into(
                dst,
                src,
                anchor,
                erasing,
                values,
                imgIdx,
                random=random,
                seed=seed,
                stream=stream,
            )

    return run


if __name__ == "__main__":
    run_benchmark("erase", erase)
