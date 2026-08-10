#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""CV-CUDA FindHomography operator benchmark - Python equivalent of BenchFindHomography.cpp"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cvcuda  # noqa: E402
from python_bench_utils import (  # noqa: E402
    create_stream_cache,
    get_input_kind,
    parse_shape,
    run_benchmark,
)

try:
    import cupy as cp
except ImportError:
    import numpy as cp


def fill_src_grid_replicated(num_samples, num_points, dtype=cp.float32):
    """Create a deterministic 2D point grid replicated across all batches."""
    grid_side = 1
    while grid_side * grid_side < num_points:
        grid_side += 1

    indices = cp.arange(num_points, dtype=cp.int32)
    grid_scale = dtype(2.0) / dtype(grid_side - 1)
    x = dtype(-1.0) + (indices % grid_side).astype(dtype) * grid_scale
    y = dtype(-1.0) + (indices // grid_side).astype(dtype) * grid_scale
    single_sample = cp.stack([x, y], axis=-1).reshape(-1)
    return cp.tile(single_sample, num_samples)


def fill_dst_projective_replicated(src_vec, transform, num_samples, num_points):
    """Apply a valid projective transform and replicate the result."""
    sample_size = num_points * 2
    single_sample = src_vec[:sample_size].reshape(num_points, 2)
    x = single_sample[:, 0]
    y = single_sample[:, 1]

    h00, h01, h02 = transform[0], transform[1], transform[2]
    h10, h11, h12 = transform[3], transform[4], transform[5]
    h20, h21, h22 = transform[6], transform[7], transform[8]

    # For x,y in [-1,1], this fixed transform keeps w in [0.965,1.035].
    w = h20 * x + h21 * y + h22
    x_transformed = (h00 * x + h01 * y + h02) / w
    y_transformed = (h10 * x + h11 * y + h12) / w

    single_dst = cp.stack([x_transformed, y_transformed], axis=-1).reshape(-1)
    return cp.tile(single_dst, num_samples)


def fill_tensor(tensor, vec, device_id):
    """Copy host vector to device tensor"""
    with cp.cuda.Device(device_id):
        if not isinstance(vec, cp.ndarray):
            vec = cp.array(vec)

        tensor_data = tensor.cuda()
        cp.cuda.runtime.memcpy(
            tensor_data.__cuda_array_interface__["data"][0],
            vec.data.ptr,
            vec.nbytes,
            cp.cuda.runtime.memcpyDeviceToDevice,
        )


def findhomography(state):
    """FindHomography operator benchmark matching C++ BenchFindHomography.cpp"""

    N, num_points = parse_shape(state.get_string("shape"))
    input_kind = get_input_kind(state.get_string("inputKind"))
    device_id = state.get_device()

    state.add_global_memory_reads(N * num_points * 4 * 4)
    state.add_global_memory_writes(N * 3 * 3 * 4)

    transform = cp.array(
        [1.05, 0.08, 0.15, -0.04, 0.97, -0.10, 0.015, -0.020, 1.0],
        dtype=cp.float32,
    )

    get_stream = create_stream_cache()

    if input_kind == "Tensor":
        src = cvcuda.Tensor((N, num_points), cvcuda.Type._2F32, "NW")
        dst = cvcuda.Tensor((N, num_points), cvcuda.Type._2F32, "NW")
        models = cvcuda.Tensor((N, 3, 3), cvcuda.Type.F32, "NHW")

        src_vec = fill_src_grid_replicated(N, num_points)
        dst_vec = fill_dst_projective_replicated(src_vec, transform, N, num_points)
        fill_tensor(src, src_vec, device_id)
        fill_tensor(dst, dst_vec, device_id)

        op = cvcuda.get_findhomography_operator(N, num_points)

        def run(launch):
            stream = get_stream(launch)
            cvcuda.findhomography_into_with_op(models, src, dst, op, stream=stream)

        return run

    # FindHomography's variable-shape API uses TensorBatch rather than
    # ImageBatchVarShape. Each point tensor is one sample in the batch.
    src_batch = cvcuda.TensorBatch(N)
    dst_batch = cvcuda.TensorBatch(N)
    models_batch = cvcuda.TensorBatch(N)

    src_vec = fill_src_grid_replicated(1, num_points)
    dst_vec = fill_dst_projective_replicated(src_vec, transform, 1, num_points)
    for _ in range(N):
        src = cvcuda.Tensor((1, num_points), cvcuda.Type._2F32, "NW")
        dst = cvcuda.Tensor((1, num_points), cvcuda.Type._2F32, "NW")
        model = cvcuda.Tensor((1, 3, 3), cvcuda.Type.F32, "NHW")
        fill_tensor(src, src_vec, device_id)
        fill_tensor(dst, dst_vec, device_id)
        src_batch.pushback(src)
        dst_batch.pushback(dst)
        models_batch.pushback(model)

    def run(launch):
        stream = get_stream(launch)
        cvcuda.findhomography_into(
            models=models_batch,
            srcPts=src_batch,
            dstPts=dst_batch,
            stream=stream,
        )

    return run


if __name__ == "__main__":
    run_benchmark("findhomography", findhomography)
