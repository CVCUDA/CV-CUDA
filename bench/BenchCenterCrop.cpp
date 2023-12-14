/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "BenchUtils.hpp"

#include <cvcuda/OpCenterCrop.hpp>

#include <nvbench/nvbench.cuh>

template<typename T>
inline void CenterCrop(nvbench::state &state, nvbench::type_list<T>)
try
{
    long3 srcShape = benchutils::GetShape<3>(state.get_string("shape"));
    long  varShape = state.get_int64("varShape");

    nvcv::Size2D cropSize;

    if (state.get_string("cropType") == "SAME")
    {
        cropSize = nvcv::Size2D{(int)srcShape.z, (int)srcShape.y};
    }
    else if (state.get_string("cropType") == "QUARTER")
    {
        cropSize = nvcv::Size2D{(int)srcShape.z / 2, (int)srcShape.y / 2};
    }
    else
    {
        throw std::invalid_argument("Invalid resizeType = " + state.get_string("resizeType"));
    }

    long3 dstShape{srcShape.x, cropSize.h, cropSize.w};

    state.add_global_memory_reads(dstShape.x * dstShape.y * dstShape.z * sizeof(T));
    state.add_global_memory_writes(dstShape.x * dstShape.y * dstShape.z * sizeof(T));

    cvcuda::CenterCrop op;

    // clang-format off

    if (varShape < 0) // negative var shape means use Tensor
    {
        nvcv::Tensor src({{srcShape.x, srcShape.y, srcShape.z, 1}, "NHWC"}, benchutils::GetDataType<T>());
        nvcv::Tensor dst({{dstShape.x, dstShape.y, dstShape.z, 1}, "NHWC"}, benchutils::GetDataType<T>());

        benchutils::FillTensor<T>(src, benchutils::RandomValues<T>());

        state.exec(nvbench::exec_tag::sync, [&op, &src, &dst, &cropSize](nvbench::launch &launch)
        {
            op(launch.get_stream(), src, dst, cropSize);
        });
    }
    else // zero and positive var shape means use ImageBatchVarShape
    {
        throw std::invalid_argument("ImageBatchVarShape not implemented for this operator");
    }
}
catch (const std::exception &err)
{
    state.skip(err.what());
}

// clang-format on

using CenterCropTypes = nvbench::type_list<uint8_t, float>;

NVBENCH_BENCH_TYPES(CenterCrop, NVBENCH_TYPE_AXES(CenterCropTypes))
    .set_type_axes_names({"InOutDataType"})
    .add_string_axis("shape", {"1x1080x1920"})
    .add_int64_axis("varShape", {-1})
    .add_string_axis("cropType", {"QUARTER"});
