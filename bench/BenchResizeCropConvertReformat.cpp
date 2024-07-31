/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cvcuda/OpResizeCropConvertReformat.hpp>
#include <cvcuda/cuda_tools/TypeTraits.hpp>

#include <nvbench/nvbench.cuh>

template<typename T>
inline void ResizeCropConvertReformat(nvbench::state &state, nvbench::type_list<T>)
try
{
    long3 srcShape = benchutils::GetShape<3>(state.get_string("shape"));
    long  varShape = state.get_int64("varShape");

    NVCVInterpolationType interpType = benchutils::GetInterpolationType(state.get_string("interpolation"));

    int2 cropPos{1, 1};

    NVCVSize2D resize;

    if (state.get_string("resizeType") == "EXPAND")
    {
        resize = NVCVSize2D{(int)(srcShape.y * 2), (int)(srcShape.z * 2)};
    }
    else if (state.get_string("resizeType") == "CONTRACT")
    {
        resize = NVCVSize2D{(int)(srcShape.y / 2), (int)(srcShape.z / 2)};
    }
    else
    {
        throw std::invalid_argument("Invalid resizeType = " + state.get_string("resizeType"));
    }

    NVCVChannelManip manip;

    if (state.get_string("manip") == "NO_OP")
    {
        manip = NVCV_CHANNEL_NO_OP;
    }
    else if (state.get_string("manip") == "REVERSE")
    {
        manip = NVCV_CHANNEL_REVERSE;
    }
    else
    {
        throw std::invalid_argument("Invalid channel manipulation = " + state.get_string("manip"));
    }

    using BT = nvcv::cuda::BaseType<T>;
    long nc  = nvcv::cuda::NumElements<T>;

    long3 dstShape{srcShape.x, resize.h - cropPos.y, resize.w - cropPos.x};

    if (dstShape.y <= 0 || dstShape.z <= 0)
    {
        throw std::invalid_argument("Invalid shape and resizeType");
    }

    state.add_global_memory_reads(srcShape.x * srcShape.y * srcShape.z * sizeof(T));
    state.add_global_memory_writes(dstShape.x * dstShape.y * dstShape.z * sizeof(T));

    cvcuda::ResizeCropConvertReformat op;

    // clang-format off

    if (varShape < 0) // negative var shape means use Tensor
    {
        nvcv::Tensor src({{srcShape.x, srcShape.y, srcShape.z, nc}, "NHWC"}, benchutils::GetDataType<BT>());
        nvcv::Tensor dst({{dstShape.x, dstShape.y, dstShape.z, nc}, "NHWC"}, benchutils::GetDataType<BT>());

        benchutils::FillTensor<T>(src, benchutils::RandomValues<T>());

        state.exec(nvbench::exec_tag::sync, [&op, &src, &dst, &resize, &interpType, &cropPos, &manip](nvbench::launch &launch)
        {
            op(launch.get_stream(), src, dst, resize, interpType, cropPos, manip);
        });
    }
    else // zero and positive var shape means use ImageBatchVarShape
    {
        nvcv::ImageBatchVarShape src(srcShape.x);
        nvcv::Tensor dst({{dstShape.x, dstShape.y, dstShape.z, nc}, "NHWC"}, benchutils::GetDataType<BT>());

        benchutils::FillImageBatch<T>(src, long2{srcShape.z, srcShape.y}, long2{varShape, varShape},
                                      benchutils::RandomValues<T>());

        state.exec(nvbench::exec_tag::sync, [&op, &src, &dst, &resize, &interpType, &cropPos, &manip](nvbench::launch &launch)
        {
            op(launch.get_stream(), src, dst, resize, interpType, cropPos, manip);
        });
    }
}
catch (const std::exception &err)
{
    state.skip(err.what());
}

// clang-format on

using ResizeCropConvertReformatTypes = nvbench::type_list<uchar3>;

NVBENCH_BENCH_TYPES(ResizeCropConvertReformat, NVBENCH_TYPE_AXES(ResizeCropConvertReformatTypes))
    .set_type_axes_names({"InOutDataType"})
    .add_string_axis("shape", {"1x1080x1920"})
    .add_int64_axis("varShape", {-1, 0})
    .add_string_axis("resizeType", {"EXPAND"})
    .add_string_axis("manip", {"NO_OP"})
    .add_string_axis("interpolation", {"LINEAR"});
