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

#include <cvcuda/OpRandomResizedCrop.hpp>

#include <nvbench/nvbench.cuh>

template<typename T>
inline void RandomResizedCrop(nvbench::state &state, nvbench::type_list<T>)
try
{
    long3 srcShape = benchutils::GetShape<3>(state.get_string("shape"));
    long  varShape = state.get_int64("varShape");

    NVCVInterpolationType interpType = benchutils::GetInterpolationType(state.get_string("interpolation"));

    long3 dstShape;

    if (state.get_string("resizeType") == "EXPAND")
    {
        dstShape = long3{srcShape.x, srcShape.y * 2, srcShape.z * 2};
    }
    else if (state.get_string("resizeType") == "CONTRACT")
    {
        dstShape = long3{srcShape.x, srcShape.y / 2, srcShape.z / 2};
    }
    else
    {
        throw std::invalid_argument("Invalid resizeType = " + state.get_string("resizeType"));
    }

    double   minScale = 0.08;
    double   maxScale = 1.0;
    double   minRatio = 0.5;
    double   maxRatio = 2.0;
    uint32_t seed     = 1234;

    state.add_global_memory_reads(srcShape.x * srcShape.y * srcShape.z * sizeof(T));
    state.add_global_memory_writes(dstShape.x * dstShape.y * dstShape.z * sizeof(T));

    cvcuda::RandomResizedCrop op(minScale, maxScale, minRatio, maxRatio, srcShape.x, seed);

    // clang-format off

    if (varShape < 0) // negative var shape means use Tensor
    {
        nvcv::Tensor src({{srcShape.x, srcShape.y, srcShape.z, 1}, "NHWC"}, benchutils::GetDataType<T>());
        nvcv::Tensor dst({{dstShape.x, dstShape.y, dstShape.z, 1}, "NHWC"}, benchutils::GetDataType<T>());

        benchutils::FillTensor<T>(src, benchutils::RandomValues<T>());

        state.exec(nvbench::exec_tag::sync, [&op, &src, &dst, &interpType](nvbench::launch &launch)
        {
            op(launch.get_stream(), src, dst, interpType);
        });
    }
    else // zero and positive var shape means use ImageBatchVarShape
    {
        nvcv::ImageBatchVarShape src(srcShape.x);
        nvcv::ImageBatchVarShape dst(dstShape.x);

        benchutils::FillImageBatch<T>(src, long2{srcShape.z, srcShape.y}, long2{varShape, varShape},
                                      benchutils::RandomValues<T>());
        benchutils::FillImageBatch<T>(dst, long2{dstShape.z, dstShape.y}, long2{varShape, varShape},
                                      benchutils::RandomValues<T>());

        state.exec(nvbench::exec_tag::sync, [&op, &src, &dst, &interpType](nvbench::launch &launch)
        {
            op(launch.get_stream(), src, dst, interpType);
        });
    }
}
catch (const std::exception &err)
{
    state.skip(err.what());
}

// clang-format on

using RandomResizedCropTypes = nvbench::type_list<uint8_t, float>;

NVBENCH_BENCH_TYPES(RandomResizedCrop, NVBENCH_TYPE_AXES(RandomResizedCropTypes))
    .set_type_axes_names({"InOutDataType"})
    .add_string_axis("shape", {"1x1080x1920"})
    .add_int64_axis("varShape", {-1})
    .add_string_axis("resizeType", {"EXPAND"})
    .add_string_axis("interpolation", {"LINEAR"});
