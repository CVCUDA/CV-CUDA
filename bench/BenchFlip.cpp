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

#include <cvcuda/OpFlip.hpp>

#include <nvbench/nvbench.cuh>

template<typename T>
inline void Flip(nvbench::state &state, nvbench::type_list<T>)
try
{
    long3 shape    = benchutils::GetShape<3>(state.get_string("shape"));
    long  varShape = state.get_int64("varShape");
    int   flipCode;

    if (state.get_string("flipType") == "HORIZONTAL")
    {
        flipCode = 0;
    }
    else if (state.get_string("flipType") == "VERTICAL")
    {
        flipCode = 1;
    }
    else if (state.get_string("flipType") == "BOTH")
    {
        flipCode = -1;
    }
    else
    {
        throw std::invalid_argument("Invalid flipType = " + state.get_string("flipType"));
    }

    state.add_global_memory_reads(shape.x * shape.y * shape.z * sizeof(T));
    state.add_global_memory_writes(shape.x * shape.y * shape.z * sizeof(T));

    cvcuda::Flip op;

    // clang-format off
    if (varShape < 0) // negative var shape means use Tensor
    {
        nvcv::Tensor src({{shape.x, shape.y, shape.z, 1}, "NHWC"}, benchutils::GetDataType<T>());
        nvcv::Tensor dst({{shape.x, shape.y, shape.z, 1}, "NHWC"}, benchutils::GetDataType<T>());

        benchutils::FillTensor<T>(src, benchutils::RandomValues<T>());

        state.exec(nvbench::exec_tag::sync, [&op, &src, &dst, &flipCode](nvbench::launch &launch)
        {
            op(launch.get_stream(), src, dst, flipCode);
        });
    }
    else // zero and positive var shape means use ImageBatchVarShape
    {
        nvcv::ImageBatchVarShape src(shape.x);
        nvcv::ImageBatchVarShape dst(shape.x);

        benchutils::FillImageBatch<T>(src, long2{shape.z, shape.y}, long2{varShape, varShape},
                                      benchutils::RandomValues<T>());
        dst.pushBack(src.begin(), src.end());

        nvcv::Tensor flipCodeTensor({{shape.x}, "N"}, nvcv::TYPE_S32);

        benchutils::FillTensor<int>(flipCodeTensor, [&flipCode](const long4 &){ return flipCode; });

        state.exec(nvbench::exec_tag::sync, [&op, &src, &dst, &flipCodeTensor](nvbench::launch &launch)
        {
            op(launch.get_stream(), src, dst, flipCodeTensor);
        });
    }
}
catch (const std::exception &err)
{
    state.skip(err.what());
}

// clang-format on

using FlipTypes = nvbench::type_list<uint8_t, float>;

NVBENCH_BENCH_TYPES(Flip, NVBENCH_TYPE_AXES(FlipTypes))
    .set_type_axes_names({"InOutDataType"})
    .add_string_axis("shape", {"1x1080x1920"})
    .add_int64_axis("varShape", {-1, 0})
    .add_string_axis("flipType", {"BOTH"});
