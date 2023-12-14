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

#include "BenchMinMaxLoc.hpp"

#include <cvcuda/OpMinMaxLoc.hpp>

#include <nvbench/nvbench.cuh>

template<typename T>
inline void MinMaxLoc(nvbench::state &state, nvbench::type_list<T>)
try
{
    long3 shape    = benchutils::GetShape<3>(state.get_string("shape"));
    long  varShape = state.get_int64("varShape");
    long  maxLocs  = state.get_int64("maxLocations");

    // clang-format off

    nvcv::Tensor minVal({{shape.x}, "N"}, nvcv::TYPE_U32);
    nvcv::Tensor minLoc({{shape.x, maxLocs}, "NM"}, nvcv::TYPE_2S32);
    nvcv::Tensor numMin({{shape.x}, "N"}, nvcv::TYPE_S32);

    nvcv::Tensor maxVal({{shape.x}, "N"}, nvcv::TYPE_U32);
    nvcv::Tensor maxLoc({{shape.x, maxLocs}, "NM"}, nvcv::TYPE_2S32);
    nvcv::Tensor numMax({{shape.x}, "N"}, nvcv::TYPE_S32);

    // clang-format on

    // R/W bandwidth rationale:
    // 1 read to find min/max + 1 read to collect their locations
    // 2 writes of min/max values (U32), locations (2S32) and quantity (S32)
    state.add_global_memory_reads(shape.x * shape.y * shape.z * sizeof(T) * 2);
    state.add_global_memory_writes(shape.x * (sizeof(uint32_t) + maxLocs * sizeof(int2) + sizeof(int)) * 2);

    cvcuda::MinMaxLoc op;

    // clang-format off

    if (varShape < 0) // negative var shape means use Tensor
    {
        nvcv::Tensor src({{shape.x, shape.y, shape.z, 1}, "NHWC"}, benchutils::GetDataType<T>());

        benchutils::FillTensorWithMinMax<T>(src, maxLocs);

        state.exec(nvbench::exec_tag::sync,
                   [&op, &src, &minVal, &minLoc, &numMin, &maxVal, &maxLoc, &numMax](nvbench::launch &launch)
        {
            op(launch.get_stream(), src, minVal, minLoc, numMin, maxVal, maxLoc, numMax);
        });
    }
    else // zero and positive var shape means use ImageBatchVarShape
    {
        nvcv::ImageBatchVarShape src(shape.x);

        benchutils::FillImageBatchWithMinMax<T>(src, long2{shape.z, shape.y}, long2{varShape, varShape}, maxLocs);

        state.exec(nvbench::exec_tag::sync,
                   [&op, &src, &minVal, &minLoc, &numMin, &maxVal, &maxLoc, &numMax](nvbench::launch &launch)
        {
            op(launch.get_stream(), src, minVal, minLoc, numMin, maxVal, maxLoc, numMax);
        });
    }
}
catch (const std::exception &err)
{
    state.skip(err.what());
}

// clang-format on

using MinMaxLocTypes = nvbench::type_list<nvbench::uint8_t, nvbench::uint32_t>;

NVBENCH_BENCH_TYPES(MinMaxLoc, NVBENCH_TYPE_AXES(MinMaxLocTypes))
    .set_type_axes_names({"InOutDataType"})
    .add_string_axis("shape", {"1x1080x1920"})
    .add_int64_axis("varShape", {-1})
    .add_int64_axis("maxLocations", {100000});
