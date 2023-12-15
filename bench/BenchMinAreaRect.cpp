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

#include <cvcuda/OpMinAreaRect.hpp>

#include <nvbench/nvbench.cuh>

template<typename T>
inline void MinAreaRect(nvbench::state &state, nvbench::type_list<T>)
try
{
    long2 shape    = benchutils::GetShape<2>(state.get_string("shape"));
    long  varShape = state.get_int64("varShape");

    state.add_global_memory_reads(shape.x * shape.y * sizeof(T));
    state.add_global_memory_writes(shape.x * 8 * sizeof(float) + shape.x * sizeof(int));

    cvcuda::MinAreaRect op(shape.x);

    // clang-format off

    if (varShape < 0) // negative var shape means use Tensor
    {
        nvcv::Tensor src({{shape.x, shape.y, 2}, "NWC"}, benchutils::GetDataType<T>());
        nvcv::Tensor dst({{shape.x, 8}, "NW"}, nvcv::TYPE_F32);
        nvcv::Tensor points({{1, shape.x}, "NW"}, nvcv::TYPE_S32);

        benchutils::FillTensor<T>(src, benchutils::RandomValues<T>());
        benchutils::FillTensor<float>(dst, benchutils::RandomValues<float>(0.f, 1.f));
        benchutils::FillTensor<int>(points, benchutils::RandomValues<int>(10, 100));

        state.exec(nvbench::exec_tag::sync, [&op, &src, &dst, &points, &shape](nvbench::launch &launch)
        {
            op(launch.get_stream(), src, dst, points, shape.x);
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

using MinAreaRectTypes = nvbench::type_list<short>;

NVBENCH_BENCH_TYPES(MinAreaRect, NVBENCH_TYPE_AXES(MinAreaRectTypes))
    .set_type_axes_names({"InOutDataType"})
    .add_string_axis("shape", {"1x1024"})
    .add_int64_axis("varShape", {-1});
