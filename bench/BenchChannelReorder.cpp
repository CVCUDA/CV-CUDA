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

#include <cvcuda/OpChannelReorder.hpp>

#include <nvbench/nvbench.cuh>

template<typename T>
inline void ChannelReorder(nvbench::state &state, nvbench::type_list<T>)
try
{
    long3 shape    = benchutils::GetShape<3>(state.get_string("shape"));
    long  varShape = state.get_int64("varShape");

    state.add_global_memory_reads(shape.x * shape.y * shape.z * sizeof(T));
    state.add_global_memory_writes(shape.x * shape.y * shape.z * sizeof(T));

    cvcuda::ChannelReorder op;

    // clang-format off

    nvcv::Tensor orders({{shape.x, 4}, "NC"}, nvcv::TYPE_S32);

    benchutils::FillTensor<int>(orders, benchutils::RandomValues<int>(0, nvcv::cuda::NumElements<T>));

    if (varShape < 0) // negative var shape means use Tensor
    {
        throw std::invalid_argument("Tensor not implemented for this operator");
    }
    else // zero and positive var shape means use ImageBatchVarShape
    {
        nvcv::ImageBatchVarShape src(shape.x);
        nvcv::ImageBatchVarShape dst(shape.x);

        benchutils::FillImageBatch<T>(src, long2{shape.z, shape.y}, long2{varShape, varShape},
                                      benchutils::RandomValues<T>());
        dst.pushBack(src.begin(), src.end());

        state.exec(nvbench::exec_tag::sync, [&op, &src, &dst, &orders](nvbench::launch &launch)
        {
            op(launch.get_stream(), src, dst, orders);
        });
    }
}
catch (const std::exception &err)
{
    state.skip(err.what());
}

// clang-format on

using ChannelReorderTypes = nvbench::type_list<uchar3, uchar4>;

NVBENCH_BENCH_TYPES(ChannelReorder, NVBENCH_TYPE_AXES(ChannelReorderTypes))
    .set_type_axes_names({"InOutDataType"})
    .add_string_axis("shape", {"1x1080x1920"})
    .add_int64_axis("varShape", {0});
