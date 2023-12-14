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

#include <cvcuda/OpStack.hpp>

#include <nvbench/nvbench.cuh>

template<typename T>
inline void Stack(nvbench::state &state, nvbench::type_list<T>)
try
{
    long3 shape = benchutils::GetShape<3>(state.get_string("shape"));

    using BT = typename nvcv::cuda::BaseType<T>;

    int ch = nvcv::cuda::NumElements<T>;

    state.add_global_memory_reads(shape.x * shape.y * shape.z * sizeof(T));
    state.add_global_memory_writes(shape.x * shape.y * shape.z * sizeof(T));

    cvcuda::Stack op;

    // clang-format off

    nvcv::TensorBatch src(nvcv::TensorBatch::CalcRequirements(shape.x));
    nvcv::Tensor dst({{shape.x, shape.y, shape.z, ch}, "NHWC"}, benchutils::GetDataType<BT>());

    for (int i = 0 ; i < shape.x; i++)
    {
        nvcv::Tensor srcIn({{shape.y, shape.z, ch}, "HWC"}, benchutils::GetDataType<BT>());
        benchutils::FillTensor<BT>(srcIn, benchutils::RandomValues<BT>());
        src.pushBack(srcIn);
    }

    state.exec(nvbench::exec_tag::sync, [&op, &src, &dst](nvbench::launch &launch)
    {
        op(launch.get_stream(), src, dst);
    });

}
catch (const std::exception &err)
{
    state.skip(err.what());
}

// clang-format on

using StackTypes = nvbench::type_list<uchar3, uint8_t, float, uchar4>;

NVBENCH_BENCH_TYPES(Stack, NVBENCH_TYPE_AXES(StackTypes))
    .set_type_axes_names({"InOutDataType"})
    .add_string_axis("shape", {"10x1080x1920"});
