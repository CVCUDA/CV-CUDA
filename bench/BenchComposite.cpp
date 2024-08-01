/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cvcuda/OpComposite.hpp>

#include <nvbench/nvbench.cuh>

template<typename T, typename M = uint8_t>
inline void Composite(nvbench::state &state, nvbench::type_list<T>)
try
{
    long3 shape    = benchutils::GetShape<3>(state.get_string("shape"));
    long  varShape = state.get_int64("varShape");

    using BT = typename nvcv::cuda::BaseType<T>;

    int ch = nvcv::cuda::NumElements<T>;

    state.add_global_memory_reads(shape.x * shape.y * shape.z * (sizeof(T) * 2 + sizeof(M)));
    state.add_global_memory_writes(shape.x * shape.y * shape.z * sizeof(T));

    cvcuda::Composite op;

    // clang-format off

    if (varShape < 0) // negative var shape means use Tensor
    {
        nvcv::Tensor fg({{shape.x, shape.y, shape.z, ch}, "NHWC"}, benchutils::GetDataType<T>());
        nvcv::Tensor bg({{shape.x, shape.y, shape.z, ch}, "NHWC"}, benchutils::GetDataType<T>());
        nvcv::Tensor mask({{shape.x, shape.y, shape.z, 1}, "NHWC"}, benchutils::GetDataType<M>());
        nvcv::Tensor dst({{shape.x, shape.y, shape.z, ch}, "NHWC"}, benchutils::GetDataType<T>());

        benchutils::FillTensor<T>(fg, benchutils::RandomValues<T>());
        benchutils::FillTensor<T>(bg, benchutils::RandomValues<T>());
        benchutils::FillTensor<M>(mask, [](const long4 &){ return 1; });

        state.exec(nvbench::exec_tag::sync, [&op, &fg, &bg, &mask, &dst](nvbench::launch &launch)
        {
            op(launch.get_stream(), fg, bg, mask, dst);
        });
    }
    else // zero and positive var shape means use ImageBatchVarShape
    {
        nvcv::ImageBatchVarShape fg(shape.x);
        nvcv::ImageBatchVarShape bg(shape.x);
        nvcv::ImageBatchVarShape mask(shape.x);
        nvcv::ImageBatchVarShape dst(shape.x);

        benchutils::FillImageBatch<T>(fg, long2{shape.z, shape.y}, long2{varShape, varShape},
                                      benchutils::RandomValues<T>());
        bg.pushBack(fg.begin(), fg.end());
        dst.pushBack(fg.begin(), fg.end());

        benchutils::FillImageBatch<M>(mask, long2{shape.z, shape.y}, long2{varShape, varShape},
                                      [](const long4 &){ return 1; });

        state.exec(nvbench::exec_tag::sync, [&op, &fg, &bg, &mask, &dst](nvbench::launch &launch)
        {
            op(launch.get_stream(), fg, bg, mask, dst);
        });
    }
}
catch (const std::exception &err)
{
    state.skip(err.what());
}

// clang-format on

using CompositeTypes = nvbench::type_list<uchar3>;

NVBENCH_BENCH_TYPES(Composite, NVBENCH_TYPE_AXES(CompositeTypes))
    .set_type_axes_names({"InOutDataType"})
    .add_string_axis("shape", {"1x1080x1920"})
    .add_int64_axis("varShape", {-1, 0});
