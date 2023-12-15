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

#include <cvcuda/OpErase.hpp>

#include <nvbench/nvbench.cuh>

template<typename T>
inline void Erase(nvbench::state &state, nvbench::type_list<T>)
try
{
    long3 shape    = benchutils::GetShape<3>(state.get_string("shape"));
    long  varShape = state.get_int64("varShape");
    int   numErase = static_cast<int>(state.get_int64("numErase"));

    bool random = true;
    int  seed   = 0;

    state.add_global_memory_reads(shape.x * shape.y * shape.z * sizeof(T)
                                  + shape.x * (sizeof(int2) + sizeof(int3) + sizeof(float) + sizeof(int)));
    state.add_global_memory_writes(shape.x * shape.y * shape.z * sizeof(T));

    cvcuda::Erase op(numErase);

    // clang-format off

    nvcv::Tensor anchor({{shape.x}, "N"}, nvcv::TYPE_2S32);
    nvcv::Tensor erasing({{shape.x}, "N"}, nvcv::TYPE_3S32);
    nvcv::Tensor values({{shape.x}, "N"}, nvcv::TYPE_F32);
    nvcv::Tensor imgIdx({{shape.x}, "N"}, nvcv::TYPE_S32);

    benchutils::FillTensor<int2>(anchor, [](const long4 &){ return int2{0, 0}; });
    benchutils::FillTensor<int3>(erasing, [](const long4 &){ return int3{10, 10, 1}; });
    benchutils::FillTensor<float>(values, [](const long4 &){ return 1.f; });
    benchutils::FillTensor<int>(imgIdx, [](const long4 &){ return 0; });

    if (varShape < 0) // negative var shape means use Tensor
    {
        nvcv::Tensor src({{shape.x, shape.y, shape.z, 1}, "NHWC"}, benchutils::GetDataType<T>());
        nvcv::Tensor dst({{shape.x, shape.y, shape.z, 1}, "NHWC"}, benchutils::GetDataType<T>());

        benchutils::FillTensor<T>(src, benchutils::RandomValues<T>());

        state.exec(nvbench::exec_tag::sync,
                   [&op, &src, &dst, &anchor, &erasing, &values, &imgIdx, &random, &seed](nvbench::launch &launch)
        {
            op(launch.get_stream(), src, dst, anchor, erasing, values, imgIdx, random, seed);
        });
    }
    else // zero and positive var shape means use ImageBatchVarShape
    {
        nvcv::ImageBatchVarShape src(shape.x);
        nvcv::ImageBatchVarShape dst(shape.x);

        benchutils::FillImageBatch<T>(src, long2{shape.z, shape.y}, long2{varShape, varShape},
                                      benchutils::RandomValues<T>());
        dst.pushBack(src.begin(), src.end());

        state.exec(nvbench::exec_tag::sync,
                   [&op, &src, &dst, &anchor, &erasing, &values, &imgIdx, &random, &seed](nvbench::launch &launch)
        {
            op(launch.get_stream(), src, dst, anchor, erasing, values, imgIdx, random, seed);
        });
    }
}
catch (const std::exception &err)
{
    state.skip(err.what());
}

// clang-format on

using EraseTypes = nvbench::type_list<uint8_t, float>;

NVBENCH_BENCH_TYPES(Erase, NVBENCH_TYPE_AXES(EraseTypes))
    .set_type_axes_names({"InOutDataType"})
    .add_string_axis("shape", {"1x1080x1920"})
    .add_int64_axis("varShape", {0})
    .add_int64_axis("numErase", {3});
