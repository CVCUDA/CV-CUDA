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

#include <cvcuda/OpGammaContrast.hpp>

#include <nvbench/nvbench.cuh>

template<typename T>
inline void GammaContrast(nvbench::state &state, nvbench::type_list<T>)
try
{
    long3 shape    = benchutils::GetShape<3>(state.get_string("shape"));
    long  varShape = state.get_int64("varShape");

    int ch = nvcv::cuda::NumElements<T>;

    state.add_global_memory_reads(shape.x * shape.y * shape.z * sizeof(T));
    state.add_global_memory_writes(shape.x * shape.y * shape.z * sizeof(T));

    cvcuda::GammaContrast op(shape.x, ch);

    // clang-format off

    nvcv::Tensor gamma({{shape.x * ch}, "N"}, nvcv::TYPE_F32);

    benchutils::FillTensor<float>(gamma, benchutils::RandomValues<float>(.5f, 1.f));

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

        state.exec(nvbench::exec_tag::sync, [&op, &src, &dst, &gamma](nvbench::launch &launch)
        {
            op(launch.get_stream(), src, dst, gamma);
        });
    }
}
catch (const std::exception &err)
{
    state.skip(err.what());
}

// clang-format on

using GammaContrastTypes = nvbench::type_list<uint8_t, uchar4>;

NVBENCH_BENCH_TYPES(GammaContrast, NVBENCH_TYPE_AXES(GammaContrastTypes))
    .set_type_axes_names({"InOutDataType"})
    .add_string_axis("shape", {"1x1080x1920"})
    .add_int64_axis("varShape", {0});
