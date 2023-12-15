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

#include <cvcuda/OpConv2D.hpp>

#include <nvbench/nvbench.cuh>

template<typename T>
inline void Conv2D(nvbench::state &state, nvbench::type_list<T>)
try
{
    long3 shape      = benchutils::GetShape<3>(state.get_string("shape"));
    long  varShape   = state.get_int64("varShape");
    int2  kernelSize = nvcv::cuda::StaticCast<int>(benchutils::GetShape<2>(state.get_string("kernelSize")));

    NVCVBorderType borderType = benchutils::GetBorderType(state.get_string("border"));

    state.add_global_memory_reads(shape.x * shape.y * shape.z * sizeof(T));
    state.add_global_memory_writes(shape.x * shape.y * shape.z * sizeof(T));

    cvcuda::Conv2D op;

    // clang-format off

    nvcv::Tensor kernelAnchor({{shape.x}, "N"}, nvcv::TYPE_2S32);

    benchutils::FillTensor<int2>(kernelAnchor, [](const long4 &){ return int2{-1, -1}; });

    if (varShape < 0) // negative var shape means use Tensor
    {
        throw std::invalid_argument("Tensor not implemented for this operator");
    }
    else // zero and positive var shape means use ImageBatchVarShape
    {
        nvcv::ImageBatchVarShape src(shape.x);
        nvcv::ImageBatchVarShape dst(shape.x);
        nvcv::ImageBatchVarShape kernel(shape.x);

        benchutils::FillImageBatch<T>(src, long2{shape.z, shape.y}, long2{varShape, varShape},
                                      benchutils::RandomValues<T>());
        dst.pushBack(src.begin(), src.end());

        benchutils::FillImageBatch<float>(kernel, long2{kernelSize.x, kernelSize.y}, long2{0, 0},
                                          benchutils::RandomValues<float>(0.f, 1.f));

        state.exec(nvbench::exec_tag::sync,
                   [&op, &src, &dst, &kernel, &kernelAnchor, &borderType](nvbench::launch &launch)
        {
            op(launch.get_stream(), src, dst, kernel, kernelAnchor, borderType);
        });
    }
}
catch (const std::exception &err)
{
    state.skip(err.what());
}

// clang-format on

using Conv2DTypes = nvbench::type_list<uint8_t, float>;

NVBENCH_BENCH_TYPES(Conv2D, NVBENCH_TYPE_AXES(Conv2DTypes))
    .set_type_axes_names({"InOutDataType"})
    .add_string_axis("shape", {"1x1080x1920"})
    .add_int64_axis("varShape", {0})
    .add_string_axis("kernelSize", {"7x7"})
    .add_string_axis("border", {"REPLICATE"});
