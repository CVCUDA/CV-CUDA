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

#include <cvcuda/OpBilateralFilter.hpp>

#include <nvbench/nvbench.cuh>

template<typename T>
inline void BilateralFilter(nvbench::state &state, nvbench::type_list<T>)
try
{
    long3 shape      = benchutils::GetShape<3>(state.get_string("shape"));
    long  varShape   = state.get_int64("varShape");
    int   diameter   = static_cast<int>(state.get_int64("diameter"));
    float sigmaSpace = static_cast<float>(state.get_float64("sigmaSpace"));
    float sigmaColor = -1.f;

    NVCVBorderType borderType = benchutils::GetBorderType(state.get_string("border"));

    state.add_global_memory_reads(shape.x * shape.y * shape.z * sizeof(T));
    state.add_global_memory_writes(shape.x * shape.y * shape.z * sizeof(T));

    cvcuda::BilateralFilter op;

    // clang-format off

    if (varShape < 0) // negative var shape means use Tensor
    {
        nvcv::Tensor src({{shape.x, shape.y, shape.z, 1}, "NHWC"}, benchutils::GetDataType<T>());
        nvcv::Tensor dst({{shape.x, shape.y, shape.z, 1}, "NHWC"}, benchutils::GetDataType<T>());

        benchutils::FillTensor<T>(src, benchutils::RandomValues<T>());

        state.exec(nvbench::exec_tag::sync,
                   [&op, &src, &dst, &diameter, &sigmaColor, &sigmaSpace, &borderType](nvbench::launch &launch)
        {
            op(launch.get_stream(), src, dst, diameter, sigmaColor, sigmaSpace, borderType);
        });
    }
    else // zero and positive var shape means use ImageBatchVarShape
    {
        nvcv::ImageBatchVarShape src(shape.x);
        nvcv::ImageBatchVarShape dst(shape.x);

        benchutils::FillImageBatch<T>(src, long2{shape.z, shape.y}, long2{varShape, varShape},
                                      benchutils::RandomValues<T>());
        dst.pushBack(src.begin(), src.end());

        nvcv::Tensor diameterTensor({{shape.x}, "N"}, nvcv::TYPE_S32);
        nvcv::Tensor sigmaSpaceTensor({{shape.x}, "N"}, nvcv::TYPE_F32);
        nvcv::Tensor sigmaColorTensor({{shape.x}, "N"}, nvcv::TYPE_F32);

        benchutils::FillTensor<int>(diameterTensor, [&diameter](const long4 &){ return diameter; });
        benchutils::FillTensor<float>(sigmaSpaceTensor, [&sigmaSpace](const long4 &){ return sigmaSpace; });
        benchutils::FillTensor<float>(sigmaColorTensor, [&sigmaColor](const long4 &){ return sigmaColor; });

        state.exec(nvbench::exec_tag::sync,
                   [&op, &src, &dst, &diameterTensor, &sigmaColorTensor, &sigmaSpaceTensor, &borderType]
                   (nvbench::launch &launch)
        {
            op(launch.get_stream(), src, dst, diameterTensor, sigmaColorTensor, sigmaSpaceTensor, borderType);
        });
    }
}
catch (const std::exception &err)
{
    state.skip(err.what());
}

// clang-format on

using BilateralFilterTypes = nvbench::type_list<uint8_t, float>;

NVBENCH_BENCH_TYPES(BilateralFilter, NVBENCH_TYPE_AXES(BilateralFilterTypes))
    .set_type_axes_names({"InOutDataType"})
    .add_string_axis("shape", {"1x1080x1920"})
    .add_int64_axis("varShape", {-1})
    .add_int64_axis("diameter", {-1})
    .add_float64_axis("sigmaSpace", {1.2})
    .add_string_axis("border", {"REFLECT"});
