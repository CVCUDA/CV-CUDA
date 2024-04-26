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

#include <cvcuda/OpBrightnessContrast.hpp>

#include <nvbench/nvbench.cuh>

template<typename T>
inline void BrightnessContrast(nvbench::state &state, nvbench::type_list<T>)
try
{
    long3 shape    = benchutils::GetShape<3>(state.get_string("shape"));
    long  varShape = state.get_int64("varShape");

    state.add_global_memory_reads(shape.x * shape.y * shape.z * sizeof(T) + shape.x * sizeof(float) * 4);
    state.add_global_memory_writes(shape.x * shape.y * shape.z * sizeof(T));

    cvcuda::BrightnessContrast op;

    // clang-format off

    nvcv::Tensor brightness({{shape.x}, "N"}, nvcv::TYPE_F32);
    nvcv::Tensor contrast({{shape.x}, "N"}, nvcv::TYPE_F32);
    nvcv::Tensor brightnessShift({{shape.x}, "N"}, nvcv::TYPE_F32);
    nvcv::Tensor contrastCenter({{shape.x}, "N"}, nvcv::TYPE_F32);

    benchutils::FillTensor<float>(brightness, benchutils::RandomValues<float>(0.f, 1.f));
    benchutils::FillTensor<float>(contrast, benchutils::RandomValues<T>());
    benchutils::FillTensor<float>(brightnessShift, benchutils::RandomValues<T>());
    benchutils::FillTensor<float>(contrastCenter, benchutils::RandomValues<T>());

    if (varShape < 0) // negative var shape means use Tensor
    {
        nvcv::Tensor src({{shape.x, shape.y, shape.z, 1}, "NHWC"}, benchutils::GetDataType<T>());
        nvcv::Tensor dst({{shape.x, shape.y, shape.z, 1}, "NHWC"}, benchutils::GetDataType<T>());

        benchutils::FillTensor<T>(src, benchutils::RandomValues<T>());

        state.exec(nvbench::exec_tag::sync,
                   [&op, &src, &dst, &brightness, &contrast, &brightnessShift, &contrastCenter]
                   (nvbench::launch &launch)
        {
            op(launch.get_stream(), src, dst, brightness, contrast, brightnessShift, contrastCenter);
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
                   [&op, &src, &dst, &brightness, &contrast, &brightnessShift, &contrastCenter]
                   (nvbench::launch &launch)
        {
            op(launch.get_stream(), src, dst, brightness, contrast, brightnessShift, contrastCenter);
        });
    }
}
catch (const std::exception &err)
{
    state.skip(err.what());
}

// clang-format on

using BrightnessContrastTypes = nvbench::type_list<uint8_t, float>;

NVBENCH_BENCH_TYPES(BrightnessContrast, NVBENCH_TYPE_AXES(BrightnessContrastTypes))
    .set_type_axes_names({"InOutDataType"})
    .add_string_axis("shape", {"1x1080x1920"})
    .add_int64_axis("varShape", {-1, 0});
