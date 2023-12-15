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

#include <cvcuda/OpGaussian.hpp>

#include <nvbench/nvbench.cuh>

template<typename T>
inline void Gaussian(nvbench::state &state, nvbench::type_list<T>)
try
{
    long3  shape    = benchutils::GetShape<3>(state.get_string("shape"));
    long   varShape = state.get_int64("varShape");
    double sigma    = state.get_float64("sigma");

    NVCVBorderType borderType = benchutils::GetBorderType(state.get_string("border"));

    int  kernelSize = (int)std::round(sigma * (std::is_same_v<nvcv::cuda::BaseType<T>, uint8_t> ? 3 : 4) * 2 + 1) | 1;
    int2 ksize2{kernelSize, kernelSize};

    nvcv::Size2D kernelSize2{kernelSize, kernelSize};
    double2      sigma2{sigma, sigma};

    state.add_global_memory_reads(shape.x * shape.y * shape.z * sizeof(T));
    state.add_global_memory_writes(shape.x * shape.y * shape.z * sizeof(T));

    cvcuda::Gaussian op(kernelSize2, shape.x);

    // clang-format off

    if (varShape < 0) // negative var shape means use Tensor
    {
        nvcv::Tensor src({{shape.x, shape.y, shape.z, 1}, "NHWC"}, benchutils::GetDataType<T>());
        nvcv::Tensor dst({{shape.x, shape.y, shape.z, 1}, "NHWC"}, benchutils::GetDataType<T>());

        benchutils::FillTensor<T>(src, benchutils::RandomValues<T>());

        state.exec(nvbench::exec_tag::sync,
                   [&op, &src, &dst, &kernelSize2, &sigma2, &borderType](nvbench::launch &launch)
        {
            op(launch.get_stream(), src, dst, kernelSize2, sigma2, borderType);
        });
    }
    else // zero and positive var shape means use ImageBatchVarShape
    {
        nvcv::ImageBatchVarShape src(shape.x);
        nvcv::ImageBatchVarShape dst(shape.x);

        benchutils::FillImageBatch<T>(src, long2{shape.z, shape.y}, long2{varShape, varShape},
                                      benchutils::RandomValues<T>());
        dst.pushBack(src.begin(), src.end());

        nvcv::Tensor kernelSizeTensor({{shape.x}, "N"}, nvcv::TYPE_2S32);
        nvcv::Tensor sigmaTensor({{shape.x}, "N"}, nvcv::TYPE_2F64);

        benchutils::FillTensor<int2>(kernelSizeTensor, [&ksize2](const long4 &){ return ksize2; });
        benchutils::FillTensor<double2>(sigmaTensor, [&sigma2](const long4 &){ return sigma2; });

        state.exec(nvbench::exec_tag::sync,
                   [&op, &src, &dst, &kernelSizeTensor, &sigmaTensor, &borderType](nvbench::launch &launch)
        {
            op(launch.get_stream(), src, dst, kernelSizeTensor, sigmaTensor, borderType);
        });
    }
}
catch (const std::exception &err)
{
    state.skip(err.what());
}

// clang-format on

using GaussianTypes = nvbench::type_list<uint8_t, float>;

NVBENCH_BENCH_TYPES(Gaussian, NVBENCH_TYPE_AXES(GaussianTypes))
    .set_type_axes_names({"InOutDataType"})
    .add_string_axis("shape", {"1x1080x1920"})
    .add_int64_axis("varShape", {-1})
    .add_float64_axis("sigma", {1.2})
    .add_string_axis("border", {"REFLECT"});
