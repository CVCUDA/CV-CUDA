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

#include <cvcuda/OpWarpAffine.hpp>

#include <nvbench/nvbench.cuh>

template<typename T>
inline void WarpAffine(nvbench::state &state, nvbench::type_list<T>)
try
{
    long3 shape    = benchutils::GetShape<3>(state.get_string("shape"));
    long  varShape = state.get_int64("varShape");

    NVCVBorderType        borderType = benchutils::GetBorderType(state.get_string("border"));
    NVCVInterpolationType interpType = benchutils::GetInterpolationType(state.get_string("interpolation"));

    int flags = interpType | ((state.get_string("inverseMap") == "Y") ? NVCV_WARP_INVERSE_MAP : 0);

    float4 borderValue{0, 0, 0, 0};

    NVCVAffineTransform transMatrix{2.f, 2.f, 0.f, 3.f, 1.f, 0.f};

    state.add_global_memory_reads(shape.x * shape.y * shape.z * sizeof(T) + 6 * sizeof(float));
    state.add_global_memory_writes(shape.x * shape.y * shape.z * sizeof(T));

    cvcuda::WarpAffine op(shape.x);

    // clang-format off

    if (varShape < 0) // negative var shape means use Tensor
    {
        nvcv::Tensor src({{shape.x, shape.y, shape.z, 1}, "NHWC"}, benchutils::GetDataType<T>());
        nvcv::Tensor dst({{shape.x, shape.y, shape.z, 1}, "NHWC"}, benchutils::GetDataType<T>());

        benchutils::FillTensor<T>(src, benchutils::RandomValues<T>());

        state.exec(nvbench::exec_tag::sync, [&op, &src, &dst, &transMatrix, &flags, &borderType, &borderValue]
                   (nvbench::launch &launch)
        {
            op(launch.get_stream(), src, dst, transMatrix, flags, borderType, borderValue);
        });
    }
    else // zero and positive var shape means use ImageBatchVarShape
    {
        nvcv::ImageBatchVarShape src(shape.x);
        nvcv::ImageBatchVarShape dst(shape.x);

        benchutils::FillImageBatch<T>(src, long2{shape.z, shape.y}, long2{varShape, varShape},
                                      benchutils::RandomValues<T>());
        dst.pushBack(src.begin(), src.end());

        nvcv::Tensor transMatrixTensor({{shape.x, 6}, "NW"}, nvcv::TYPE_F32);

        benchutils::FillTensor<float>(transMatrixTensor, [&transMatrix](const long4 &c){ return transMatrix[c.y]; });

        state.exec(nvbench::exec_tag::sync,
                   [&op, &src, &dst, &transMatrixTensor, &flags, &borderType, &borderValue](nvbench::launch &launch)
        {
            op(launch.get_stream(), src, dst, transMatrixTensor, flags, borderType, borderValue);
        });
    }
}
catch (const std::exception &err)
{
    state.skip(err.what());
}

// clang-format on

using WarpAffineTypes = nvbench::type_list<uint8_t, float>;

NVBENCH_BENCH_TYPES(WarpAffine, NVBENCH_TYPE_AXES(WarpAffineTypes))
    .set_type_axes_names({"InOutDataType"})
    .add_string_axis("shape", {"1x1080x1920"})
    .add_int64_axis("varShape", {-1})
    .add_string_axis("border", {"REFLECT"})
    .add_string_axis("interpolation", {"CUBIC"})
    .add_string_axis("inverseMap", {"Y"});
