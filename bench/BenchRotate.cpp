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

#include <cvcuda/OpRotate.hpp>

#include <nvbench/nvbench.cuh>

template<typename T>
inline void Rotate(nvbench::state &state, nvbench::type_list<T>)
try
{
    long3 shape    = benchutils::GetShape<3>(state.get_string("shape"));
    long  varShape = state.get_int64("varShape");

    NVCVInterpolationType interpType = benchutils::GetInterpolationType(state.get_string("interpolation"));

    double  angleDeg = 123.456;
    double2 shift{12.34, 12.34};

    state.add_global_memory_reads(shape.x * shape.y * shape.z * sizeof(T));
    state.add_global_memory_writes(shape.x * shape.y * shape.z * sizeof(T));

    cvcuda::Rotate op(shape.x);

    // clang-format off

    if (varShape < 0) // negative var shape means use Tensor
    {
        nvcv::Tensor src({{shape.x, shape.y, shape.z, 1}, "NHWC"}, benchutils::GetDataType<T>());
        nvcv::Tensor dst({{shape.x, shape.y, shape.z, 1}, "NHWC"}, benchutils::GetDataType<T>());

        benchutils::FillTensor<T>(src, benchutils::RandomValues<T>());

        state.exec(nvbench::exec_tag::sync, [&op, &src, &dst, &angleDeg, &shift, &interpType](nvbench::launch &launch)
        {
            op(launch.get_stream(), src, dst, angleDeg, shift, interpType);
        });
    }
    else // zero and positive var shape means use ImageBatchVarShape
    {
        nvcv::ImageBatchVarShape src(shape.x);
        nvcv::ImageBatchVarShape dst(shape.x);

        benchutils::FillImageBatch<T>(src, long2{shape.z, shape.y}, long2{varShape, varShape},
                                      benchutils::RandomValues<T>());
        dst.pushBack(src.begin(), src.end());

        nvcv::Tensor angleDegTensor({{shape.x}, "N"}, nvcv::TYPE_F64);
        nvcv::Tensor shiftTensor({{shape.x, 2}, "NW"}, nvcv::TYPE_F64);

        benchutils::FillTensor<double>(angleDegTensor, [&angleDeg](const long4 &){ return angleDeg; });
        benchutils::FillTensor<double>(shiftTensor,
                                       [&shift](const long4 &c){ return nvcv::cuda::GetElement(shift, c.y); });

        state.exec(nvbench::exec_tag::sync,
                   [&op, &src, &dst, &angleDegTensor, &shiftTensor, &interpType](nvbench::launch &launch)
        {
            op(launch.get_stream(), src, dst, angleDegTensor, shiftTensor, interpType);
        });
    }
}
catch (const std::exception &err)
{
    state.skip(err.what());
}

// clang-format on

using RotateTypes = nvbench::type_list<uint8_t, float>;

NVBENCH_BENCH_TYPES(Rotate, NVBENCH_TYPE_AXES(RotateTypes))
    .set_type_axes_names({"InOutDataType"})
    .add_string_axis("shape", {"1x1080x1920"})
    .add_int64_axis("varShape", {-1, 0})
    .add_string_axis("interpolation", {"CUBIC"});
