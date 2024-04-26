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

#include <cvcuda/OpAdaptiveThreshold.hpp>

#include <nvbench/nvbench.cuh>

template<typename T>
inline void AdaptiveThreshold(nvbench::state &state, nvbench::type_list<T>)
try
{
    long3 shape     = benchutils::GetShape<3>(state.get_string("shape"));
    long  varShape  = state.get_int64("varShape");
    int   blockSize = static_cast<int>(state.get_int64("blockSize"));

    NVCVThresholdType         threshType = NVCV_THRESH_BINARY;
    NVCVAdaptiveThresholdType adaptType  = NVCV_ADAPTIVE_THRESH_GAUSSIAN_C;

    double maxValue = 123.;
    double c        = -2.3;

    state.add_global_memory_reads(shape.x * shape.y * shape.z * sizeof(T));
    state.add_global_memory_writes(shape.x * shape.y * shape.z * sizeof(T));

    cvcuda::AdaptiveThreshold op(blockSize, shape.x);

    // clang-format off

    if (varShape < 0) // negative var shape means use Tensor
    {
        nvcv::Tensor src({{shape.x, shape.y, shape.z, 1}, "NHWC"}, benchutils::GetDataType<T>());
        nvcv::Tensor dst({{shape.x, shape.y, shape.z, 1}, "NHWC"}, benchutils::GetDataType<T>());

        benchutils::FillTensor<T>(src, benchutils::RandomValues<T>());

        state.exec(nvbench::exec_tag::sync,
                   [&op, &src, &dst, &maxValue, &adaptType, &threshType, &blockSize, &c](nvbench::launch &launch)
        {
            op(launch.get_stream(), src, dst, maxValue, adaptType, threshType, blockSize, c);
        });
    }
    else // zero and positive var shape means use ImageBatchVarShape
    {
        nvcv::ImageBatchVarShape src(shape.x);
        nvcv::ImageBatchVarShape dst(shape.x);

        benchutils::FillImageBatch<T>(src, long2{shape.z, shape.y}, long2{varShape, varShape},
                                      benchutils::RandomValues<T>());
        dst.pushBack(src.begin(), src.end());

        nvcv::Tensor maxValueTensor({{shape.x}, "N"}, nvcv::TYPE_F64);
        nvcv::Tensor blockSizeTensor({{shape.x}, "N"}, nvcv::TYPE_S32);
        nvcv::Tensor cTensor({{shape.x}, "N"}, nvcv::TYPE_F64);

        benchutils::FillTensor<double>(maxValueTensor, [&maxValue](const long4 &){ return maxValue; });
        benchutils::FillTensor<int>(maxValueTensor, [&blockSize](const long4 &){ return blockSize; });
        benchutils::FillTensor<double>(cTensor, [&c](const long4 &){ return c; });

        state.exec(nvbench::exec_tag::sync,
                   [&op, &src, &dst, &maxValueTensor, &adaptType, &threshType, &blockSizeTensor, &cTensor]
                   (nvbench::launch &launch)
        {
            op(launch.get_stream(), src, dst, maxValueTensor, adaptType, threshType, blockSizeTensor, cTensor);
        });
    }
}
catch (const std::exception &err)
{
    state.skip(err.what());
}

// clang-format on

using AdaptiveThresholdTypes = nvbench::type_list<uint8_t>;

NVBENCH_BENCH_TYPES(AdaptiveThreshold, NVBENCH_TYPE_AXES(AdaptiveThresholdTypes))
    .set_type_axes_names({"InOutDataType"})
    .add_string_axis("shape", {"1x1080x1920"})
    .add_int64_axis("varShape", {-1, 0})
    .add_int64_axis("blockSize", {7});
