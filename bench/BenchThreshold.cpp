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

#include <cvcuda/OpThreshold.hpp>

#include <nvbench/nvbench.cuh>

template<typename T>
inline void Threshold(nvbench::state &state, nvbench::type_list<T>)
try
{
    long3 shape    = benchutils::GetShape<3>(state.get_string("shape"));
    long  varShape = state.get_int64("varShape");

    uint32_t threshType = NVCV_THRESH_BINARY | (std::is_same_v<T, uint8_t> ? NVCV_THRESH_OTSU : 0);

    state.add_global_memory_reads(shape.x * shape.y * shape.z * sizeof(T));
    state.add_global_memory_writes(shape.x * shape.y * shape.z * sizeof(T));

    cvcuda::Threshold op(threshType, shape.x);

    // clang-format off

    nvcv::Tensor thresh({{shape.x}, "N"}, nvcv::TYPE_F64);
    nvcv::Tensor maxval({{shape.x}, "N"}, nvcv::TYPE_F64);

    benchutils::FillTensor<double>(thresh, benchutils::RandomValues<T>());
    benchutils::FillTensor<double>(maxval, benchutils::RandomValues<T>());

    if (varShape < 0) // negative var shape means use Tensor
    {
        nvcv::Tensor src({{shape.x, shape.y, shape.z, 1}, "NHWC"}, benchutils::GetDataType<T>());
        nvcv::Tensor dst({{shape.x, shape.y, shape.z, 1}, "NHWC"}, benchutils::GetDataType<T>());

        benchutils::FillTensor<T>(src, benchutils::RandomValues<T>());

        state.exec(nvbench::exec_tag::sync, [&op, &src, &dst, &thresh, &maxval](nvbench::launch &launch)
        {
            op(launch.get_stream(), src, dst, thresh, maxval);
        });
    }
    else // zero and positive var shape means use ImageBatchVarShape
    {
        nvcv::ImageBatchVarShape src(shape.x);
        nvcv::ImageBatchVarShape dst(shape.x);

        benchutils::FillImageBatch<T>(src, long2{shape.z, shape.y}, long2{varShape, varShape},
                                      benchutils::RandomValues<T>());
        dst.pushBack(src.begin(), src.end());

        state.exec(nvbench::exec_tag::sync, [&op, &src, &dst, &thresh, &maxval](nvbench::launch &launch)
        {
            op(launch.get_stream(), src, dst, thresh, maxval);
        });
    }
}
catch (const std::exception &err)
{
    state.skip(err.what());
}

// clang-format on

using ThresholdTypes = nvbench::type_list<uint8_t, float>;

NVBENCH_BENCH_TYPES(Threshold, NVBENCH_TYPE_AXES(ThresholdTypes))
    .set_type_axes_names({"InOutDataType"})
    .add_string_axis("shape", {"1x1080x1920"})
    .add_int64_axis("varShape", {-1, 0});
