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

#include <cvcuda/OpHistogram.hpp>

#include <nvbench/nvbench.cuh>

template<typename T>
inline void Histogram(nvbench::state &state, nvbench::type_list<T>)
try
{
    long3 shape    = benchutils::GetShape<3>(state.get_string("shape"));
    long  varShape = state.get_int64("varShape");

    int          numBins = 256;
    nvcv::Tensor mask{nullptr};

    state.add_global_memory_reads(shape.x * shape.y * shape.z * sizeof(T));
    state.add_global_memory_writes(numBins * sizeof(int));

    cvcuda::Histogram op;

    // clang-format off

    if (varShape < 0) // negative var shape means use Tensor
    {
        nvcv::Tensor src({{shape.x, shape.y, shape.z, 1}, "NHWC"}, benchutils::GetDataType<T>());
        nvcv::Tensor hist({{shape.x, numBins, 1}, "HWC"}, nvcv::TYPE_S32);

        benchutils::FillTensor<T>(src, benchutils::RandomValues<T>());
        benchutils::FillTensor<int>(hist, [](const long4 &){ return 0; });

        state.exec(nvbench::exec_tag::sync, [&op, &src, &mask, &hist](nvbench::launch &launch)
        {
            op(launch.get_stream(), src, mask, hist);
        });
    }
    else // zero and positive var shape means use ImageBatchVarShape
    {
        throw std::invalid_argument("ImageBatchVarShape not implemented for this operator");
    }
}
catch (const std::exception &err)
{
    state.skip(err.what());
}

// clang-format on

using HistogramTypes = nvbench::type_list<uint8_t>;

NVBENCH_BENCH_TYPES(Histogram, NVBENCH_TYPE_AXES(HistogramTypes))
    .set_type_axes_names({"InOutDataType"})
    .add_string_axis("shape", {"1x1080x1920"})
    .add_int64_axis("varShape", {-1});
