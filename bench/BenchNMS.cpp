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

#include <cvcuda/OpNonMaximumSuppression.hpp>

#include <nvbench/nvbench.cuh>

template<typename T, typename S = float, typename M = uint8_t>
inline void NMS(nvbench::state &state, nvbench::type_list<T>)
try
{
    long2 shape    = benchutils::GetShape<2>(state.get_string("shape"));
    long  varShape = state.get_int64("varShape");
    float scThr    = static_cast<float>(state.get_float64("scoreThreshold"));
    float iouThr   = static_cast<float>(state.get_float64("iouThreshold"));

    // R/W bandwidth rationale:
    // 1 read of scores (F32) to mask out lower scores boxes + 1 read of boxes (4S16) for IoU threshold
    // 2 writes of masks (U8) by score and IoU thresholds
    state.add_global_memory_reads(shape.x * shape.y * (sizeof(T) + sizeof(S)));
    state.add_global_memory_writes(shape.x * shape.y * sizeof(M) * 2);

    cvcuda::NonMaximumSuppression op;

    // clang-format off

    if (varShape < 0) // negative var shape means use Tensor
    {
        nvcv::Tensor srcBB({{shape.x, shape.y}, "NB"}, benchutils::GetDataType<T>());
        nvcv::Tensor srcSc({{shape.x, shape.y}, "NB"}, benchutils::GetDataType<S>());
        nvcv::Tensor dstMk({{shape.x, shape.y}, "NB"}, benchutils::GetDataType<M>());

        benchutils::FillTensor<T>(srcBB, benchutils::RandomValues<T>(10, 50));
        benchutils::FillTensor<S>(srcSc, benchutils::RandomValues<S>());

        state.exec(nvbench::exec_tag::sync, [&op, &srcBB, &dstMk, &srcSc, &scThr, &iouThr](nvbench::launch &launch)
        {
            op(launch.get_stream(), srcBB, dstMk, srcSc, scThr, iouThr);
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

using NMSTypes = nvbench::type_list<short4>;

NVBENCH_BENCH_TYPES(NMS, NVBENCH_TYPE_AXES(NMSTypes))
    .set_type_axes_names({"InOutDataType"})
    .add_string_axis("shape", {"1x1024", "32x1024"})
    .add_int64_axis("varShape", {-1})
    .add_float64_axis("scoreThreshold", {0.5})
    .add_float64_axis("iouThreshold", {0.75});
