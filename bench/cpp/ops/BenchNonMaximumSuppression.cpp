/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "../CppBenchUtils.hpp"
#include "ops/generated/BenchNonMaximumSuppressionConfig.hpp"

#include <cvcuda/OpNonMaximumSuppression.hpp>

#include <nvbench/nvbench.cuh>

template<typename T, typename S = float, typename M = uint8_t>
inline void nms(nvbench::state &state, nvbench::type_list<T>)
try
{
    int2 shape  = benchutils::GetShape<2, int2>(state.get_string("shape"));
    auto scThr  = static_cast<float>(state.get_float64("scoreThreshold"));
    auto iouThr = static_cast<float>(state.get_float64("iouThreshold"));

    // R/W bandwidth rationale:
    // 1 read of scores (F32) to mask out lower scores boxes + 1 read of boxes (4S16) for IoU threshold
    // 2 writes of masks (U8) by score and IoU thresholds
    state.add_global_memory_reads(shape.x * shape.y * (sizeof(T) + sizeof(S)));
    state.add_global_memory_writes(shape.x * shape.y * sizeof(M) * 2);

    cvcuda::NonMaximumSuppression op;

    // clang-format off

    // Config only specifies the Tensor variant (inputKind=Tensor); no ImageBatchVarShape support
    nvcv::Tensor srcBB({{shape.x, shape.y}, "NB"}, benchutils::GetDataType<T>());
    nvcv::Tensor srcSc({{shape.x, shape.y}, "NB"}, benchutils::GetDataType<S>());
    nvcv::Tensor dstMk({{shape.x, shape.y}, "NB"}, benchutils::GetDataType<M>());

    // srcBB: deterministic per-element [10, 50] cycle. The Python bench
    // mirrors this with `10 + (b*7 + c*11 + n*3) % 41` over (N, num_boxes, 4),
    // giving bit-identical bytes for the box coordinates.
    benchutils::FillTensor<T>(srcBB, [](const long4_16a &c) {
        long n = c.x;
        long b = c.y;
        auto v = [&b, &n](long ch) { return (short)(10 + (b * 7 + ch * 11 + n * 3) % 41); };
        return T{v(0), v(1), v(2), v(3)};
    });
    benchutils::FillTensor<S>(srcSc, benchutils::LcgValues<S>());
    // Output tensor - zero-fill (output only, no need for random values)
    benchutils::FillTensor<M>(dstMk, [](const long4_16a &) { return M{0}; });

    benchutils::warmup_and_exec(state, BENCH_NONMAXIMUMSUPPRESSION_WARMUP_ITERATIONS,
        [&op, &srcBB, &dstMk, &srcSc, &scThr, &iouThr](cudaStream_t s) { op(s, srcBB, dstMk, srcSc, scThr, iouThr); });
}
CVCUDA_BENCH_SKIP_ERRORS(state)

// clang-format on

// Use auto-generated type list from bench_params.json

NVBENCH_BENCH_TYPES(nms, NVBENCH_TYPE_AXES(BENCH_NONMAXIMUMSUPPRESSION_TYPES))
BENCH_NONMAXIMUMSUPPRESSION_AXES;
