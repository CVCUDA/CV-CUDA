/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "ops/generated/BenchImageAreaConfig.hpp"

#include <cvcuda/OpImageArea.hpp>

#include <nvbench/nvbench.cuh>

template<typename T>
inline void imagearea(nvbench::state &state, nvbench::type_list<T>)
try
{
    int3 shape = benchutils::GetShape<3, int3>(state.get_string("shape"));
    auto layout = benchutils::GetStringParam(state, "layout", "NHWC");

    // ImageArea is tensor-only: the operator consumes a single image tensor and produces a
    // rank-1 [N] int32 tensor, which has no ImageBatchVarShape equivalent.
    if (layout != "NHWC" && layout != "NCHW")
    {
        state.skip("ImageArea benchmark supports only NHWC and NCHW layouts");
        return;
    }

    using BT = typename nvcv::cuda::BaseType<T>;
    const int ch = nvcv::cuda::NumElements<T>;

    // Only the output is written; pixel values are never read by the operator.
    state.add_global_memory_writes(static_cast<long>(shape.x) * sizeof(int32_t));

    nvcv::Tensor src = layout == "NCHW"
                         ? nvcv::Tensor({{shape.x, ch, shape.y, shape.z}, "NCHW"}, benchutils::GetDataType<BT>())
                         : nvcv::Tensor({{shape.x, shape.y, shape.z, ch}, "NHWC"}, benchutils::GetDataType<BT>());
    nvcv::Tensor dst({{shape.x}, "N"}, nvcv::TYPE_S32);

    cvcuda::ImageArea op;

    benchutils::warmup_and_exec(state, BENCH_IMAGEAREA_WARMUP_ITERATIONS,
                                [&op, &src, &dst](cudaStream_t s) { op(s, src, dst); });
}
CVCUDA_BENCH_SKIP_ERRORS(state)

// Use auto-generated type list from bench_params.json

NVBENCH_BENCH_TYPES(imagearea, NVBENCH_TYPE_AXES(BENCH_IMAGEAREA_TYPES))
BENCH_IMAGEAREA_AXES;
