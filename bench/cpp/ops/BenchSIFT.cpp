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
#include "ops/generated/BenchSIFTConfig.hpp"

#include <cvcuda/OpSIFT.hpp>

#include <nvbench/nvbench.cuh>

#include <string>

template<typename T>
inline void sift(nvbench::state &state, nvbench::type_list<T>)
try
{
    int3                        shape        = benchutils::GetShape<3, int3>(state.get_string("shape"));
    const benchutils::InputKind inputKind    = benchutils::GetInputKind(state.get_string("inputKind"));
    const std::string           layout       = state.get_string("layout");
    auto                        capacity     = benchutils::GetIntParam<int>(state, "maxCapacity");
    auto                        numOctLayers = benchutils::GetIntParam<int>(state, "numOctaveLayers");
    auto                        contThr      = static_cast<float>(state.get_float64("contrastThreshold"));
    auto                        edgeThr      = static_cast<float>(state.get_float64("edgeThreshold"));
    auto                        initSigma    = static_cast<float>(state.get_float64("initSigma"));

    NVCVSIFTFlagType flags;

    int3 maxShape;

    if (layout != "NHWC" && layout != "NCHW")
    {
        throw std::invalid_argument("Invalid layout = " + layout);
    }

    if (state.get_string("expandInput") == "Y")
    {
        flags    = NVCV_SIFT_USE_EXPANDED_INPUT;
        maxShape = int3{shape.z * 2, shape.y * 2, shape.x};
    }
    else if (state.get_string("expandInput") == "N")
    {
        flags    = NVCV_SIFT_USE_ORIGINAL_INPUT;
        maxShape = int3{shape.z, shape.y, shape.x};
    }
    else
    {
        throw std::invalid_argument("Invalid expandInput = " + state.get_string("expandInput"));
    }

    // Each pyramid has shape approximately (3 + L) * N * (2 HW size) * F32
    std::size_t pyrSize = (numOctLayers + 3) * shape.x * (maxShape.x * maxShape.y * 2) * sizeof(float);

    // R/W bandwidth rationale:
    // 1 read of input (U8) to build (F32) pyramids, 1 read of Gauss and 1 read of DoG pyramids
    // 1 write of Gauss and 1 write of DoG pyramids, 1 write of 4 output data
    state.add_global_memory_reads(shape.x * shape.y * shape.z * sizeof(T) + 2 * pyrSize);
    state.add_global_memory_writes(2 * pyrSize + shape.x * sizeof(int)
                                   + shape.x * capacity * (sizeof(float4) + sizeof(float3) + 128 * sizeof(T)));

    cvcuda::SIFT op(maxShape, numOctLayers);

    // clang-format off

    if (inputKind == benchutils::InputKind::Tensor)
    {
        nvcv::Tensor src = layout == "NCHW" ? nvcv::Tensor({{shape.x, 1, shape.y, shape.z}, "NCHW"}, nvcv::TYPE_U8)
                                            : nvcv::Tensor({{shape.x, shape.y, shape.z, 1}, "NHWC"}, nvcv::TYPE_U8);
        nvcv::Tensor dstC({{shape.x, capacity}, "NM"}, nvcv::TYPE_4F32);
        nvcv::Tensor dstM({{shape.x, capacity}, "NM"}, nvcv::TYPE_3F32);
        nvcv::Tensor dstD({{shape.x, capacity, 128}, "NMD"}, nvcv::TYPE_U8);
        nvcv::Tensor dstN({{shape.x}, "N"}, nvcv::TYPE_S32);

        benchutils::FillTensor<T>(src, benchutils::LcgValues<T>());

        benchutils::warmup_and_exec(state, BENCH_SIFT_WARMUP_ITERATIONS,
            [&op, &src, &dstC, &dstM, &dstD, &dstN, &numOctLayers, &contThr, &edgeThr, &initSigma, &flags](cudaStream_t s) {
                op(s, src, dstC, dstM, dstD, dstN, numOctLayers, contThr, edgeThr, initSigma, flags);
            });
    }
    else // zero and positive var shape means use ImageBatchVarShape
    {
        throw std::invalid_argument("ImageBatchVarShape not implemented for this operator");
    }
}
CVCUDA_BENCH_SKIP_ERRORS(state)

// clang-format on

// Use auto-generated type list from bench_params.json

NVBENCH_BENCH_TYPES(sift, NVBENCH_TYPE_AXES(BENCH_SIFT_TYPES))
BENCH_SIFT_AXES;
