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

#include <cvcuda/OpSIFT.hpp>

#include <nvbench/nvbench.cuh>

template<typename T>
inline void SIFT(nvbench::state &state, nvbench::type_list<T>)
try
{
    long3 shape        = benchutils::GetShape<3>(state.get_string("shape"));
    long  varShape     = state.get_int64("varShape");
    int   capacity     = static_cast<int>(state.get_int64("maxCapacity"));
    int   numOctLayers = static_cast<int>(state.get_int64("numOctaveLayers"));
    float contThr      = static_cast<float>(state.get_float64("contrastThreshold"));
    float edgeThr      = static_cast<float>(state.get_float64("edgeThreshold"));
    float initSigma    = static_cast<float>(state.get_float64("initSigma"));

    NVCVSIFTFlagType flags;

    int3 maxShape;

    if (state.get_string("expandInput") == "Y")
    {
        flags    = NVCV_SIFT_USE_EXPANDED_INPUT;
        maxShape = int3{(int)shape.z * 2, (int)shape.y * 2, (int)shape.x};
    }
    else if (state.get_string("expandInput") == "N")
    {
        flags    = NVCV_SIFT_USE_ORIGINAL_INPUT;
        maxShape = int3{(int)shape.z, (int)shape.y, (int)shape.x};
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

    if (varShape < 0) // negative var shape means use Tensor
    {
        nvcv::Tensor src({{shape.x, shape.y, shape.z, 1}, "NHWC"}, nvcv::TYPE_U8);
        nvcv::Tensor dstC({{shape.x, capacity}, "NM"}, nvcv::TYPE_4F32);
        nvcv::Tensor dstM({{shape.x, capacity}, "NM"}, nvcv::TYPE_3F32);
        nvcv::Tensor dstD({{shape.x, capacity, 128}, "NMD"}, nvcv::TYPE_U8);
        nvcv::Tensor dstN({{shape.x}, "N"}, nvcv::TYPE_S32);

        benchutils::FillTensor<T>(src, benchutils::RandomValues<T>());

        state.exec(nvbench::exec_tag::sync,
                   [&op, &src, &dstC, &dstM, &dstD, &dstN, &numOctLayers, &contThr, &edgeThr, &initSigma, &flags]
                   (nvbench::launch &launch)
        {
            op(launch.get_stream(), src, dstC, dstM, dstD, dstN, numOctLayers, contThr, edgeThr, initSigma, flags);
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

using SIFTTypes = nvbench::type_list<uint8_t>;

NVBENCH_BENCH_TYPES(SIFT, NVBENCH_TYPE_AXES(SIFTTypes))
    .set_type_axes_names({"InOutDataType"})
    .add_string_axis("shape", {"1x1080x1920"})
    .add_int64_axis("varShape", {-1})
    .add_int64_axis("maxCapacity", {10000})
    .add_int64_axis("numOctaveLayers", {3})
    .add_float64_axis("contrastThreshold", {0.04})
    .add_float64_axis("edgeThreshold", {10.0})
    .add_float64_axis("initSigma", {1.6})
    .add_string_axis("expandInput", {"Y"});
