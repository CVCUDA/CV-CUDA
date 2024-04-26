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

#include <cvcuda/OpMorphology.hpp>

#include <nvbench/nvbench.cuh>

template<typename T>
inline void Morphology(nvbench::state &state, nvbench::type_list<T>)
try
{
    long3 shape      = benchutils::GetShape<3>(state.get_string("shape"));
    long  varShape   = state.get_int64("varShape");
    int   iteration  = static_cast<int>(state.get_int64("iteration"));
    int2  kernelSize = nvcv::cuda::StaticCast<int>(benchutils::GetShape<2>(state.get_string("kernelSize")));

    NVCVBorderType borderType = benchutils::GetBorderType(state.get_string("border"));

    NVCVMorphologyType morphType;

    if (state.get_string("morphType") == "ERODE")
    {
        morphType = NVCV_ERODE;
    }
    else if (state.get_string("morphType") == "DILATE")
    {
        morphType = NVCV_DILATE;
    }
    else if (state.get_string("morphType") == "OPEN")
    {
        morphType = NVCV_OPEN;
    }
    else if (state.get_string("morphType") == "CLOSE")
    {
        morphType = NVCV_CLOSE;
    }

    nvcv::Size2D mask{kernelSize.x, kernelSize.y};
    int2         anchor{-1, -1};

    int bwIteration = (morphType == NVCV_OPEN || morphType == NVCV_CLOSE || iteration > 1) ? 2 * iteration : iteration;

    state.add_global_memory_reads(shape.x * shape.y * shape.z * sizeof(T) * bwIteration);
    state.add_global_memory_writes(shape.x * shape.y * shape.z * sizeof(T) * bwIteration);

    cvcuda::Morphology op;

    // clang-format off

    if (varShape < 0) // negative var shape means use Tensor
    {
        nvcv::Tensor src({{shape.x, shape.y, shape.z, 1}, "NHWC"}, benchutils::GetDataType<T>());
        nvcv::Tensor dst({{shape.x, shape.y, shape.z, 1}, "NHWC"}, benchutils::GetDataType<T>());

        benchutils::FillTensor<T>(src, benchutils::RandomValues<T>());

        nvcv::Tensor workspace{nullptr};

        if (morphType == NVCV_OPEN || morphType == NVCV_CLOSE || iteration > 1)
        {
            workspace = nvcv::Tensor({{shape.x, shape.y, shape.z, 1}, "NHWC"}, benchutils::GetDataType<T>());
        }

        state.exec(nvbench::exec_tag::sync,
                   [&op, &src, &dst, &workspace, &morphType, &mask, &anchor, &iteration, &borderType]
                   (nvbench::launch &launch)
        {
            op(launch.get_stream(), src, dst, workspace, morphType, mask, anchor, iteration, borderType);
        });
    }
    else // zero and positive var shape means use ImageBatchVarShape
    {
        nvcv::ImageBatchVarShape src(shape.x);
        nvcv::ImageBatchVarShape dst(shape.x);

        benchutils::FillImageBatch<T>(src, long2{shape.z, shape.y}, long2{varShape, varShape},
                                      benchutils::RandomValues<T>());
        dst.pushBack(src.begin(), src.end());

        nvcv::Tensor maskTensor({{shape.x}, "N"}, nvcv::TYPE_2S32);
        nvcv::Tensor anchorTensor({{shape.x}, "N"}, nvcv::TYPE_2S32);

        benchutils::FillTensor<int2>(maskTensor, [&mask](const long4 &){ return int2{mask.w, mask.h}; });
        benchutils::FillTensor<int2>(anchorTensor, [&anchor](const long4 &){ return anchor; });

        nvcv::ImageBatchVarShape workspace{nullptr};

        if (morphType == NVCV_OPEN || morphType == NVCV_CLOSE || iteration > 1)
        {
            workspace = nvcv::ImageBatchVarShape(shape.x);

            workspace.pushBack(dst.begin(), dst.end());
        }

        state.exec(nvbench::exec_tag::sync,
                   [&op, &src, &dst, &workspace, &morphType, &maskTensor, &anchorTensor, &iteration, &borderType]
                   (nvbench::launch &launch)
        {
            op(launch.get_stream(), src, dst, workspace, morphType, maskTensor, anchorTensor, iteration, borderType);
        });
    }
}
catch (const std::exception &err)
{
    state.skip(err.what());
}

// clang-format on

using MorphologyTypes = nvbench::type_list<uint8_t, float>;

NVBENCH_BENCH_TYPES(Morphology, NVBENCH_TYPE_AXES(MorphologyTypes))
    .set_type_axes_names({"InOutDataType"})
    .add_string_axis("shape", {"1x1080x1920"})
    .add_int64_axis("varShape", {-1, 0})
    .add_int64_axis("iteration", {1})
    .add_string_axis("kernelSize", {"3x3"})
    .add_string_axis("morphType", {"ERODE", "DILATE", "OPEN", "CLOSE"})
    .add_string_axis("border", {"REPLICATE"});
