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
#include "ops/generated/BenchMorphologyConfig.hpp"

#include <cvcuda/OpMorphology.hpp>
#include <cvcuda/OpReformat.hpp>

#include <nvbench/nvbench.cuh>

#include <functional>

namespace {

NVCVMorphologyType GetMorphologyType(const std::string &morphTypeStr)
{
    if (morphTypeStr == "ERODE")
    {
        return NVCV_ERODE;
    }
    if (morphTypeStr == "DILATE")
    {
        return NVCV_DILATE;
    }
    if (morphTypeStr == "OPEN")
    {
        return NVCV_OPEN;
    }
    if (morphTypeStr == "CLOSE")
    {
        return NVCV_CLOSE;
    }
    throw std::invalid_argument("Unexpected morphology type = " + morphTypeStr);
}

bool NeedsWorkspace(NVCVMorphologyType morphType, int iteration)
{
    return morphType == NVCV_OPEN || morphType == NVCV_CLOSE || iteration > 1;
}

void AddMemoryCounters(nvbench::state &state, int3 shape, long elementSize, int channels, int bwIteration,
                       bool isFakePlanar)
{
    const long bytes = static_cast<long>(shape.x) * shape.y * shape.z * channels * elementSize * bwIteration;
    const long scale = isFakePlanar ? 3 : 1;

    state.add_global_memory_reads(scale * bytes);
    state.add_global_memory_writes(scale * bytes);
}

template<typename BT>
nvcv::Tensor MakeTensor(int3 shape, int channels, bool isPlanar)
{
    return isPlanar ? nvcv::Tensor(
               {
                   {shape.x, channels, shape.y, shape.z},
                   "NCHW"
    },
               benchutils::GetDataType<BT>())
                    : nvcv::Tensor({{shape.x, shape.y, shape.z, channels}, "NHWC"}, benchutils::GetDataType<BT>());
}

template<typename BT>
void RunFakePlanarTensor(nvbench::state &state, cvcuda::Morphology &op, int3 shape, int channels, bool needsWorkspace,
                         NVCVMorphologyType morphType, nvcv::Size2D mask, int2 anchor, int iteration,
                         NVCVBorderType borderType)
{
    nvcv::Tensor src(
        {
            {shape.x, channels, shape.y, shape.z},
            "NCHW"
    },
        benchutils::GetDataType<BT>());
    nvcv::Tensor interSrc(
        {
            {shape.x, shape.y, shape.z, channels},
            "NHWC"
    },
        benchutils::GetDataType<BT>());
    nvcv::Tensor interDst(
        {
            {shape.x, shape.y, shape.z, channels},
            "NHWC"
    },
        benchutils::GetDataType<BT>());
    nvcv::Tensor dst(
        {
            {shape.x, channels, shape.y, shape.z},
            "NCHW"
    },
        benchutils::GetDataType<BT>());

    benchutils::FillTensor<BT>(src, benchutils::CheckerboardValues<BT>());

    nvcv::Tensor workspace{nullptr};
    if (needsWorkspace)
    {
        workspace = nvcv::Tensor(
            {
                {shape.x, shape.y, shape.z, channels},
                "NHWC"
        },
            benchutils::GetDataType<BT>());
    }

    const nvcv::OptionalTensorConstRef workspaceRef
        = workspace ? nvcv::OptionalTensorConstRef{std::cref(workspace)} : nvcv::OptionalTensorConstRef{nvcv::NullOpt};

    cvcuda::Reformat reformatOp;

    benchutils::warmup_and_exec(state, BENCH_MORPHOLOGY_WARMUP_ITERATIONS,
                                [&op, &reformatOp, &src, &interSrc, &interDst, &dst, &workspaceRef, morphType, mask,
                                 anchor, iteration, borderType](cudaStream_t s)
                                {
                                    reformatOp(s, src, interSrc);
                                    op(s, interSrc, interDst, workspaceRef, morphType, mask, anchor, iteration,
                                       borderType);
                                    reformatOp(s, interDst, dst);
                                });
}

template<typename T, typename BT>
void RunTensor(nvbench::state &state, cvcuda::Morphology &op, int3 shape, int channels, bool isPlanar,
               bool isFakePlanar, bool needsWorkspace, NVCVMorphologyType morphType, nvcv::Size2D mask, int2 anchor,
               int iteration, NVCVBorderType borderType)
{
    if (isFakePlanar)
    {
        RunFakePlanarTensor<BT>(state, op, shape, channels, needsWorkspace, morphType, mask, anchor, iteration,
                                borderType);
        return;
    }

    nvcv::Tensor src = MakeTensor<BT>(shape, channels, isPlanar);
    nvcv::Tensor dst = MakeTensor<BT>(shape, channels, isPlanar);

    benchutils::FillTensor<BT>(src, benchutils::CheckerboardValues<BT>());

    nvcv::Tensor workspace{nullptr};
    if (needsWorkspace)
    {
        workspace = MakeTensor<BT>(shape, channels, isPlanar);
    }

    const nvcv::OptionalTensorConstRef workspaceRef
        = workspace ? nvcv::OptionalTensorConstRef{std::cref(workspace)} : nvcv::OptionalTensorConstRef{nvcv::NullOpt};

    benchutils::warmup_and_exec(
        state, BENCH_MORPHOLOGY_WARMUP_ITERATIONS,
        [&op, &src, &dst, &workspaceRef, morphType, mask, anchor, iteration, borderType](cudaStream_t s)
        { op(s, src, dst, workspaceRef, morphType, mask, anchor, iteration, borderType); });
}

template<typename T>
void RunVarShape(nvbench::state &state, cvcuda::Morphology &op, int3 shape, bool isPlanar, bool needsWorkspace,
                 NVCVMorphologyType morphType, nvcv::Size2D mask, int2 anchor, int iteration, NVCVBorderType borderType)
{
    nvcv::ImageBatchVarShape src(shape.x);
    nvcv::ImageBatchVarShape dst(shape.x);

    if (isPlanar)
    {
        benchutils::FillPlanarImageBatch<T>(src, long2{shape.z, shape.y}, long2{0, 0});
        benchutils::FillPlanarImageBatchLike<T>(dst, src);
    }
    else
    {
        benchutils::FillImageBatch<T>(src, long2{shape.z, shape.y}, long2{0, 0}, benchutils::CheckerboardValues<T>());
        // Use FillImageBatchLike to ensure dst has same per-sample shapes as src.
        benchutils::FillImageBatchLike<T>(dst, src, [](const long4_16a &) { return T{0}; });
    }

    nvcv::Tensor maskTensor({{shape.x}, "N"}, nvcv::TYPE_2S32);
    nvcv::Tensor anchorTensor({{shape.x}, "N"}, nvcv::TYPE_2S32);

    benchutils::FillTensor<int2>(maskTensor, [mask](const long4_16a &) { return int2{mask.w, mask.h}; });
    benchutils::FillTensor<int2>(anchorTensor, [anchor](const long4_16a &) { return anchor; });

    nvcv::ImageBatchVarShape workspace{nullptr};
    if (needsWorkspace)
    {
        workspace = nvcv::ImageBatchVarShape(shape.x);
        if (isPlanar)
        {
            benchutils::FillPlanarImageBatchLike<T>(workspace, src);
        }
        else
        {
            // Create separate workspace images with no memory aliasing with dst.
            benchutils::FillImageBatch<T>(workspace, long2{shape.z, shape.y}, long2{0, 0},
                                          [](const long4_16a &) { return T{0}; });
        }
    }

    const nvcv::OptionalImageBatchVarShapeConstRef workspaceRef
        = workspace ? nvcv::OptionalImageBatchVarShapeConstRef{std::cref(workspace)}
                    : nvcv::OptionalImageBatchVarShapeConstRef{nvcv::NullOpt};

    benchutils::warmup_and_exec(
        state, BENCH_MORPHOLOGY_WARMUP_ITERATIONS,
        [&op, &src, &dst, &workspaceRef, morphType, &maskTensor, &anchorTensor, iteration, borderType](cudaStream_t s)
        { op(s, src, dst, workspaceRef, morphType, maskTensor, anchorTensor, iteration, borderType); });
}

} // namespace

template<typename T>
inline void morphology(nvbench::state &state, nvbench::type_list<T>)
try
{
    int3                        shape     = benchutils::GetShape<3, int3>(state.get_string("shape"));
    auto                        layout    = benchutils::GetStringParam(state, "layout", "NHWC");
    const benchutils::InputKind inputKind = benchutils::GetInputKind(state.get_string("inputKind"));
    int                         iteration = benchutils::GetIntParam<int>(state, "iteration");
    int2 kernelSize = nvcv::cuda::StaticCast<int>(benchutils::GetShape<2>(state.get_string("kernelSize")));

    NVCVBorderType     borderType = benchutils::GetBorderType(state.get_string("border"));
    NVCVMorphologyType morphType  = GetMorphologyType(state.get_string("morphType"));

    nvcv::Size2D mask{kernelSize.x, kernelSize.y};
    int2         anchor{-1, -1};

    int bwIteration = (morphType == NVCV_OPEN || morphType == NVCV_CLOSE || iteration > 1) ? 2 * iteration : iteration;

    const bool needsWorkspace = NeedsWorkspace(morphType, iteration);
    const bool isPlanar       = layout == "NCHW";
    const bool isFakePlanar   = layout == "NCHW_FAKE";
    if (layout != "NHWC" && layout != "NCHW" && layout != "NCHW_FAKE")
    {
        state.skip("Morphology benchmark supports only NHWC, NCHW, and NCHW_FAKE layouts");
        return;
    }
    if (isFakePlanar && inputKind == benchutils::InputKind::VarShape)
    {
        state.skip("Fake-planar (NCHW_FAKE) Morphology benchmark is tensor-only");
        return;
    }

    using BT = typename nvcv::cuda::BaseType<T>;

    int ch = nvcv::cuda::NumElements<T>;

    AddMemoryCounters(state, shape, sizeof(BT), ch, bwIteration, isFakePlanar);

    cvcuda::Morphology op;

    // clang-format off

    if (inputKind == benchutils::InputKind::Tensor)
    {
        RunTensor<T, BT>(state, op, shape, ch, isPlanar, isFakePlanar, needsWorkspace, morphType, mask, anchor,
                         iteration, borderType);
    }
    else // zero and positive var shape means use ImageBatchVarShape
    {
        RunVarShape<T>(state, op, shape, isPlanar, needsWorkspace, morphType, mask, anchor, iteration, borderType);
    }
}
CVCUDA_BENCH_SKIP_ERRORS(state)

// clang-format on

// Use auto-generated type list from bench_params.json

NVBENCH_BENCH_TYPES(morphology, NVBENCH_TYPE_AXES(BENCH_MORPHOLOGY_TYPES))
BENCH_MORPHOLOGY_AXES;
