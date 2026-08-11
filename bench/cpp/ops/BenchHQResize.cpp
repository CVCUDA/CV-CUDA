/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "ops/generated/BenchHQResizeConfig.hpp"

#include <cvcuda/OpHQResize.hpp>
#include <cvcuda/OpReformat.hpp>

#include <nvbench/nvbench.cuh>

struct HQResizeBenchParams
{
    long3                 srcShape;
    long3                 dstShape;
    nvcv::Size2D          srcSize;
    nvcv::Size2D          dstSize;
    nvcv::DataType        dtype;
    NVCVInterpolationType interpolation;
    bool                  antialias;
    bool                  planar;
    int                   channels;
};

inline cvcuda::UniqueWorkspace AllocateHQResizeTensorWorkspace(cvcuda::HQResize &op, const HQResizeBenchParams &params)
{
    HQResizeTensorShapeI inShapeDesc{
        {params.srcSize.h, params.srcSize.w},
        2,
        params.channels
    };
    HQResizeTensorShapeI outShapeDesc{
        {params.dstSize.h, params.dstSize.w},
        2,
        params.channels
    };
    return cvcuda::AllocateWorkspace(op.getWorkspaceRequirements(static_cast<int32_t>(params.srcShape.x), inShapeDesc,
                                                                 outShapeDesc, params.interpolation,
                                                                 params.interpolation, params.antialias));
}

inline cvcuda::UniqueWorkspace AllocateHQResizeBatchWorkspace(cvcuda::HQResize &op, const HQResizeBenchParams &params)
{
    // ImageBatch and TensorBatch planar inputs are expanded internally to
    // one single-channel sample per plane. Size the workspace for those
    // N*C samples, matching the operator's planar parity tests.
    const int                         wsChannels  = params.planar ? 1 : params.channels;
    const int                         repetitions = params.planar ? params.channels : 1;
    const int                         wsSamples   = static_cast<int>(params.srcShape.x) * repetitions;
    std::vector<HQResizeTensorShapeI> inShapes;
    std::vector<HQResizeTensorShapeI> outShapes;
    inShapes.reserve(wsSamples);
    outShapes.reserve(wsSamples);
    for (int i = 0; i < wsSamples; ++i)
    {
        inShapes.push_back(HQResizeTensorShapeI{
            {params.srcSize.h, params.srcSize.w},
            2,
            wsChannels
        });
        outShapes.push_back(HQResizeTensorShapeI{
            {params.dstSize.h, params.dstSize.w},
            2,
            wsChannels
        });
    }
    HQResizeTensorShapesI inShapeDesc{inShapes.data(), wsSamples, 2, wsChannels};
    HQResizeTensorShapesI outShapeDesc{outShapes.data(), wsSamples, 2, wsChannels};
    return cvcuda::AllocateWorkspace(op.getWorkspaceRequirements(
        wsSamples, inShapeDesc, outShapeDesc, params.interpolation, params.interpolation, params.antialias));
}

template<typename T>
inline void RunFakePlanarHQResizeBench(nvbench::state &state, cvcuda::HQResize &op, const HQResizeBenchParams &params)
{
    cvcuda::UniqueWorkspace ws = AllocateHQResizeTensorWorkspace(op, params);

    // clang-format off
    nvcv::Tensor src     ({{params.srcShape.x, params.channels, params.srcShape.y, params.srcShape.z}, "NCHW"}, params.dtype);
    nvcv::Tensor interSrc({{params.srcShape.x, params.srcShape.y, params.srcShape.z, params.channels}, "NHWC"}, params.dtype);
    nvcv::Tensor interDst({{params.dstShape.x, params.dstShape.y, params.dstShape.z, params.channels}, "NHWC"}, params.dtype);
    nvcv::Tensor dst     ({{params.dstShape.x, params.channels, params.dstShape.y, params.dstShape.z}, "NCHW"}, params.dtype);
    // clang-format on

    benchutils::FillTensor<T>(src, benchutils::CheckerboardValues<T>());

    cvcuda::Reformat reformatOp;
    benchutils::warmup_and_exec(state, BENCH_HQRESIZE_WARMUP_ITERATIONS,
                                [&op, &reformatOp, &ws, &src, &interSrc, &interDst, &dst,
                                 interpolation = params.interpolation, antialias = params.antialias](cudaStream_t s)
                                {
                                    reformatOp(s, src, interSrc); // NCHW->NHWC
                                    op(s, ws.get(), interSrc, interDst, interpolation, interpolation, antialias);
                                    reformatOp(s, interDst, dst); // NHWC->NCHW
                                });
}

template<typename T>
inline void RunTensorHQResizeBench(nvbench::state &state, cvcuda::HQResize &op, const HQResizeBenchParams &params)
{
    cvcuda::UniqueWorkspace ws = AllocateHQResizeTensorWorkspace(op, params);

    // clang-format off
    nvcv::Tensor src = params.planar ? nvcv::Tensor({{params.srcShape.x, params.channels, params.srcShape.y, params.srcShape.z}, "NCHW"}, params.dtype)
                                     : nvcv::Tensor({{params.srcShape.x, params.srcShape.y, params.srcShape.z, params.channels}, "NHWC"}, params.dtype);
    nvcv::Tensor dst = params.planar ? nvcv::Tensor({{params.dstShape.x, params.channels, params.dstShape.y, params.dstShape.z}, "NCHW"}, params.dtype)
                                     : nvcv::Tensor({{params.dstShape.x, params.dstShape.y, params.dstShape.z, params.channels}, "NHWC"}, params.dtype);
    // clang-format on

    benchutils::FillTensor<T>(src, benchutils::CheckerboardValues<T>());

    benchutils::warmup_and_exec(
        state, BENCH_HQRESIZE_WARMUP_ITERATIONS,
        [&op, &ws, &src, &dst, interpolation = params.interpolation, antialias = params.antialias](cudaStream_t s)
        { op(s, ws.get(), src, dst, interpolation, interpolation, antialias); });
}

template<typename T>
inline void FillHQResizeImageBatches(nvcv::ImageBatchVarShape &src, nvcv::ImageBatchVarShape &dst,
                                     const HQResizeBenchParams &params)
{
    if (params.planar)
    {
        benchutils::FillPlanarImageBatch<T>(src, long2{params.srcShape.z, params.srcShape.y}, long2{0, 0}, true);
        benchutils::FillPlanarImageBatch<T>(dst, long2{params.dstShape.z, params.dstShape.y}, long2{0, 0}, false);
        return;
    }

    benchutils::FillImageBatch<T>(src, long2{params.srcShape.z, params.srcShape.y}, long2{0, 0},
                                  benchutils::CheckerboardValues<T>());
    benchutils::FillImageBatch<T>(dst, long2{params.dstShape.z, params.dstShape.y}, long2{0, 0},
                                  [](const long4_16a &) { return T{0}; });
}

template<typename T>
inline void RunVarShapeHQResizeBench(nvbench::state &state, cvcuda::HQResize &op, const HQResizeBenchParams &params)
{
    if (params.channels != 3)
    {
        state.skip("HQResize ImageBatchVarShape benchmark currently requires three channels");
        return;
    }

    cvcuda::UniqueWorkspace ws = AllocateHQResizeBatchWorkspace(op, params);

    nvcv::ImageBatchVarShape src(static_cast<int32_t>(params.srcShape.x));
    nvcv::ImageBatchVarShape dst(static_cast<int32_t>(params.dstShape.x));

    if constexpr (std::is_same_v<T, uint8_t>)
    {
        FillHQResizeImageBatches<uchar3>(src, dst, params);
    }
    else if constexpr (std::is_same_v<T, float>)
    {
        FillHQResizeImageBatches<float3>(src, dst, params);
    }
    else
    {
        state.skip("Unsupported HQResize ImageBatchVarShape benchmark data type");
        return;
    }

    benchutils::warmup_and_exec(
        state, BENCH_HQRESIZE_WARMUP_ITERATIONS,
        [&op, &ws, &src, &dst, interpolation = params.interpolation, antialias = params.antialias](cudaStream_t s)
        { op(s, ws.get(), src, dst, interpolation, interpolation, antialias); });
}

template<typename T>
inline void RunTensorBatchHQResizeBench(nvbench::state &state, cvcuda::HQResize &op, const HQResizeBenchParams &params)
{
    cvcuda::UniqueWorkspace ws = AllocateHQResizeBatchWorkspace(op, params);

    nvcv::TensorBatch srcTensors(static_cast<int32_t>(params.srcShape.x));
    nvcv::TensorBatch dstTensors(static_cast<int32_t>(params.dstShape.x));
    for (int i = 0; i < params.srcShape.x; ++i)
    {
        // clang-format off
        nvcv::Tensor src = params.planar ? nvcv::Tensor({{params.channels, params.srcShape.y, params.srcShape.z}, "CHW"}, params.dtype)
                                         : nvcv::Tensor({{params.srcShape.y, params.srcShape.z, params.channels}, "HWC"}, params.dtype);
        nvcv::Tensor dst = params.planar ? nvcv::Tensor({{params.channels, params.dstShape.y, params.dstShape.z}, "CHW"}, params.dtype)
                                         : nvcv::Tensor({{params.dstShape.y, params.dstShape.z, params.channels}, "HWC"}, params.dtype);
        // clang-format on

        benchutils::FillTensor<T>(src, benchutils::CheckerboardValues<T>());
        srcTensors.pushBack(src);
        dstTensors.pushBack(dst);
    }

    benchutils::warmup_and_exec(state, BENCH_HQRESIZE_WARMUP_ITERATIONS,
                                [&op, &ws, &srcTensors, &dstTensors, interpolation = params.interpolation,
                                 antialias = params.antialias](cudaStream_t s)
                                { op(s, ws.get(), srcTensors, dstTensors, interpolation, interpolation, antialias); });
}

template<typename T>
inline void hqresize(nvbench::state &state, nvbench::type_list<T>)
try
{
    long3                 srcShape      = benchutils::GetShape<3>(state.get_string("shape"));
    bool                  antialias     = benchutils::GetIntParam<bool>(state, "antialias");
    NVCVInterpolationType interpolation = benchutils::GetInterpolationType(state.get_string("interpolation"));
    const std::string     inputKind     = state.get_string("inputKind");
    auto                  layout        = benchutils::GetStringParam(state, "layout", "NHWC");
    if (layout != "NHWC" && layout != "NCHW" && layout != "NCHW_FAKE")
    {
        state.skip("HQResize benchmark supports only NHWC, NCHW, and NCHW_FAKE layouts");
        return;
    }
    if (inputKind != "Tensor" && inputKind != "VarShape" && inputKind != "TensorBatch")
    {
        state.skip("HQResize benchmark supports only Tensor, VarShape, and TensorBatch input kinds");
        return;
    }
    const bool planar     = layout == "NCHW";
    const bool fakePlanar = layout == "NCHW_FAKE";
    // Channel count is an explicit axis (default 1): the legacy single-channel NHWC profiles omit it,
    // while the planar-comparison profiles (NHWC/NCHW/NCHW_FAKE) set numChannels=3 so the three layouts
    // are compared at parity.
    const auto channels = static_cast<int>(state.get_int64_or_default("numChannels", 1));

    // NCHW_FAKE ("fake planar") is a tensor-only comparison path: planar data is reformatted to
    // interleaved, resized with the interleaved kernel, and reformatted back to planar -- all timed
    // together -- so the native planar (NCHW) path can be shown faster than this convert->resize->convert
    // pipeline.
    if (fakePlanar && inputKind != "Tensor")
    {
        state.skip("Fake-planar (NCHW_FAKE) HQResize benchmark is tensor-only");
        return;
    }

    const std::string resizeType = state.get_string("resizeType");
    long3             dstShape   = benchutils::GetResizeOutputShape(srcShape, resizeType);
    if (dstShape.y >= srcShape.y && dstShape.z >= srcShape.z && (dstShape.y > srcShape.y || dstShape.z > srcShape.z)
        && antialias)
    {
        state.skip("Antialias is no-op for expanding");
        return;
    }

    nvcv::Size2D srcSize{(int)srcShape.z, (int)srcShape.y};
    nvcv::Size2D dstSize{(int)dstShape.z, (int)dstShape.y};

    nvcv::DataType dtype{benchutils::GetDataType<T>()};

    const long srcBytes = srcShape.x * srcShape.y * srcShape.z * channels * static_cast<long>(sizeof(T));
    const long dstBytes = dstShape.x * dstShape.y * dstShape.z * channels * static_cast<long>(sizeof(T));
    if (fakePlanar)
    {
        // reformat(NCHW->NHWC) + resize + reformat(NHWC->NCHW): reads src twice + dst once,
        // writes the interleaved src once + dst twice.
        state.add_global_memory_reads(2 * srcBytes + dstBytes);
        state.add_global_memory_writes(srcBytes + 2 * dstBytes);
    }
    else
    {
        state.add_global_memory_reads(srcBytes);
        state.add_global_memory_writes(dstBytes);
    }

    const HQResizeBenchParams params{srcShape,      dstShape,  srcSize, dstSize, dtype,
                                     interpolation, antialias, planar,  channels};
    cvcuda::HQResize          op;

    if (fakePlanar) // tensor-only: NCHW -> NHWC -> hqresize -> NHWC -> NCHW
    {
        RunFakePlanarHQResizeBench<T>(state, op, params);
    }
    else if (inputKind == "Tensor")
    {
        RunTensorHQResizeBench<T>(state, op, params);
    }
    else if (inputKind == "VarShape")
    {
        RunVarShapeHQResizeBench<T>(state, op, params);
    }
    else // TensorBatch
    {
        RunTensorBatchHQResizeBench<T>(state, op, params);
    }
}

CVCUDA_BENCH_SKIP_ERRORS(state)

// Use auto-generated type list from bench_params.json

NVBENCH_BENCH_TYPES(hqresize, NVBENCH_TYPE_AXES(BENCH_HQRESIZE_TYPES))
BENCH_HQRESIZE_AXES;
