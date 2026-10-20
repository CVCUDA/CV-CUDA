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

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

struct HQResizeBenchParams
{
    long3                     srcShape;
    long3                     dstShape;
    nvcv::Size2D              srcSize;
    nvcv::Size2D              dstSize;
    nvcv::DataType            dtype;
    NVCVInterpolationType     interpolation;
    bool                      antialias;
    bool                      planar;
    int                       channels;
    std::vector<HQResizeRoiF> rois;
};

std::optional<std::pair<int, int>> ParsePositiveSquareOrShape(std::string_view value)
{
    if (value.empty() || value.find_first_not_of("0123456789x") != std::string_view::npos)
    {
        return std::nullopt;
    }

    try
    {
        const std::string dimensions{value};
        if (value.find('x') == std::string_view::npos)
        {
            const auto side = benchutils::GetShape<1>(dimensions);
            return side.x > 0 ? std::optional{std::pair{static_cast<int>(side.x), static_cast<int>(side.x)}}
                              : std::nullopt;
        }

        const auto shape = benchutils::GetShape<2>(dimensions);
        return shape.x > 0 && shape.y > 0
                 ? std::optional{std::pair{static_cast<int>(shape.x), static_cast<int>(shape.y)}}
                 : std::nullopt;
    }
    catch (const std::invalid_argument &)
    {
        return std::nullopt;
    }
    catch (const std::out_of_range &)
    {
        return std::nullopt;
    }
}

std::optional<HQResizeRoiF> GetVLMCenterCropRoi(int inputHeight, int inputWidth, int outputHeight, int outputWidth)
{
    const double resizeScale
        = std::max(static_cast<double>(outputHeight) / inputHeight, static_cast<double>(outputWidth) / inputWidth);
    const double scaledHeight = static_cast<double>(inputHeight) * resizeScale + 0.5;
    const double scaledWidth  = static_cast<double>(inputWidth) * resizeScale + 0.5;
    if (scaledHeight > std::numeric_limits<int>::max() || scaledWidth > std::numeric_limits<int>::max())
    {
        return std::nullopt;
    }
    const auto  resizeHeight = static_cast<int>(scaledHeight);
    const auto  resizeWidth  = static_cast<int>(scaledWidth);
    const int   cropY        = (resizeHeight - outputHeight) / 2;
    const int   cropX        = (resizeWidth - outputWidth) / 2;
    const float scaleY       = static_cast<float>(inputHeight) / static_cast<float>(resizeHeight);
    const float scaleX       = static_cast<float>(inputWidth) / static_cast<float>(resizeWidth);
    return HQResizeRoiF{
        {               static_cast<float>(cropY) * scaleY,               static_cast<float>(cropX) * scaleX},
        {static_cast<float>(cropY + outputHeight) * scaleY, static_cast<float>(cropX + outputWidth) * scaleX}
    };
}

template<typename T>
bool ValidateVLMCenterCropInput(nvbench::state &state, const long3 &srcShape, std::string_view inputKind,
                                std::string_view layout, int channels, NVCVInterpolationType interpolation)
{
    if (inputKind != "VarShape")
    {
        state.skip("VLM center crop requires inputKind=VarShape");
        return false;
    }
    if (layout != "NHWC")
    {
        state.skip("VLM center crop requires layout=NHWC");
        return false;
    }
    if (srcShape.y <= 0)
    {
        state.skip("VLM center crop requires a positive input height");
        return false;
    }
    if (srcShape.z < srcShape.y)
    {
        state.skip("VLM center crop requires landscape input (width >= height)");
        return false;
    }
    if (srcShape.y > std::numeric_limits<int>::max() || srcShape.z > std::numeric_limits<int>::max())
    {
        state.skip("VLM center crop input dimensions exceed the supported range");
        return false;
    }
    if constexpr (!std::is_same_v<T, uint8_t>)
    {
        state.skip("VLM center crop requires InOutDataType=uint8");
        return false;
    }
    if (channels != 3)
    {
        state.skip("VLM center crop requires numChannels=3");
        return false;
    }
    if (interpolation != NVCV_INTERP_CUBIC)
    {
        state.skip("VLM center crop requires interpolation=CUBIC");
        return false;
    }
    return true;
}

inline HQResizeRoisF BatchRois(const HQResizeBenchParams &params)
{
    if (params.rois.empty())
    {
        return {};
    }
    return {static_cast<int32_t>(params.rois.size()), 2, params.rois.data()};
}

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

    const HQResizeRoisF roiDesc = BatchRois(params);
    return cvcuda::AllocateWorkspace(op.getWorkspaceRequirements(
        wsSamples, inShapeDesc, outShapeDesc, params.interpolation, params.interpolation, params.antialias, roiDesc));
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

    const HQResizeRoisF rois = BatchRois(params);
    benchutils::warmup_and_exec(
        state, BENCH_HQRESIZE_WARMUP_ITERATIONS,
        [&op, &ws, &src, &dst, interpolation = params.interpolation, antialias = params.antialias, rois](cudaStream_t s)
        { op(s, ws.get(), src, dst, interpolation, interpolation, antialias, rois); });
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

    const HQResizeRoisF rois = BatchRois(params);
    benchutils::warmup_and_exec(
        state, BENCH_HQRESIZE_WARMUP_ITERATIONS,
        [&op, &ws, &srcTensors, &dstTensors, interpolation = params.interpolation, antialias = params.antialias,
         rois](cudaStream_t s)
        { op(s, ws.get(), srcTensors, dstTensors, interpolation, interpolation, antialias, rois); });
}

template<typename T>
void RunHQResizeBench(nvbench::state &state, cvcuda::HQResize &op, const HQResizeBenchParams &params,
                      std::string_view inputKind, bool fakePlanar)
{
    if (fakePlanar)
    {
        // NCHW -> NHWC -> HQResize -> NHWC -> NCHW.
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
    else
    {
        // TensorBatch is the only remaining validated input kind.
        RunTensorBatchHQResizeBench<T>(state, op, params);
    }
}

} // namespace

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

    const auto                 resizeType = state.get_string("resizeType");
    long3                      dstShape;
    std::vector<HQResizeRoiF>  rois;
    long                       srcBytes;
    auto                       effectiveInputHeight = static_cast<double>(srcShape.y);
    auto                       effectiveInputWidth  = static_cast<double>(srcShape.z);
    constexpr std::string_view kVLMCenterCropPrefix = "VLM_CENTER_CROP_";
    if (resizeType.rfind(kVLMCenterCropPrefix, 0) == 0)
    {
        const auto output
            = ParsePositiveSquareOrShape(std::string_view{resizeType}.substr(kVLMCenterCropPrefix.size()));
        if (!output)
        {
            state.skip("VLM center-crop output must be a positive HxW or square value");
            return;
        }
        const auto [outputHeight, outputWidth] = *output;
        if (!ValidateVLMCenterCropInput<T>(state, srcShape, inputKind, layout, channels, interpolation))
        {
            return;
        }

        // Model the standard resize-to-cover followed by a centered crop as
        // one equivalent source ROI; bench_hqresize.py mirrors this contract.
        const auto roi = GetVLMCenterCropRoi(static_cast<int>(srcShape.y), static_cast<int>(srcShape.z), outputHeight,
                                             outputWidth);
        if (!roi)
        {
            state.skip("VLM center-crop intermediate dimensions exceed the supported range");
            return;
        }
        dstShape              = long3{srcShape.x, outputHeight, outputWidth};
        effectiveInputHeight  = static_cast<double>(roi->hi[0] - roi->lo[0]);
        effectiveInputWidth   = static_cast<double>(roi->hi[1] - roi->lo[1]);
        const long readHeight = std::clamp(static_cast<long>(std::ceil(roi->hi[0])), 0L, srcShape.y)
                              - std::clamp(static_cast<long>(std::floor(roi->lo[0])), 0L, srcShape.y);
        const long readWidth = std::clamp(static_cast<long>(std::ceil(roi->hi[1])), 0L, srcShape.z)
                             - std::clamp(static_cast<long>(std::floor(roi->lo[1])), 0L, srcShape.z);
        srcBytes = srcShape.x * readHeight * readWidth * channels * static_cast<long>(sizeof(T));
        rois.assign(static_cast<std::size_t>(srcShape.x), *roi);
    }
    else
    {
        dstShape = benchutils::GetResizeOutputShape(srcShape, resizeType);
        srcBytes = srcShape.x * srcShape.y * srcShape.z * channels * static_cast<long>(sizeof(T));
    }
    const auto outputHeight = static_cast<double>(dstShape.y);
    const auto outputWidth  = static_cast<double>(dstShape.z);
    if (outputHeight >= effectiveInputHeight && outputWidth >= effectiveInputWidth
        && (outputHeight > effectiveInputHeight || outputWidth > effectiveInputWidth) && antialias)
    {
        state.skip("Antialias is no-op for expanding");
        return;
    }

    nvcv::Size2D srcSize{static_cast<int>(srcShape.z), static_cast<int>(srcShape.y)};
    nvcv::Size2D dstSize{static_cast<int>(dstShape.z), static_cast<int>(dstShape.y)};

    nvcv::DataType dtype{benchutils::GetDataType<T>()};

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

    const HQResizeBenchParams params{srcShape,      dstShape,  srcSize, dstSize,  dtype,
                                     interpolation, antialias, planar,  channels, std::move(rois)};
    cvcuda::HQResize          op;

    RunHQResizeBench<T>(state, op, params, inputKind, fakePlanar);
}

CVCUDA_BENCH_SKIP_ERRORS(state)

// Use auto-generated type list from bench_params.json

NVBENCH_BENCH_TYPES(hqresize, NVBENCH_TYPE_AXES(BENCH_HQRESIZE_TYPES))
BENCH_HQRESIZE_AXES;
