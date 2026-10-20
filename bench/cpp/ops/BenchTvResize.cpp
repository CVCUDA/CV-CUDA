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
#include "ops/generated/BenchTvResizeConfig.hpp"

#include <cvcuda/OpReformat.hpp>
#include <cvcuda/OpTvResize.hpp>
#include <cvcuda/Workspace.hpp>

#include <nvbench/nvbench.cuh>

#include <algorithm>
#include <string_view>
#include <type_traits>
#include <vector>

namespace {

template<typename T>
T TvResizeRamp(long index, long sample)
{
    const auto value = (index * 17 + sample * 13) % 251;
    if constexpr (std::is_same_v<T, float>)
    {
        return static_cast<float>(value) / 251.f;
    }
    else
    {
        return static_cast<T>(value);
    }
}

std::vector<int2> TvResizeSizes(int count, int height, int width, std::string_view batchCase, bool output)
{
    if (count < 0)
    {
        throw std::invalid_argument("Invalid TvResize batchCase/count");
    }
    if (batchCase == "uniform")
    {
        return std::vector<int2>(count, int2{height, width});
    }
    if (count <= 0 || count % 3 != 0 || (batchCase != "ordinary" && batchCase != "mixed-singleton"))
    {
        throw std::invalid_argument("Invalid TvResize batchCase/count");
    }
    std::vector<int2> sizes;
    if (output)
    {
        sizes = {
            { 720, 1280},
            {1080, 1920},
            {   4,    3}
        };
    }
    else
    {
        sizes = {
            {                           1080, 1920},
            {                           2160, 3840},
            {batchCase == "ordinary" ? 7 : 1,    7}
        };
    }
    sizes.resize(count);
    for (int sample = 3; sample < count; ++sample)
    {
        sizes[sample] = sizes[sample % 3];
    }
    return sizes;
}

template<typename T>
void FillTvResizeTensor(nvcv::Tensor &tensor, bool planar, int width, int channels)
{
    benchutils::FillTensor<T>(tensor,
                              [planar, width, channels](const long4_16a &p)
                              {
                                  const long y = planar ? p.z : p.y;
                                  const long x = planar ? p.w : p.z;
                                  const long c = planar ? p.y : p.w;
                                  return TvResizeRamp<T>((y * width + x) * channels + c, p.x);
                              });
}

template<typename T>
void FillTvResizeImage(nvcv::Image &image, int sample, int channels)
{
    auto data = image.exportData<nvcv::ImageDataStridedCuda>();
    CVCUDA_CHECK_DATA(data);
    const auto    &plane = data->plane(0);
    std::vector<T> pixels(static_cast<size_t>(plane.height) * plane.width * channels);
    std::generate(pixels.begin(), pixels.end(),
                  [sample, index = 0L]() mutable
                  {
                      auto value = TvResizeRamp<T>(index, sample);
                      ++index;
                      return value;
                  });
    const size_t rowBytes = static_cast<size_t>(plane.width) * channels * sizeof(T);
    CUDA_CHECK_ERROR(cudaMemcpy2D(plane.basePtr, plane.rowStride, pixels.data(), rowBytes, rowBytes, plane.height,
                                  cudaMemcpyHostToDevice));
}

} // namespace

template<typename T>
inline void tvresize(nvbench::state &state, nvbench::type_list<T>)
try
{
    using BT             = nvcv::cuda::BaseType<T>;
    const auto shape     = benchutils::GetShape<3, int3>(state.get_string("shape"));
    const auto output    = benchutils::GetShape<2, int2>(state.get_string("outputShape"));
    const auto layout    = state.get_string("layout");
    const auto kind      = state.get_string("inputKind");
    const auto batchCase = state.get_string("batchCase");
    const auto mode      = state.get_string("interpolation");
    const auto interpolation
        = mode == "NEAREST_FLOOR" ? NVCV_INTERP_NEAREST_FLOOR : benchutils::GetInterpolationType(mode);
    const auto antialias = state.get_int64("antialias") != 0;
    const auto channels  = static_cast<int>(state.get_int64("numChannels"));
    const bool fake      = layout == "NCHW_FAKE";
    const bool planar    = layout == "NCHW" || fake;
    if ((layout != "NHWC" && !planar) || (kind != "Tensor" && kind != "VarShape")
        || (kind == "VarShape" && (layout != "NHWC" || channels != nvcv::cuda::NumElements<T>))
        || (kind == "Tensor" && batchCase != "uniform") || (fake && channels > 4))
    {
        state.skip("Unsupported TvResize benchmark axis combination");
        return;
    }
    auto srcSizes = TvResizeSizes(shape.x, shape.y, shape.z, batchCase, false);
    auto dstSizes = TvResizeSizes(shape.x, output.x, output.y, batchCase, true);
    long srcBytes = 0;
    long dstBytes = 0;
    for (int n = 0; n < shape.x; ++n)
    {
        srcBytes += static_cast<long>(srcSizes[n].x) * srcSizes[n].y * channels * sizeof(BT);
        dstBytes += static_cast<long>(dstSizes[n].x) * dstSizes[n].y * channels * sizeof(BT);
    }
    state.add_global_memory_reads(fake ? 2 * srcBytes + dstBytes : srcBytes);
    state.add_global_memory_writes(fake ? srcBytes + 2 * dstBytes : dstBytes);

    cvcuda::TvResize op;
    if (kind == "VarShape")
    {
        nvcv::ImageBatchVarShape src(shape.x);
        nvcv::ImageBatchVarShape dst(shape.x);
        std::vector<int32_t>     srcH;
        std::vector<int32_t>     srcW;
        std::vector<int32_t>     dstH;
        std::vector<int32_t>     dstW;
        for (int n = 0; n < shape.x; ++n)
        {
            srcH.push_back(srcSizes[n].x);
            srcW.push_back(srcSizes[n].y);
            dstH.push_back(dstSizes[n].x);
            dstW.push_back(dstSizes[n].y);
            nvcv::Image input({srcW.back(), srcH.back()}, benchutils::GetFormat<T>());
            nvcv::Image outputImage({dstW.back(), dstH.back()}, benchutils::GetFormat<T>());
            FillTvResizeImage<BT>(input, n, channels);
            src.pushBack(input);
            dst.pushBack(outputImage);
        }
        auto ws = cvcuda::AllocateWorkspace(op.getWorkspaceRequirements(
            shape.x, srcH.data(), srcW.data(), dstH.data(), dstW.data(), channels, interpolation, antialias));
        benchutils::warmup_and_exec(state, BENCH_TVRESIZE_WARMUP_ITERATIONS,
                                    [&op, &ws, &src, &dst, interpolation, antialias](cudaStream_t s)
                                    { op(s, ws.get(), src, dst, interpolation, antialias); });
        return;
    }

    const auto dtype = benchutils::GetDataType<BT>();
    auto       ws    = cvcuda::AllocateWorkspace(
                 op.getWorkspaceRequirements(shape.x, shape.y, shape.z, output.x, output.y, channels, interpolation, antialias));
    // clang-format off
    nvcv::Tensor src = planar ? nvcv::Tensor({{shape.x, channels, shape.y, shape.z}, "NCHW"}, dtype)
                              : nvcv::Tensor({{shape.x, shape.y, shape.z, channels}, "NHWC"}, dtype);
    nvcv::Tensor dst = planar ? nvcv::Tensor({{shape.x, channels, output.x, output.y}, "NCHW"}, dtype)
                              : nvcv::Tensor({{shape.x, output.x, output.y, channels}, "NHWC"}, dtype);
    // clang-format on
    FillTvResizeTensor<BT>(src, planar, shape.z, channels);
    if (fake)
    {
        // Regression reference: include both layout conversions in the timed region.
        // clang-format off
        nvcv::Tensor interSrc({{shape.x, shape.y, shape.z, channels}, "NHWC"}, dtype);
        nvcv::Tensor interDst({{shape.x, output.x, output.y, channels}, "NHWC"}, dtype);
        // clang-format on
        cvcuda::Reformat reformat;
        benchutils::warmup_and_exec(
            state, BENCH_TVRESIZE_WARMUP_ITERATIONS,
            [&reformat, &src, &interSrc, &op, &ws, &interDst, &dst, interpolation, antialias](cudaStream_t s)
            {
                reformat(s, src, interSrc);
                op(s, ws.get(), interSrc, interDst, interpolation, antialias);
                reformat(s, interDst, dst);
            });
    }
    else
    {
        benchutils::warmup_and_exec(state, BENCH_TVRESIZE_WARMUP_ITERATIONS,
                                    [&op, &ws, &src, &dst, interpolation, antialias](cudaStream_t s)
                                    { op(s, ws.get(), src, dst, interpolation, antialias); });
    }
}
CVCUDA_BENCH_SKIP_ERRORS(state)

NVBENCH_BENCH_TYPES(tvresize, NVBENCH_TYPE_AXES(BENCH_TVRESIZE_TYPES))
BENCH_TVRESIZE_AXES;
