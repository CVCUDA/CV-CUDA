/* Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * SPDX-License-Identifier: Apache-2.0
 *
 * Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
 * Copyright (C) 2009-2010, Willow Garage Inc., all rights reserved.
 * Copyright (C) 2014-2015, Itseez Inc., all rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "Nvtx.hpp"
#include "OpGaussian.hpp"
#include "PlanarTensorView.hpp"
#include "SafeSize.hpp"
#include "TensorAccessValidation.hpp"

#include "FilterCommon.cuh"

#include <cvcuda/cuda_tools/DropCast.hpp>
#include <cvcuda/cuda_tools/MathWrappers.hpp>
#include <cvcuda/cuda_tools/StaticCast.hpp>
#include <nvcv/Exception.hpp>
#include <nvcv/util/CheckError.hpp>
#include <nvcv/util/Math.hpp>

#include <algorithm>
#include <cmath>
#include <iterator>
#include <type_traits>

namespace cvcuda::priv {

namespace {

namespace cuda   = nvcv::cuda;
namespace util   = nvcv::util;
namespace filter = cvcuda::priv::filter;

constexpr int kBlockWidth  = 16;
constexpr int kBlockHeight = 16;

// Geometry of the coefficient-generation launch; unrelated to the filter block above.
constexpr int kCoefBlockWidth  = 32;
constexpr int kCoefBlockHeight = 4;

// ---------------------------------------------------------------------------------------------
// Coefficient generation
// ---------------------------------------------------------------------------------------------

__global__ void ComputeGaussianKernel(float *kernel, int2 kernelSize, double2 sigma)
{
    int2 coord = cuda::StaticCast<int>(cuda::DropCast<2>(blockIdx * blockDim + threadIdx));

    if (coord.x >= kernelSize.x || coord.y >= kernelSize.y)
    {
        return;
    }

    int2 half{kernelSize.x / 2, kernelSize.y / 2};

    kernel[coord.y * kernelSize.x + coord.x] = filter::ComputeSingleGaussianValue(coord, half, sigma);
}

__global__ void ComputeGaussianKernelVarShape(cuda::Tensor3DWrap<float, int32_t> kernel, int dataKernelSize,
                                              int2 maxKernelSize, cuda::Tensor1DWrap<int2, int32_t> kernelSizeArr,
                                              cuda::Tensor1DWrap<double2, int32_t> sigmaArr)
{
    int3 coord = cuda::StaticCast<int>(blockIdx * blockDim + threadIdx);

    int2 kernelSize = kernelSizeArr[coord.z];

    double2 sigma = sigmaArr[coord.z];

    if (sigma.y <= 0)
        sigma.y = sigma.x;

    // automatic detection of kernel size from sigma
    if (kernelSize.x <= 0 && sigma.x > 0)
        kernelSize.x = cuda::round<int>(sigma.x * dataKernelSize * 2 + 1) | 1;
    if (kernelSize.y <= 0 && sigma.y > 0)
        kernelSize.y = cuda::round<int>(sigma.y * dataKernelSize * 2 + 1) | 1;

    NVCV_CUDA_ASSERT(kernelSize.x > 0 && (kernelSize.x % 2 == 1) && kernelSize.x <= maxKernelSize.x,
                     "E Wrong kernelSize.x = %d, expected > 0, odd and <= %d\n", kernelSize.x, maxKernelSize.x);
    NVCV_CUDA_ASSERT(kernelSize.y > 0 && (kernelSize.y % 2 == 1) && kernelSize.y <= maxKernelSize.y,
                     "E Wrong kernelSize.y = %d, expected > 0, odd and <= %d\n", kernelSize.y, maxKernelSize.y);

    if (coord.x >= kernelSize.x || coord.y >= kernelSize.y)
    {
        return;
    }

    int2 half{kernelSize.x / 2, kernelSize.y / 2};

    sigma.x = cuda::max(sigma.x, 0.0);
    sigma.y = cuda::max(sigma.y, 0.0);

    kernel[coord] = filter::ComputeSingleGaussianValue(coord, half, sigma);
}

// ---------------------------------------------------------------------------------------------
// Tensor path
// ---------------------------------------------------------------------------------------------

// Which output width the tiling ladder may use, decided from the value type alone.
//
// `kX4Always` types profile fastest at four outputs per thread everywhere. Scalar 2- and 4-byte
// integers and halves are the layout-dependent case: the same instantiation serves a packed
// single-channel tensor (which wants x4) and a flattened planar plane (which wants x2), so the
// caller's `enableX4` picks between them. Everything else is capped at x2 by register pressure.
template<typename T>
struct OutputWidthPolicy
{
    using BaseT = cuda::BaseType<T>;

    static constexpr bool kX4Always          = std::is_same_v<BaseT, filter::uchar> || std::is_same_v<T, float>;
    static constexpr bool kX4LayoutDependent = cuda::NumElements<T> == 1 && !kX4Always;
    static constexpr bool kDynamicX4Possible = kX4Always || kX4LayoutDependent || std::is_same_v<T, float3>;
};

// The dynamic-size ladder every launcher below ends with: step down through the output widths until
// the tile fits the shared-memory budget, then stop tiling altogether. Written once because the
// three launchers differ only in which untiled kernel they fall back to.
template<bool DynamicX4Possible, class GridFor, class SrcWrapper, class Dst, class Coefs, class Untiled>
void LaunchDynamicLadder(bool enableX4, const size_t (&sharedBytes)[3], dim3 block, const GridFor &gridFor,
                         cudaStream_t stream, SrcWrapper src, Dst dst, Coefs coefs, const Untiled &untiled)
{
    if constexpr (DynamicX4Possible)
    {
        if (enableX4 && sharedBytes[2] <= filter::kMaxSharedBytes)
        {
            filter::Filter2DTiled<4, true><<<gridFor(4), block, sharedBytes[2], stream>>>(src, dst, coefs);
            return;
        }
    }
    if (sharedBytes[1] <= filter::kMaxSharedBytes)
    {
        filter::Filter2DTiled<2, true><<<gridFor(2), block, sharedBytes[1], stream>>>(src, dst, coefs);
        return;
    }
    if (sharedBytes[0] <= filter::kMaxSharedBytes)
    {
        filter::Filter2DTiled<1, true><<<gridFor(1), block, sharedBytes[0], stream>>>(src, dst, coefs);
        return;
    }
    untiled();
}

template<bool DynamicX4Possible, class GridFor, class SrcWrapper, class Dst, class Coefs>
void LaunchDynamicLadder(bool enableX4, const size_t (&sharedBytes)[3], dim3 block, const GridFor &gridFor,
                         cudaStream_t stream, SrcWrapper src, Dst dst, Coefs coefs)
{
    LaunchDynamicLadder<DynamicX4Possible>(enableX4, sharedBytes, block, gridFor, stream, src, dst, coefs,
                                           [&]
                                           { filter::Filter2D<<<gridFor(1), block, 0, stream>>>(src, dst, coefs); });
}

template<typename T, NVCVBorderType B>
void GaussianTensor(const nvcv::TensorDataStridedCuda &inData, const nvcv::TensorDataStridedCuda &outData,
                    const float *kernel, int2 kernelSize, int2 kernelAnchor, float borderValue, bool enableX4,
                    cudaStream_t stream)
{
    using Policy = OutputWidthPolicy<T>;

    auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(inData);
    NVCV_ASSERT(inAccess);

    auto outAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(outData);
    NVCV_ASSERT(outAccess);

    filter::RequireInt32Addressable(*inAccess, *outAccess);

    const int2 dstSize{outAccess->numCols(), outAccess->numRows()};

    dim3 block(kBlockWidth, kBlockHeight);

    auto gridFor = [&](int outputs)
    {
        return dim3(util::DivUp(dstSize.x, kBlockWidth * outputs), util::DivUp(dstSize.y, kBlockHeight),
                    outAccess->numSamples());
    };

    auto src = cuda::CreateBorderWrapNHW<const T, B, int32_t>(inData, cuda::SetAll<T>(borderValue));
    auto dst = filter::MakeUniformDst(cuda::CreateTensorWrapNHW<T, int32_t>(outData), dstSize);

    auto coefs = filter::MakeUniformCoefs(kernel, kernelSize, kernelAnchor);

    const size_t sharedBytes[] = {filter::TiledSharedBytes(1, block, kernelSize, sizeof(T)),
                                  filter::TiledSharedBytes(2, block, kernelSize, sizeof(T)),
                                  filter::TiledSharedBytes(4, block, kernelSize, sizeof(T))};

    const bool launched = filter::TryCommonKernelSizes(
        kernelSize,
        [&](auto ksize)
        {
            constexpr int  kSize = decltype(ksize)::value;
            // The fixed sizes are the ones whose tile provably fits, so they need no runtime budget
            // test -- only this static one.
            constexpr bool kStageHalfAsFloat = std::is_same_v<T, __half>;
            using SharedT                    = std::conditional_t<kStageHalfAsFloat, float, T>;
            constexpr size_t kWideBytes      = filter::FixedSharedBytes<4, kSize, T, kBlockWidth, kBlockHeight>();
            constexpr size_t kNarrowBytes    = filter::FixedSharedBytes<2, kSize, SharedT, kBlockWidth, kBlockHeight>();

            if constexpr (Policy::kX4Always)
            {
                static_assert(kWideBytes <= filter::kMaxSharedBytes);
                filter::Filter2DTiledFixed<4, kSize, kSize, true>
                    <<<gridFor(4), block, sharedBytes[2], stream>>>(src, dst, coefs);
            }
            else if constexpr (Policy::kX4LayoutDependent)
            {
                static_assert(kWideBytes <= filter::kMaxSharedBytes);
                static_assert(kNarrowBytes <= filter::kMaxSharedBytes);
                if (enableX4)
                {
                    filter::Filter2DTiledFixed<4, kSize, kSize, true, false, true, false, kStageHalfAsFloat>
                        <<<gridFor(4), block, sharedBytes[2], stream>>>(src, dst, coefs);
                }
                else
                {
                    filter::Filter2DTiledFixed<2, kSize, kSize, true, false, true, kStageHalfAsFloat, kStageHalfAsFloat>
                        <<<gridFor(2), block, kNarrowBytes, stream>>>(src, dst, coefs);
                }
            }
            else
            {
                static_assert(kNarrowBytes <= filter::kMaxSharedBytes);
                filter::Filter2DTiledFixed<2, kSize, kSize, true>
                    <<<gridFor(2), block, sharedBytes[1], stream>>>(src, dst, coefs);
            }
        });

    if (!launched)
    {
        LaunchDynamicLadder<Policy::kDynamicX4Possible>(enableX4, sharedBytes, block, gridFor, stream, src, dst, coefs);
    }

    NVCV_CHECK_THROW(cudaGetLastError());
}

// ---------------------------------------------------------------------------------------------
// Var-shape paths
// ---------------------------------------------------------------------------------------------

// Var-shape dispatch has no packed-versus-planar ambiguity to resolve for the fixed sizes -- planar
// batches go to their own launcher -- so every scalar type takes the wide tile there.
template<typename T>
struct VarShapeOutputWidthPolicy
{
    using BaseT = cuda::BaseType<T>;

    static constexpr bool kX4Always          = std::is_same_v<BaseT, filter::uchar> || cuda::NumElements<T> == 1;
    static constexpr bool kDynamicX4Possible = kX4Always || std::is_same_v<T, float3>;
};

template<typename T, NVCVBorderType B>
void GaussianVarShapeInterleaved(const nvcv::ImageBatchVarShapeDataStridedCuda &inData,
                                 const nvcv::ImageBatchVarShapeDataStridedCuda &outData,
                                 const filter::GaussianBatchCoefs &coefs, bool enableX4, float borderValue,
                                 cudaStream_t stream)
{
    using Policy = VarShapeOutputWidthPolicy<T>;

    cuda::BorderVarShapeWrap<const T, B> src(inData, cuda::SetAll<T>(borderValue));

    auto dst = filter::MakeVarShapeDst(cuda::ImageBatchVarShapeWrap<T>(outData));

    const int2 maxKernelSize = coefs.maxKernelSize;

    dim3 block(kBlockWidth, kBlockHeight);

    auto gridFor = [&](int outputs)
    {
        return dim3(util::DivUp(inData.maxSize().w, kBlockWidth * outputs),
                    util::DivUp(inData.maxSize().h, kBlockHeight), outData.numImages());
    };

    const size_t sharedBytes[] = {filter::TiledSharedBytes(1, block, maxKernelSize, sizeof(T)),
                                  filter::TiledSharedBytes(2, block, maxKernelSize, sizeof(T)),
                                  filter::TiledSharedBytes(4, block, maxKernelSize, sizeof(T))};

    const bool launched = filter::TryCommonKernelSizes(
        maxKernelSize,
        [&](auto ksize)
        {
            constexpr int kSize    = decltype(ksize)::value;
            constexpr int kOutputs = Policy::kX4Always ? 4 : 2;
            static_assert(filter::FixedSharedBytes<kOutputs, kSize, T, kBlockWidth, kBlockHeight>()
                          <= filter::kMaxSharedBytes);
            // Measured: the block-level bounds test costs these launches more than it saves, because
            // the extent is a per-image load here. See the note on Filter2DTiledFixed.
            filter::Filter2DTiledFixed<kOutputs, kSize, kSize, false, /*LoadCoefsBeforeBounds=*/false,
                                       /*BlockLevelBoundsCheck=*/false, false, std::is_same_v<T, __half>>
                <<<gridFor(kOutputs), block, sharedBytes[kOutputs / 2], stream>>>(src, dst, coefs);
        });

    if (!launched)
    {
        LaunchDynamicLadder<Policy::kDynamicX4Possible>(enableX4, sharedBytes, block, gridFor, stream, src, dst, coefs);
    }

    NVCV_CHECK_THROW(cudaGetLastError());
}

template<typename T, NVCVBorderType B>
void GaussianVarShapePlanar(const nvcv::ImageBatchVarShapeDataStridedCuda &inData,
                            const nvcv::ImageBatchVarShapeDataStridedCuda &outData,
                            const filter::GaussianBatchCoefs &coefs, int channels, float borderValue,
                            cudaStream_t stream)
{
    // Planar images carry one channel per plane, so only the scalar cases reach here and the wide
    // tile is available exactly for the two dtypes whose loads are cheap enough to feed it.
    constexpr bool kX4Always = std::is_same_v<T, filter::uchar> || std::is_same_v<T, float>;

    cuda::BorderVarShapeWrap<const T, B> src(inData, cuda::SetAll<T>(borderValue));

    auto dst = filter::MakeVarShapePlanarDst(cuda::ImageBatchVarShapeWrap<T>(outData), channels);

    const int2 maxKernelSize = coefs.maxKernelSize;

    dim3          block(kBlockWidth, kBlockHeight);
    const int64_t tiledGridZ = static_cast<int64_t>(outData.numImages()) * channels;

    dim3 fallbackGrid(util::DivUp(outData.maxSize().w, kBlockWidth), util::DivUp(outData.maxSize().h, kBlockHeight),
                      outData.numImages());

    auto untiled = [&]
    {
        filter::Filter2DPlanar<<<fallbackGrid, block, 0, stream>>>(src, dst, coefs, channels);
    };

    // Tiling gives every (image, plane) its own block, so it needs gridDim.z = numImages * channels;
    // past the cap the untiled kernel walks the planes inside one thread instead.
    if (tiledGridZ > filter::kMaxGridZ)
    {
        untiled();
        NVCV_CHECK_THROW(cudaGetLastError());
        return;
    }

    auto gridFor = [&](int outputs)
    {
        return dim3(util::DivUp(outData.maxSize().w, kBlockWidth * outputs),
                    util::DivUp(outData.maxSize().h, kBlockHeight), static_cast<unsigned int>(tiledGridZ));
    };

    const size_t sharedBytes[] = {filter::TiledSharedBytes(1, block, maxKernelSize, sizeof(T)),
                                  filter::TiledSharedBytes(2, block, maxKernelSize, sizeof(T)),
                                  filter::TiledSharedBytes(4, block, maxKernelSize, sizeof(T))};

    const bool launched = filter::TryCommonKernelSizes(
        maxKernelSize,
        [&](auto ksize)
        {
            constexpr int kSize    = decltype(ksize)::value;
            constexpr int kOutputs = kX4Always ? 4 : 2;
            static_assert(filter::FixedSharedBytes<kOutputs, kSize, T, kBlockWidth, kBlockHeight>()
                          <= filter::kMaxSharedBytes);
            // Measured: the block-level bounds test costs these launches more than it saves, because
            // the extent is a per-image load here. See the note on Filter2DTiledFixed.
            filter::Filter2DTiledFixed<kOutputs, kSize, kSize, false, /*LoadCoefsBeforeBounds=*/false,
                                       /*BlockLevelBoundsCheck=*/false>
                <<<gridFor(kOutputs), block, sharedBytes[kOutputs / 2], stream>>>(src, dst, coefs);
        });

    if (!launched)
    {
        // Inside kX4Always the caller's enableX4 is already true for exactly these two dtypes, so
        // the wide arm needs no second test.
        LaunchDynamicLadder<kX4Always>(kX4Always, sharedBytes, block, gridFor, stream, src, dst, coefs, untiled);
    }

    NVCV_CHECK_THROW(cudaGetLastError());
}

// The sigma-to-window-size rule is coarser for 8-bit data, whose quantization hides the tail.
inline int DataKernelSizeFactor(filter::ElemType elemType)
{
    return elemType == filter::ElemType::kU8 ? 3 : 4;
}

} // namespace

ScopedCudaEvent ScopedCudaEvent::Create(unsigned flags)
{
    ScopedCudaEvent event;
    NVCV_CHECK_THROW(cudaEventCreateWithFlags(&event.m_event, flags));
    return event;
}

GaussianCoefficientBuffer::GaussianCoefficientBuffer(size_t numFloats)
    // Timing is never read off this event; it only orders one stream behind another. Creating it
    // in the initializer list rather than after the allocation below means a throw from cudaMalloc
    // unwinds a fully constructed member, where a raw handle assigned afterwards would leak.
    : m_ready(ScopedCudaEvent::Create(cudaEventDisableTiming))
{
    if (numFloats > 0)
    {
        NVCV_CHECK_THROW(cudaMalloc(&m_data, numFloats * sizeof(float)));
    }
}

GaussianCoefficientBuffer::~GaussianCoefficientBuffer()
{
    NVCV_CHECK_LOG(cudaFree(m_data));
}

void GaussianCoefficientBuffer::RecordReady(cudaStream_t stream)
{
    NVCV_CHECK_THROW(cudaEventRecord(m_ready.get(), stream));
}

void GaussianCoefficientBuffer::WaitReady(cudaStream_t stream)
{
    NVCV_CHECK_THROW(cudaStreamWaitEvent(stream, m_ready.get(), 0));
}

void GaussianCoefficientBuffer::WaitForAllUses(cudaStream_t stream)
{
    // Not a structured binding: NVCV_CHECK_THROW expands to a lambda, and capturing one is C++20.
    for (const auto &use : m_lastUse)
    {
        // Work already queued on this stream is ordered by the stream itself.
        if (use.first != stream)
        {
            cudaEvent_t event = use.second.get();
            NVCV_CHECK_THROW(cudaStreamWaitEvent(stream, event, 0));
        }
    }
}

void GaussianCoefficientBuffer::RecordUse(cudaStream_t stream)
{
    std::lock_guard<std::mutex> lock(m_mutex);

    auto it = m_lastUse.find(stream);
    if (it == m_lastUse.end())
    {
        if (m_lastUse.size() >= kPruneThreshold)
        {
            PruneRetiredUses();
        }
        it = m_lastUse.emplace(stream, ScopedCudaEvent::Create(cudaEventDisableTiming)).first;
    }

    NVCV_CHECK_THROW(cudaEventRecord(it->second.get(), stream));
}

void GaussianCoefficientBuffer::PruneRetiredUses()
{
    for (auto it = m_lastUse.begin(); it != m_lastUse.end();)
    {
        // cudaEventQuery reports success for a retired event; waiting on one is a no-op, so
        // dropping it loses no ordering.
        it = (cudaEventQuery(it->second.get()) == cudaSuccess) ? m_lastUse.erase(it) : std::next(it);
    }
}

Gaussian::Gaussian(nvcv::Size2D maxKernelSize, int maxBatchSize)
    : m_maxKernelSize(maxKernelSize)
    , m_maxBatchSize(maxBatchSize)
    // The coefficient scratch belongs to the device that was current when it was allocated, so it is
    // created per device exactly like the operator instance it replaces.
    , m_coefs(
          [maxKernelSize](int)
          {
              return std::make_unique<GaussianCoefficientBuffer>(
                  CheckedMulMany({CheckedNonNegativeToSize(maxKernelSize.w, "maxKernelSize.w"),
                                  CheckedNonNegativeToSize(maxKernelSize.h, "maxKernelSize.h")},
                                 "Gaussian kernel allocation size overflow"));
          })
    , m_varShapeCoefs(
          [maxKernelSize, maxBatchSize](int)
          {
              return std::make_unique<GaussianCoefficientBuffer>(
                  maxBatchSize > 0 ? CheckedMulMany({CheckedPositiveToSize(maxKernelSize.w, "maxKernelSize.w"),
                                                     CheckedPositiveToSize(maxKernelSize.h, "maxKernelSize.h"),
                                                     CheckedPositiveToSize(maxBatchSize, "maxBatchSize")},
                                                    "GaussianVarShape kernel allocation size overflow")
                                   : 0);
          })
{
}

void Gaussian::operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out, nvcv::Size2D kernelSize,
                          double2 sigma, NVCVBorderType borderMode) const
{
    CVCUDA_NVTX_RANGE("cvcuda::Gaussian::operator()[Tensor]");
    auto inData  = RequireCudaTensor(in, "Input");
    auto outData = RequireCudaTensor(out, "Output");

    RequireSameDtype(*inData, *outData);

    filter::ValidateTensorLayout(inData->layout());
    filter::ValidateTensorLayout(outData->layout());
    RequireSameLayout(*inData, *outData);

    filter::ValidateBorderMode(borderMode);

    const filter::ElemType elemType = filter::ClassifyFilterElemType(inData->dtype());

    if (sigma.y <= 0)
        sigma.y = sigma.x;

    // automatic detection of kernel size from sigma
    const int dataKernelSize = DataKernelSizeFactor(elemType);
    if (kernelSize.w <= 0 && sigma.x > 0)
        kernelSize.w = cuda::round<int>(sigma.x * dataKernelSize * 2 + 1) | 1;
    if (kernelSize.h <= 0 && sigma.y > 0)
        kernelSize.h = cuda::round<int>(sigma.y * dataKernelSize * 2 + 1) | 1;

    if (!(kernelSize.w > 0 && kernelSize.w % 2 == 1 && kernelSize.w <= m_maxKernelSize.w && kernelSize.h > 0
          && kernelSize.h % 2 == 1 && kernelSize.h <= m_maxKernelSize.h))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid kernel size %dx%d", kernelSize.w,
                              kernelSize.h);
    }

    sigma.x = std::max(sigma.x, 0.0);
    sigma.y = std::max(sigma.y, 0.0);

    auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*inData);
    NVCV_ASSERT(inAccess);

    const int channels = inAccess->numChannels();
    filter::ValidateChannelCount(channels);

    const int2 size{kernelSize.w, kernelSize.h};

    // Generated only when this window is not the one already in the buffer. EnsureWindow owns the
    // cross-stream ordering that reuse needs; see the note on GaussianCoefficientBuffer.
    auto  &buffer = m_coefs.get();
    float *coefs  = buffer.data();

    buffer.EnsureWindow(stream, size, sigma,
                        [&](float *dst)
                        {
                            dim3 block(kCoefBlockWidth, kCoefBlockHeight);
                            dim3 grid(util::DivUp(size.x, kCoefBlockWidth), util::DivUp(size.y, kCoefBlockHeight));
                            ComputeGaussianKernel<<<grid, block, 0, stream>>>(dst, size, sigma);
                            NVCV_CHECK_THROW(cudaGetLastError());
                        });

    const int2      anchor{size.x / 2, size.y / 2};
    constexpr float borderValue = 0.f;

    auto planarViews = PlanarSingleChannelViews(*inData, *outData);

    const nvcv::TensorDataStridedCuda &src = planarViews ? planarViews->first : *inData;
    const nvcv::TensorDataStridedCuda &dst = planarViews ? planarViews->second : *outData;

    // A flattened planar plane is a wide single-channel image, but its rows are only as long as one
    // channel, so it stops short of the x4 tile that a packed single-channel tensor pays for.
    const bool enableX4 = planarViews
                            ? (elemType == filter::ElemType::kU8 || elemType == filter::ElemType::kF32)
                            : (elemType == filter::ElemType::kU8 || channels == 1
                               || (elemType == filter::ElemType::kF32 && channels == 3 && size.x > 5 && size.y > 5));

    filter::RunTypeBorderSwitch(elemType, planarViews ? 1 : channels, borderMode,
                                [&](auto value, auto border)
                                {
                                    GaussianTensor<decltype(value), decltype(border)::value>(
                                        src, dst, coefs, size, anchor, borderValue, enableX4, stream);
                                });

    // The filter above is the last reader of the window on this stream. A submit that later evicts
    // it orders its regenerate behind this point.
    buffer.RecordUse(stream);
}

void Gaussian::operator()(cudaStream_t stream, const nvcv::ImageBatchVarShape &in, const nvcv::ImageBatchVarShape &out,
                          const nvcv::Tensor &kernelSize, const nvcv::Tensor &sigma, NVCVBorderType borderMode) const
{
    CVCUDA_NVTX_RANGE("cvcuda::Gaussian::operator()[ImageBatchVarShape]");
    auto inData         = RequireCudaVarShape(in, stream, "Input");
    auto outData        = RequireCudaVarShape(out, stream, "Output");
    auto kernelSizeData = RequireCudaTensor(kernelSize, "Kernel size");
    auto sigmaData      = RequireCudaTensor(sigma, "Kernel sigma");

    RequireUniqueFormat(*inData, "input");
    RequireUniqueFormat(*outData, "output");

    if (m_maxBatchSize <= 0 || inData->numImages() > m_maxBatchSize)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid maximum batch size");
    }

    // ComputeGaussianKernelVarShape indexes both parameter tensors up to numImages, and the filter
    // dispatch wraps the output with the input's element type and channel count.
    RequireSameNumImages(*inData, *outData);
    RequireSameFormat(*inData, *outData);

    // The two dispatches size their grids from opposite batches -- interleaved from the input's
    // maxSize, planar from the output's -- so a per-image extent mismatch leaves part of the output
    // unwritten on one path and blurs against the border on the other. Read the sizes off the
    // handles: the strided batch data keeps its image list in device memory.
    cvcuda::priv::RequireSameVarShapeImageSizes(in, out);

    cvcuda::priv::RequirePerImageParam(*kernelSizeData, nvcv::TYPE_2S32, inData->numImages(), "Kernel size");
    cvcuda::priv::RequirePerImageParam(*sigmaData, nvcv::TYPE_2F64, inData->numImages(), "Sigma");

    const filter::BatchLayout inLayout  = filter::ClassifyBatchLayout(*inData);
    const filter::BatchLayout outLayout = filter::ClassifyBatchLayout(*outData);
    filter::RequireSameBatchLayout(inLayout, outLayout);

    filter::ValidateBorderMode(borderMode);

    const filter::ElemType elemType = filter::ClassifyFilterElemType(inData->uniqueFormat());

    const int channels = inData->uniqueFormat().numChannels();
    filter::ValidateChannelCount(channels);

    const int2      maxKernelSize{m_maxKernelSize.w, m_maxKernelSize.h};
    const int       dataKernelSize = DataKernelSizeFactor(elemType);
    constexpr float borderValue    = 0.f;

    const int kernelPitch2 = static_cast<int>(maxKernelSize.x * sizeof(float));
    const int kernelPitch1 = maxKernelSize.y * kernelPitch2;

    cuda::Tensor3DWrap<float, int32_t>   kernelTensor(m_varShapeCoefs.get().data(), kernelPitch1, kernelPitch2);
    cuda::Tensor1DWrap<int2, int32_t>    kernelSizeTensor(*kernelSizeData);
    cuda::Tensor1DWrap<double2, int32_t> sigmaTensor(*sigmaData);

    {
        dim3 block(kCoefBlockWidth, kCoefBlockHeight);
        dim3 grid(util::DivUp(maxKernelSize.x, kCoefBlockWidth), util::DivUp(maxKernelSize.y, kCoefBlockHeight),
                  outData->numImages());
        ComputeGaussianKernelVarShape<<<grid, block, 0, stream>>>(kernelTensor, dataKernelSize, maxKernelSize,
                                                                  kernelSizeTensor, sigmaTensor);
        NVCV_CHECK_THROW(cudaGetLastError());
    }

    const filter::GaussianBatchCoefs coefs{kernelTensor, kernelSizeTensor, sigmaTensor, maxKernelSize, dataKernelSize};

    if (inLayout.planar)
    {
        filter::RequirePlanarGridZ(outData->numImages());

        filter::RunBaseTypeBorderSwitch(elemType, borderMode,
                                        [&](auto base, auto border)
                                        {
                                            GaussianVarShapePlanar<decltype(base), decltype(border)::value>(
                                                *inData, *outData, coefs, channels, borderValue, stream);
                                        });
        return;
    }

    const bool enableX4
        = elemType == filter::ElemType::kU8 || channels == 1
       || (elemType == filter::ElemType::kF32 && channels == 3 && maxKernelSize.x > 5 && maxKernelSize.y > 5);

    filter::RunTypeBorderSwitch(elemType, channels, borderMode,
                                [&](auto value, auto border)
                                {
                                    GaussianVarShapeInterleaved<decltype(value), decltype(border)::value>(
                                        *inData, *outData, coefs, enableX4, borderValue, stream);
                                });
}

} // namespace cvcuda::priv
