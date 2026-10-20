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

#include "CudaDeviceUtils.hpp"
#include "FilterSchedulePolicy.hpp"
#include "Nvtx.hpp"
#include "OpAverageBlur.hpp"
#include "PlanarTensorView.hpp"
#include "TensorAccessValidation.hpp"

#include "FilterCommon.cuh"

#include <nvcv/Exception.hpp>
#include <nvcv/util/CheckError.hpp>
#include <nvcv/util/Math.hpp>

#include <type_traits>

namespace cvcuda::priv {

namespace {

namespace cuda   = nvcv::cuda;
namespace util   = nvcv::util;
namespace filter = cvcuda::priv::filter;

constexpr int kBlockWidth  = 32;
constexpr int kBlockHeight = 8;

// AverageBlur never materializes its kernel: every tap has the same weight, so the window is fully
// described by (size, anchor). A tensor carries one window for the whole batch; a var-shape batch
// carries one per image, with (-1, -1) meaning "kernel center".
struct UniformBoxParams
{
    int2 size;
    int2 anchor;

    inline __device__ int2 Size(int) const
    {
        return size;
    }

    inline __device__ int2 Anchor(int, int2) const
    {
        return anchor;
    }

    inline __device__ float Linear(int, int, int2 kernelSize) const
    {
        return static_cast<float>(1.0 / (kernelSize.x * kernelSize.y));
    }
};

// Per-image constant-weight coefficient policy for var-shape box filters.
// Satisfies the Coefs contract in FilterCommon.cuh so that the VarShape paths
// can reuse Filter2DTiledFixed instead of duplicating the tiling logic with its
// own bespoke border-wrapper loop.
struct VarShapeBoxCoefs
{
    cuda::Tensor1DWrap<int2, int32_t> sizes;
    cuda::Tensor1DWrap<int2, int32_t> anchors;

    inline __device__ int2 Size(int b) const
    {
        int2 sz = sizes[b];
        NVCV_CUDA_ASSERT(sz.x > 0 && sz.y > 0, "E Wrong kernelSize=(%d,%d), expected both > 0", sz.x, sz.y);
        return sz;
    }

    inline __device__ int2 Anchor(int b, int2 kernelSize) const
    {
        int2 a = anchors[b];
        if (a.x < 0)
            a.x = kernelSize.x / 2;
        if (a.y < 0)
            a.y = kernelSize.y / 2;
        return a;
    }

    inline __device__ float At(int b, int, int) const
    {
        // sizes[b] is L1-cached after the Size() call in the kernel body.
        int2 sz = sizes[b];
        return 1.f / float(sz.x * sz.y);
    }

    inline __device__ float Linear(int, int, int2 kernelSize) const
    {
        // kernelSize is already in registers in the tiled kernel; no extra global load.
        return 1.f / float(kernelSize.x * kernelSize.y);
    }
};

// Output columns per thread for interleaved launches. Same heuristic as Conv2D: scalar/uchar4 get a
// wide tile to saturate the byte pipe; float3 stays narrow to cap register pressure. float4 never
// reaches a tiled launch (see AverageBlurVarShapeInterleaved) but stays in the table for Conv2D
// parity.
template<typename D>
constexpr int kOutputs = std::is_same_v<D, filter::uchar> || std::is_same_v<D, uchar4>
                           ? 8
                           : (std::is_same_v<D, float3> || std::is_same_v<D, float4> ? 2 : 4);

// Planar dispatch is per base type, so only the scalar (byte) case gets the wider tile.
template<typename D>
constexpr int kPlanarOutputs = std::is_same_v<D, filter::uchar> ? 8 : 4;

// KW/KH > 0 makes the window size a compile-time constant so the tap loops unroll; 0 keeps them
// dynamic.
template<int KW, int KH, bool IsPlanar, class Work, class W>
inline __device__ void AccumulateBox(Work &res, const W &src, int x, int y, int plane, int sample, int2 kernelSize,
                                     int2 anchor, float kernelValue)
{
    if constexpr (KW > 0)
    {
#pragma unroll
        for (int i = 0; i < KH; ++i)
        {
            const int cy = y - anchor.y + i;

#pragma unroll
            for (int j = 0; j < KW; ++j)
            {
                res = res + src[filter::SrcCoord<IsPlanar>(x - anchor.x + j, cy, plane, sample)] * kernelValue;
            }
        }
    }
    else
    {
        for (int i = 0; i < kernelSize.y; ++i)
        {
            const int cy = y - anchor.y + i;

            for (int j = 0; j < kernelSize.x; ++j)
            {
                res = res + src[filter::SrcCoord<IsPlanar>(x - anchor.x + j, cy, plane, sample)] * kernelValue;
            }
        }
    }
}

// Pixels whose whole window lies inside the source read through the raw wrapper, skipping the border
// wrapper's per-tap coordinate clamp.
template<int KW, int KH, bool InteriorFastPath, bool IsPlanar, class Work, class Border, class Raw>
inline __device__ void BoxTaps(Work &res, const Border &borderSrc, const Raw &src, int x, int y, int plane, int sample,
                               int2 kernelSize, int2 anchor, float kernelValue, bool isInterior)
{
    if constexpr (InteriorFastPath)
    {
        if (isInterior)
        {
            AccumulateBox<KW, KH, IsPlanar>(res, src, x, y, plane, sample, kernelSize, anchor, kernelValue);
            return;
        }
    }
    AccumulateBox<KW, KH, IsPlanar>(res, borderSrc, x, y, plane, sample, kernelSize, anchor, kernelValue);
}

inline __device__ bool IsWindowInterior(int x, int y, int width, int height, int2 kernelSize, int2 anchor)
{
    return x >= anchor.x && x < width - (kernelSize.x - anchor.x - 1) && y >= anchor.y
        && y < height - (kernelSize.y - anchor.y - 1);
}

// Host-callable as well as device-callable: the tensor submit resolves this on the host to pick the
// unrolled 5x5 instantiation, while the var-shape kernel evaluates it per image on the device.
inline __host__ __device__ bool IsCenteredFiveByFive(int2 kernelSize, int2 anchor)
{
    return kernelSize.x == 5 && kernelSize.y == 5 && anchor.x == 2 && anchor.y == 2;
}

// `VarShapePath` moves the 5x5 test into the kernel, where the per-image window lives. A tensor
// submit resolves it on the host and instantiates KW/KH directly.
// A planar submit walks its planes inside the thread, the way the legacy planar kernel did, rather
// than folding (sample, plane) into gridDim.z the way the tiled path does: one thread already holds
// the window coordinates, and every plane reuses them. `channels` is ignored when the destination is
// interleaved, where the body collapses to the single plane-0 store it had before.
template<int KW, int KH, bool VarShapePath, class Border, class Raw, class Dst, class Params>
__global__ void AverageBlur2D(Border borderSrc, Raw src, Dst dst, Params params, int channels)
{
    using T                = typename Dst::ValueType;
    using work_type        = cuda::ConvertBaseTypeTo<float, T>;
    constexpr bool kPlanar = Dst::kPlanar;

    const int x         = blockIdx.x * blockDim.x + threadIdx.x;
    const int y         = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = blockIdx.z;

    const int width  = dst.Width(batch_idx, 0);
    const int height = dst.Height(batch_idx, 0);

    if (x >= width || y >= height)
        return;

    // The interior shortcut samples through the raw wrap, so the interior test has to be judged
    // against the source extent. AverageBlur always produces output of the same spatial size as
    // the input, so the destination extent (already in registers as width/height) equals the source
    // extent. Loading src.width/height again via the raw wrap costs an extra global load plus the
    // BSSY/BSYNC convergence barrier the compiler inserts around the conditional loading path --
    // roughly 10 instructions of prologue overhead per thread, measured as +14-16% on A100. Using
    // the destination extent is correct here and matches the legacy kernel's structure.
    const int2  kernelSize  = (KW > 0) ? int2{KW, KH} : params.Size(batch_idx);
    const int2  anchor      = (KW > 0) ? int2{KW / 2, KH / 2} : params.Anchor(batch_idx, kernelSize);
    const bool  isInterior  = IsWindowInterior(x, y, width, height, kernelSize, anchor);
    const float kernelValue = static_cast<float>(1.0 / (kernelSize.x * kernelSize.y));

    // Everything above is plane-invariant, so the window shape is resolved once and the plane loop
    // carries only the taps and the store. Re-testing the window inside the loop costs more the
    // fewer planes there are to amortize it over, which is why a 3-plane image felt it worst.
    const int planes = kPlanar ? channels : 1;

    auto emitPlanes = [&](auto kw, auto kh, auto interiorFast, float weight)
    {
        constexpr int  KWv = decltype(kw)::value;
        constexpr int  KHv = decltype(kh)::value;
        constexpr bool IFv = decltype(interiorFast)::value;

        auto emitShared = [&](const auto &wrap)
        {
            for (int plane = 0; plane < planes; ++plane)
            {
                work_type res = cuda::SetAll<work_type>(0);

                AccumulateBox<KWv, KHv, kPlanar>(res, wrap, x, y, plane, batch_idx, kernelSize, anchor, weight);

                *dst.Ptr(batch_idx, plane, y, x) = cuda::SaturateCast<T>(res);
            }
        };

        // Where the body holds a single store, give each side of the interior test its own copy of
        // it, the way the legacy kernel was written. Funnelling both into one store after the
        // branch reads better but costs real time: the compiler has to reconverge the two paths
        // before that store, so in a warp straddling an edge the interior lanes wait for the border
        // lanes to finish their clamped taps instead of retiring. Duplicating it is worth 3.7
        // points on the scalar 5x5 tensor submit.
        //
        // A planar body stores once per plane, so the same duplication would copy the whole plane
        // loop. That pushes the var-shape scalar kernels from 32 to 40 registers -- 100% to 75%
        // occupancy on sm_80 -- and measured 7-9% slower on the planar direct path. There the
        // shared store is the cheaper of the two, so keep it.
        if constexpr (IFv && !kPlanar)
        {
            if (isInterior)
            {
                emitShared(src);
                return;
            }
            emitShared(borderSrc);
        }
        else
        {
            for (int plane = 0; plane < planes; ++plane)
            {
                work_type res = cuda::SetAll<work_type>(0);

                BoxTaps<KWv, KHv, IFv, kPlanar>(res, borderSrc, src, x, y, plane, batch_idx, kernelSize, anchor, weight,
                                                isInterior);

                *dst.Ptr(batch_idx, plane, y, x) = cuda::SaturateCast<T>(res);
            }
        }
    };

    if constexpr (VarShapePath)
    {
        if (IsCenteredFiveByFive(kernelSize, anchor))
        {
            emitPlanes(std::integral_constant<int, 5>{}, std::integral_constant<int, 5>{}, std::true_type{},
                       static_cast<float>(1.0 / 25));
            return;
        }
    }

    emitPlanes(std::integral_constant<int, KW>{}, std::integral_constant<int, KH>{},
               std::bool_constant < (KW > 0) || !VarShapePath > {}, kernelValue);
}

// Reuse the overlapping input columns while preserving each output's tap/FMA order.
template<class Border, class Raw, class Dst>
__global__ void AverageBlur5x5X4(Border borderSrc, Raw src, Dst dst)
{
    using T          = typename Dst::ValueType;
    const int x      = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    const int y      = blockIdx.y * blockDim.y + threadIdx.y;
    const int sample = blockIdx.z;
    const int width  = dst.Width(sample, 0);
    const int height = dst.Height(sample, 0);
    if (x >= width || y >= height)
        return;

    constexpr float weight = static_cast<float>(1.0 / 25);
    if (x >= 2 && x < width - 5 && y >= 2 && y < height - 2)
    {
        float result[4] = {};
#pragma unroll
        for (int i = 0; i < 5; ++i)
        {
            float row[8];
#pragma unroll
            for (int j = 0; j < 8; ++j) row[j] = src[int3{x - 2 + j, y - 2 + i, sample}];
#pragma unroll
            for (int k = 0; k < 4; ++k)
#pragma unroll
                for (int j = 0; j < 5; ++j) result[k] = result[k] + row[k + j] * weight;
        }
#pragma unroll
        for (int k = 0; k < 4; ++k) *dst.Ptr(sample, 0, y, x + k) = cuda::SaturateCast<T>(result[k]);
    }
    else
    {
#pragma unroll
        for (int k = 0; k < 4; ++k)
        {
            if (x + k >= width)
                return;
            float result = 0;
            AccumulateBox<5, 5, false>(result, borderSrc, x + k, y, 0, sample, int2{5, 5}, int2{2, 2}, weight);
            *dst.Ptr(sample, 0, y, x + k) = cuda::SaturateCast<T>(result);
        }
    }
}

template<typename T, NVCVBorderType B>
void AverageBlurTensor(const nvcv::TensorDataStridedCuda &inData, const nvcv::TensorDataStridedCuda &outData,
                       int2 kernelSize, int2 kernelAnchor, float borderValue, cudaStream_t stream)
{
    auto inAccess  = nvcv::TensorDataAccessStridedImagePlanar::Create(inData);
    auto outAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(outData);
    cvcuda::priv::RequireImageAccess(inAccess, "Input");
    cvcuda::priv::RequireImageAccess(outAccess, "Output");

    filter::RequireInt32Addressable(*inAccess, *outAccess);

    // The interior shortcut samples through the raw source wrap rather than the border wrap, so a
    // larger output would read past the input extent instead of clamping.
    filter::RequireSameImageShape(*inAccess, *outAccess);

    const int2 dstSize{outAccess->numCols(), outAccess->numRows()};

    dim3 block(kBlockWidth, kBlockHeight);
    dim3 grid(util::DivUp(dstSize.x, kBlockWidth), util::DivUp(dstSize.y, kBlockHeight), outAccess->numSamples());

    auto borderSrc = cuda::CreateBorderWrapNHW<const T, B, int32_t>(inData, cuda::SetAll<T>(borderValue));
    auto src       = cuda::CreateTensorWrapNHW<const T, int32_t>(inData);
    auto dst       = filter::MakeUniformDst(cuda::CreateTensorWrapNHW<T, int32_t>(outData), dstSize);

    UniformBoxParams params{kernelSize, kernelAnchor};

    if constexpr (cuda::NumElements<T> == 1)
    {
        if (IsCenteredFiveByFive(kernelSize, kernelAnchor))
        {
            if constexpr (B == NVCV_BORDER_REPLICATE && std::is_same_v<T, filter::uchar>)
            {
                dim3 reuseGrid(util::DivUp(dstSize.x, kBlockWidth * 4), grid.y, grid.z);
                AverageBlur5x5X4<<<reuseGrid, block, 0, stream>>>(borderSrc, src, dst);
            }
            else
            {
                AverageBlur2D<5, 5, false><<<grid, block, 0, stream>>>(borderSrc, src, dst, params, 1);
            }
            NVCV_CHECK_THROW(cudaGetLastError());
            return;
        }
    }

    if constexpr (std::is_same_v<T, uchar3>)
    {
        if (kernelSize.x == 5 && kernelSize.y == 5)
        {
            constexpr int    outputs      = kOutputs<T>;
            constexpr size_t kSharedBytes = filter::FixedSharedBytes<outputs, 5, T, kBlockWidth, kBlockHeight>();
            static_assert(kSharedBytes <= filter::kMaxSharedBytes);

            dim3 tiledGrid(util::DivUp(dstSize.x, kBlockWidth * outputs), util::DivUp(dstSize.y, kBlockHeight),
                           outAccess->numSamples());
            filter::Filter2DTiledFixed<outputs, 5, 5, true>
                <<<tiledGrid, block, kSharedBytes, stream>>>(borderSrc, dst, params);
            NVCV_CHECK_THROW(cudaGetLastError());
            return;
        }
    }

    AverageBlur2D<0, 0, false><<<grid, block, 0, stream>>>(borderSrc, src, dst, params, 1);
    NVCV_CHECK_THROW(cudaGetLastError());
}

template<typename T, NVCVBorderType B>
void AverageBlurVarShapeInterleaved(const nvcv::ImageBatchVarShapeDataStridedCuda &inData,
                                    const nvcv::ImageBatchVarShapeDataStridedCuda &outData,
                                    const nvcv::TensorDataStridedCuda             &kernelSizeData,
                                    const nvcv::TensorDataStridedCuda &kernelAnchorData, int2 maxKernelSize,
                                    float borderValue, cudaStream_t stream)
{
    cuda::BorderVarShapeWrap<const T, B>  src(inData, cuda::SetAll<T>(borderValue));
    cuda::ImageBatchVarShapeWrap<const T> rawSrc(inData);

    auto dst = filter::MakeVarShapeDst(cuda::ImageBatchVarShapeWrap<T>(outData));

    VarShapeBoxCoefs coefs{cuda::Tensor1DWrap<int2, int32_t>(kernelSizeData),
                           cuda::Tensor1DWrap<int2, int32_t>(kernelAnchorData)};

    dim3 block(kBlockWidth, kBlockHeight);
    dim3 grid(util::DivUp(inData.maxSize().w, kBlockWidth), util::DivUp(inData.maxSize().h, kBlockHeight),
              outData.numImages());

    // One thread per pixel, taps read straight through the raw wrap wherever the window is interior.
    auto direct = [&]
    {
        AverageBlur2D<0, 0, true><<<grid, block, 0, stream>>>(src, rawSrc, dst, coefs, 1);
    };

    // An interleaved submit stages whole pixels, so the trade is decided by the pixel. See
    // FilterSchedulePolicy.hpp for the measurements behind the rule.
    int sm = 0;
    NVCV_CHECK_THROW(cvcuda::priv::GetCurrentDeviceSM(sm));

    if (cvcuda::priv::FilterScheduleForSM(sm, sizeof(cuda::BaseType<T>), cuda::NumElements<T>, /*planar=*/false)
        == cvcuda::priv::FilterSchedule::kTiled)
    {
        constexpr int outputs = kOutputs<T>;

        dim3 tiledGrid(util::DivUp(inData.maxSize().w, kBlockWidth * outputs),
                       util::DivUp(inData.maxSize().h, kBlockHeight), outData.numImages());

        const size_t sharedBytes = filter::TiledSharedBytes(outputs, block, maxKernelSize, sizeof(T));

        const bool launched
            = filter::TryCommonKernelSizes(maxKernelSize,
                                           [&](auto ksize)
                                           {
                                               constexpr int kSize = decltype(ksize)::value;
                                               if (sharedBytes <= filter::kMaxSharedBytes)
                                               {
                                                   filter::Filter2DTiledFixed<outputs, kSize, kSize, false>
                                                       <<<tiledGrid, block, sharedBytes, stream>>>(src, dst, coefs);
                                               }
                                               else
                                               {
                                                   direct();
                                               }
                                           });

        if (!launched)
        {
            direct();
        }
    }
    else
    {
        direct();
    }

    NVCV_CHECK_THROW(cudaGetLastError());
}

template<typename T, NVCVBorderType B>
void AverageBlurVarShapePlanar(const nvcv::ImageBatchVarShapeDataStridedCuda &inData,
                               const nvcv::ImageBatchVarShapeDataStridedCuda &outData,
                               const nvcv::TensorDataStridedCuda             &kernelSizeData,
                               const nvcv::TensorDataStridedCuda &kernelAnchorData, int channels, int2 maxKernelSize,
                               float borderValue, cudaStream_t stream)
{
    cuda::BorderVarShapeWrap<const T, B>  src(inData, cuda::SetAll<T>(borderValue));
    cuda::ImageBatchVarShapeWrap<const T> rawSrc(inData);

    // Fold (sample, plane) into gridDim.z so every block owns one plane and can
    // stage a full (outputs × block + halo) × (block + halo) tile.
    auto dst = filter::MakeVarShapePlanarDst(cuda::ImageBatchVarShapeWrap<T>(outData), channels);

    VarShapeBoxCoefs coefs{cuda::Tensor1DWrap<int2, int32_t>(kernelSizeData),
                           cuda::Tensor1DWrap<int2, int32_t>(kernelAnchorData)};

    constexpr int outputs = kPlanarOutputs<T>;

    dim3          block(kBlockWidth, kBlockHeight);
    const int64_t tiledGridZ = static_cast<int64_t>(outData.numImages()) * channels;

    dim3 fallbackGrid(util::DivUp(outData.maxSize().w, kBlockWidth), util::DivUp(outData.maxSize().h, kBlockHeight),
                      outData.numImages());

    auto untiled = [&]
    {
        filter::Filter2DPlanar<<<fallbackGrid, block, 0, stream>>>(src, dst, coefs, channels);
    };

    // One thread per output pixel, walking the planes itself and taking the raw wrap wherever the
    // window is interior -- the schedule the legacy planar kernel ran. Unlike `untiled` it keeps the
    // interior fast path and the unrolled 5x5, which is the whole reason it is competitive.
    auto direct = [&]
    {
        AverageBlur2D<0, 0, true><<<fallbackGrid, block, 0, stream>>>(src, rawSrc, dst, coefs, channels);
    };

    // Past the CUDA grid-z cap use Filter2DPlanar, which walks planes inside one
    // thread rather than folding them into gridDim.z.
    if (tiledGridZ > filter::kMaxGridZ)
    {
        untiled();
        NVCV_CHECK_THROW(cudaGetLastError());
        return;
    }

    // A plane is scalar, so the staged tile holds base-type values and the trade is decided by that
    // width, not the pixel's. See FilterSchedulePolicy.hpp for the measurements.
    int sm = 0;
    NVCV_CHECK_THROW(cvcuda::priv::GetCurrentDeviceSM(sm));

    if (cvcuda::priv::FilterScheduleForSM(sm, sizeof(cuda::BaseType<T>), cuda::NumElements<T>, /*planar=*/true)
        == cvcuda::priv::FilterSchedule::kDirect)
    {
        direct();
        NVCV_CHECK_THROW(cudaGetLastError());
        return;
    }

    dim3 tiledGrid(util::DivUp(outData.maxSize().w, kBlockWidth * outputs),
                   util::DivUp(outData.maxSize().h, kBlockHeight), static_cast<unsigned int>(tiledGridZ));

    const size_t sharedBytes = filter::TiledSharedBytes(outputs, block, maxKernelSize, sizeof(T));

    const bool launched
        = filter::TryCommonKernelSizes(maxKernelSize,
                                       [&](auto ksize)
                                       {
                                           constexpr int kSize = decltype(ksize)::value;
                                           if (sharedBytes <= filter::kMaxSharedBytes)
                                           {
                                               filter::Filter2DTiledFixed<outputs, kSize, kSize, false>
                                                   <<<tiledGrid, block, sharedBytes, stream>>>(src, dst, coefs);
                                           }
                                           else
                                           {
                                               untiled();
                                           }
                                       });

    if (!launched)
    {
        untiled();
    }

    NVCV_CHECK_THROW(cudaGetLastError());
}

} // namespace

AverageBlur::AverageBlur(nvcv::Size2D maxKernelSize, int maxBatchSize)
    : m_maxKernelSize(maxKernelSize)
    , m_maxBatchSize(maxBatchSize)
{
}

void AverageBlur::operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out,
                             nvcv::Size2D kernelSize, int2 kernelAnchor, NVCVBorderType borderMode) const
{
    CVCUDA_NVTX_RANGE("cvcuda::AverageBlur::operator()[Tensor]");
    auto inData  = RequireCudaTensor(in, "Input");
    auto outData = RequireCudaTensor(out, "Output");

    RequireSameDtype(*inData, *outData);

    filter::ValidateTensorLayout(inData->layout());
    filter::ValidateTensorLayout(outData->layout());
    RequireSameLayout(*inData, *outData);

    filter::ValidateBorderMode(borderMode);

    const filter::ElemType elemType = filter::ClassifyFilterElemType(inData->dtype());

    if (!(kernelSize.w > 0 && kernelSize.w % 2 == 1 && kernelSize.w <= m_maxKernelSize.w && kernelSize.h > 0
          && kernelSize.h % 2 == 1 && kernelSize.h <= m_maxKernelSize.h))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid kernel size %dx%d", kernelSize.w,
                              kernelSize.h);
    }

    if (!(kernelAnchor.x == -1 || (kernelAnchor.x >= 0 && kernelAnchor.x < kernelSize.w)))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid kernel anchor x %d", kernelAnchor.x);
    }

    if (!(kernelAnchor.y == -1 || (kernelAnchor.y >= 0 && kernelAnchor.y < kernelSize.h)))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid kernel anchor y %d", kernelAnchor.y);
    }

    auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*inData);
    NVCV_ASSERT(inAccess);

    const int channels = inAccess->numChannels();
    filter::ValidateChannelCount(channels);

    if (kernelAnchor.x < 0)
        kernelAnchor.x = kernelSize.w / 2;
    if (kernelAnchor.y < 0)
        kernelAnchor.y = kernelSize.h / 2;

    const int2      size{kernelSize.w, kernelSize.h};
    constexpr float borderValue = 0.f;

    // A packed planar tensor is (N*C) single-channel images laid end to end, so the interleaved
    // scalar kernel covers it once the view is flattened.
    auto planarViews = PlanarSingleChannelViews(*inData, *outData);

    const nvcv::TensorDataStridedCuda &src = planarViews ? planarViews->first : *inData;
    const nvcv::TensorDataStridedCuda &dst = planarViews ? planarViews->second : *outData;

    filter::RunTypeBorderSwitch(elemType, planarViews ? 1 : channels, borderMode,
                                [&](auto value, auto border) {
                                    AverageBlurTensor<decltype(value), decltype(border)::value>(
                                        src, dst, size, kernelAnchor, borderValue, stream);
                                });
}

void AverageBlur::operator()(cudaStream_t stream, const nvcv::ImageBatchVarShape &in,
                             const nvcv::ImageBatchVarShape &out, const nvcv::Tensor &kernelSize,
                             const nvcv::Tensor &kernelAnchor, NVCVBorderType borderMode) const
{
    CVCUDA_NVTX_RANGE("cvcuda::AverageBlur::operator()[ImageBatchVarShape]");
    auto inData           = RequireCudaVarShape(in, stream, "Input");
    auto outData          = RequireCudaVarShape(out, stream, "Output");
    auto kernelSizeData   = RequireCudaTensor(kernelSize, "Kernel size");
    auto kernelAnchorData = RequireCudaTensor(kernelAnchor, "Kernel anchor");

    RequireUniqueFormat(*inData, "input");
    RequireUniqueFormat(*outData, "output");

    if (m_maxBatchSize <= 0 || inData->numImages() > m_maxBatchSize)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid maximum batch size");
    }

    // The dispatch below derives the element type and channel count from the input alone and then
    // wraps the output with them, and indexes both parameter tensors up to numImages.
    RequireSameNumImages(*inData, *outData);
    RequireSameFormat(*inData, *outData);

    cvcuda::priv::RequireSameVarShapeImageSizes(in, out);

    cvcuda::priv::RequirePerImageParam(*kernelSizeData, nvcv::TYPE_2S32, inData->numImages(), "Kernel size");
    cvcuda::priv::RequirePerImageParam(*kernelAnchorData, nvcv::TYPE_2S32, inData->numImages(), "Kernel anchor");

    const filter::BatchLayout inLayout  = filter::ClassifyBatchLayout(*inData);
    const filter::BatchLayout outLayout = filter::ClassifyBatchLayout(*outData);
    filter::RequireSameBatchLayout(inLayout, outLayout);

    filter::ValidateBorderMode(borderMode);

    const filter::ElemType elemType = filter::ClassifyFilterElemType(inData->uniqueFormat());

    const int channels = inData->uniqueFormat().numChannels();
    filter::ValidateChannelCount(channels);

    constexpr float borderValue = 0.f;

    if (inLayout.planar)
    {
        filter::RequirePlanarGridZ(outData->numImages());

        filter::RunBaseTypeBorderSwitch(elemType, borderMode,
                                        [&](auto base, auto border)
                                        {
                                            AverageBlurVarShapePlanar<decltype(base), decltype(border)::value>(
                                                *inData, *outData, *kernelSizeData, *kernelAnchorData, channels,
                                                int2{m_maxKernelSize.w, m_maxKernelSize.h}, borderValue, stream);
                                        });
        return;
    }

    filter::RunTypeBorderSwitch(elemType, channels, borderMode,
                                [&](auto value, auto border)
                                {
                                    AverageBlurVarShapeInterleaved<decltype(value), decltype(border)::value>(
                                        *inData, *outData, *kernelSizeData, *kernelAnchorData,
                                        int2{m_maxKernelSize.w, m_maxKernelSize.h}, borderValue, stream);
                                });
}

} // namespace cvcuda::priv
