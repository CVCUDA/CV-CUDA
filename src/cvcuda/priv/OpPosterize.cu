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

#include "Nvtx.hpp"
#include "OpPosterize.hpp"

#include <cvcuda/cuda_tools/ImageBatchVarShapeWrap.hpp>
#include <cvcuda/cuda_tools/MathOps.hpp>
#include <cvcuda/cuda_tools/StaticCast.hpp>
#include <cvcuda/cuda_tools/TensorWrap.hpp>
#include <cvcuda/cuda_tools/TypeTraits.hpp>
#include <nvcv/DataType.hpp>
#include <nvcv/Exception.hpp>
#include <nvcv/TensorData.hpp>
#include <nvcv/TensorDataAccess.hpp>
#include <nvcv/TensorLayout.hpp>
#include <nvcv/util/CheckError.hpp>
#include <nvcv/util/Math.hpp>

#include <type_traits>

namespace cuda = nvcv::cuda;
namespace util = nvcv::util;

namespace {

// out = in & mask, per channel component (mask keeps the top `bits` bits of each value).
template<typename T, typename BT>
inline __device__ T PosterizeElem(T pixel, BT mask)
{
    static constexpr int numChannels = cuda::NumElements<T>;
    T                    out{};
#pragma unroll
    for (int c = 0; c < numChannels; ++c)
    {
        cuda::GetElement(out, c) = static_cast<BT>(cuda::GetElement(pixel, c) & mask);
    }
    return out;
}

template<bool IsPlanar>
inline __device__ std::conditional_t<IsPlanar, int4, int3> GetCoordForLayout(int3 nhwCoord, int p)
{
    if constexpr (!IsPlanar)
    {
        return nhwCoord;
    }
    else
    {
        return {nhwCoord.x, nhwCoord.y, p, nhwCoord.z};
    }
}

template<bool IsPlanar, class SrcWrapper, class DstWrapper, typename BT>
inline __device__ void DoPosterize(SrcWrapper src, DstWrapper dst, const int2 size, const int p, BT mask)
{
    using SrcT                       = std::remove_const_t<typename SrcWrapper::ValueType>;
    using DstT                       = typename DstWrapper::ValueType;
    static constexpr int numChannels = cuda::NumElements<SrcT>;
    static_assert(numChannels == cuda::NumElements<DstT>);
    static_assert(!IsPlanar || numChannels == 1);

    int3 nhwCoord = cuda::StaticCast<int>(blockIdx * blockDim + threadIdx);
    if (nhwCoord.x >= size.x || nhwCoord.y >= size.y)
    {
        return;
    }
    auto coord = GetCoordForLayout<IsPlanar>(nhwCoord, p);
    dst[coord] = PosterizeElem<DstT>(src[coord], mask);
}

// Posterize kernel -----------------------------------------------------------------------

// Tensor variant
template<bool isPlanar, class SrcWrapper, class DstWrapper, typename BT>
__global__ void Posterize(SrcWrapper src, DstWrapper dst, int2 size, int numPlanes, BT mask)
{
    assert(isPlanar || numPlanes == 1);
    if constexpr (!isPlanar)
    {
        DoPosterize<isPlanar>(src, dst, size, 0, mask);
    }
    else
    {
        for (int p = 0; p < numPlanes; p++)
        {
            DoPosterize<isPlanar>(src, dst, size, p, mask);
        }
    }
}

// VarShape variant
template<bool isPlanar, class SrcWrapper, class DstWrapper, typename BT>
__global__ void Posterize(SrcWrapper src, DstWrapper dst, int numPlanes, BT mask)
{
    assert(isPlanar || numPlanes == 1);
    int  z = blockIdx.z;
    int2 size{dst.width(z), dst.height(z)};

    if constexpr (!isPlanar)
    {
        DoPosterize<isPlanar>(src, dst, size, 0, mask);
    }
    else
    {
        for (int p = 0; p < numPlanes; p++)
        {
            DoPosterize<isPlanar>(src, dst, size, p, mask);
        }
    }
}

// Compute the posterize mask for a base type given bits-to-keep. Uses a wide intermediate so the
// shift never hits the type-width UB (bits == 0 -> mask 0; bits == W -> all ones).
template<typename BT>
inline BT PosterizeMask(int bits)
{
    constexpr int W = static_cast<int>(sizeof(BT) * 8);
    uint32_t      m;
    if (bits <= 0)
    {
        m = 0u;
    }
    else if (bits >= W)
    {
        m = ~0u;
    }
    else
    {
        m = ~((1u << (W - bits)) - 1u);
    }
    return static_cast<BT>(m);
}

// Vectorized planar / 1-channel kernels --------------------------------------------------
//
// The scalar kernels above move one element/thread/plane, leaving the planar and 1-channel-interleaved
// paths memory-latency bound (long-scoreboard stalls, low BWUtil) -- each warp has a single outstanding
// load. These map each (sample, plane) to grid.z and have each thread issue NGROUP wide vector loads
// (uchar4 / ushort4) before compute, raising memory-level parallelism. Pure bitwise (in & mask), so no
// extra compute; per element bit-identical to PosterizeElem. Modeled on legacy/normalize_planar.cuh;
// caller guards sizeof(Vec4)-aligned base+strides with a scalar fallback; per-thread tail handles width%4.
template<typename T, int Size = sizeof(T)>
struct PosterizeVec4Type;

template<typename T>
struct PosterizeVec4Type<T, 1>
{
    using type = uchar4;
};

template<typename T>
struct PosterizeVec4Type<T, 2>
{
    using type = ushort4;
};

template<int NGROUP, typename BT>
__global__ void PosterizePlanarVec4Kernel(cuda::Tensor4DWrap<const BT, int32_t> src,
                                          cuda::Tensor4DWrap<BT, int32_t> dst, int4 inout_size, BT mask)
{
    const int g0      = blockIdx.x * blockDim.x * NGROUP + threadIdx.x;
    const int src_y   = blockIdx.y * blockDim.y + threadIdx.y;
    const int nc      = blockIdx.z;
    const int batch   = nc / inout_size.y;
    const int channel = nc % inout_size.y;
    const int width   = inout_size.w;

    if (g0 * 4 >= width || src_y >= inout_size.z)
    {
        return;
    }

    using Vec4 = typename PosterizeVec4Type<BT>::type;

    int  cx[NGROUP];
    bool full[NGROUP];
    Vec4 in4[NGROUP];
#pragma unroll
    for (int i = 0; i < NGROUP; ++i)
    {
        cx[i]   = (g0 + i * blockDim.x) * 4;
        full[i] = cx[i] + 4 <= width;
        if (full[i])
        {
            in4[i] = *reinterpret_cast<const Vec4 *>(src.ptr(batch, channel, src_y, cx[i]));
        }
    }

#pragma unroll
    for (int i = 0; i < NGROUP; ++i)
    {
        if (full[i])
        {
            Vec4 out4;
            out4.x                                                           = static_cast<BT>(in4[i].x & mask);
            out4.y                                                           = static_cast<BT>(in4[i].y & mask);
            out4.z                                                           = static_cast<BT>(in4[i].z & mask);
            out4.w                                                           = static_cast<BT>(in4[i].w & mask);
            *reinterpret_cast<Vec4 *>(dst.ptr(batch, channel, src_y, cx[i])) = out4;
        }
        else if (cx[i] < width)
        {
            for (int x = cx[i]; x < width; ++x)
            {
                *dst.ptr(batch, channel, src_y, x) = static_cast<BT>(*src.ptr(batch, channel, src_y, x) & mask);
            }
        }
    }
}

template<int NGROUP, typename BT>
__global__ void PosterizePlanarVarShapeVec4Kernel(cuda::ImageBatchVarShapeWrap<const BT> src,
                                                  cuda::ImageBatchVarShapeWrap<BT> dst, int num_channels, BT mask)
{
    const int g0      = blockIdx.x * blockDim.x * NGROUP + threadIdx.x;
    const int dst_y   = blockIdx.y * blockDim.y + threadIdx.y;
    const int nc      = blockIdx.z;
    const int batch   = nc / num_channels;
    const int channel = nc % num_channels;
    const int width   = dst.width(batch, channel);

    if (g0 * 4 >= width || dst_y >= dst.height(batch, channel))
    {
        return;
    }

    using Vec4 = uchar4; // 1-byte planes only (caller guards sizeof(BT) == 1)

    int  cx[NGROUP];
    bool full[NGROUP];
    Vec4 in4[NGROUP];
#pragma unroll
    for (int i = 0; i < NGROUP; ++i)
    {
        cx[i]   = (g0 + i * blockDim.x) * 4;
        full[i] = cx[i] + 4 <= width;
        if (full[i])
        {
            in4[i] = *reinterpret_cast<const Vec4 *>(src.ptr(batch, channel, dst_y, cx[i]));
        }
    }

#pragma unroll
    for (int i = 0; i < NGROUP; ++i)
    {
        if (full[i])
        {
            Vec4 out4;
            out4.x                                                           = static_cast<BT>(in4[i].x & mask);
            out4.y                                                           = static_cast<BT>(in4[i].y & mask);
            out4.z                                                           = static_cast<BT>(in4[i].z & mask);
            out4.w                                                           = static_cast<BT>(in4[i].w & mask);
            *reinterpret_cast<Vec4 *>(dst.ptr(batch, channel, dst_y, cx[i])) = out4;
        }
        else if (cx[i] < width)
        {
            for (int x = cx[i]; x < width; ++x)
            {
                *dst.ptr(batch, channel, dst_y, x) = static_cast<BT>(*src.ptr(batch, channel, dst_y, x) & mask);
            }
        }
    }
}

// Run Posterize kernel -------------------------------------------------------------------

template<bool isPlanar, typename ValueT, class SrcData, class DstData>
inline void RunPosterize(cudaStream_t stream, const SrcData &srcData, const DstData &dstData, int bits)
{
    using BT        = cuda::BaseType<ValueT>;
    constexpr int W = static_cast<int>(sizeof(BT) * 8);
    if (bits < 0 || bits > W)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Posterize 'bits' must be in [0, %d] for this data type", W);
    }
    const BT mask = PosterizeMask<BT>(bits);

    dim3 block(32, 4, 1);
    if constexpr (std::is_same_v<SrcData, nvcv::TensorDataStridedCuda>)
    {
        auto srcAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(srcData);
        auto dstAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(dstData);
        int2 size      = cuda::StaticCast<int>(long2{srcAccess->numCols(), srcAccess->numRows()});

        // Each sample maps to one grid.z block (planar channels looped inside); grid.z cap 65535.
        if (srcAccess->numSamples() > 65535)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Batch size exceeds the CUDA grid.z limit of 65535");
        }
        dim3 grid(util::DivUp(size.x, block.x), util::DivUp(size.y, block.y), srcAccess->numSamples());

        int64_t inMaxStride  = srcAccess->sampleStride() * srcAccess->numSamples();
        int64_t outMaxStride = dstAccess->sampleStride() * dstAccess->numSamples();
        if (std::max(inMaxStride, outMaxStride) > cuda::TypeTraits<int32_t>::max)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_OVERFLOW, "Input or output size exceeds %d. Tensor is too large.",
                                  cuda::TypeTraits<int32_t>::max);
        }
        using StrideType = int32_t;

        if constexpr (!isPlanar)
        {
            bool launchedVec = false;
            // 1-channel interleaved (U16) is byte-identical to a single plane and latency-bound; route
            // it through the vectorized planar kernel (C == 1). Multi-channel interleaved (uchar3/
            // uchar4) already moves 3-4 B/thread at the bandwidth ridge -> keep the scalar kernel.
            if constexpr (cuda::NumElements<ValueT> == 1)
            {
                using Vec4            = typename PosterizeVec4Type<BT>::type;
                constexpr int NGROUP  = 4;
                const int64_t sStride = srcAccess->sampleStride(), rStride = srcAccess->rowStride();
                const int64_t dsStride = dstAccess->sampleStride(), drStride = dstAccess->rowStride();
                const bool    aligned = reinterpret_cast<uintptr_t>(srcData.basePtr()) % sizeof(Vec4) == 0
                                  && reinterpret_cast<uintptr_t>(dstData.basePtr()) % sizeof(Vec4) == 0
                                  && sStride % sizeof(Vec4) == 0 && rStride % sizeof(Vec4) == 0
                                  && dsStride % sizeof(Vec4) == 0 && drStride % sizeof(Vec4) == 0;
                if (aligned)
                {
                    auto srcV = cuda::Tensor4DWrap<const BT, StrideType>(srcData.basePtr(), static_cast<int>(sStride),
                                                                         static_cast<int>(sStride),
                                                                         static_cast<int>(rStride));
                    auto dstV
                        = cuda::Tensor4DWrap<BT, StrideType>(dstData.basePtr(), static_cast<int>(dsStride),
                                                             static_cast<int>(dsStride), static_cast<int>(drStride));
                    dim3 vgrid(util::DivUp(util::DivUp(size.x, 4), static_cast<int>(block.x) * NGROUP),
                               util::DivUp(size.y, block.y), srcAccess->numSamples());
                    PosterizePlanarVec4Kernel<NGROUP, BT><<<vgrid, block, 0, stream>>>(
                        srcV, dstV, int4{static_cast<int>(srcAccess->numSamples()), 1, size.y, size.x}, mask);
                    launchedVec = true;
                }
            }
            if (!launchedVec)
            {
                auto src = cuda::CreateTensorWrapNHW<const ValueT, StrideType>(srcData);
                auto dst = cuda::CreateTensorWrapNHW<ValueT, StrideType>(dstData);
                Posterize<isPlanar><<<grid, block, 0, stream>>>(src, dst, size, 1, mask);
            }
        }
        else
        {
            const int numPlanes   = srcAccess->numPlanes();
            const int numSamples  = static_cast<int>(srcAccess->numSamples());
            bool      launchedVec = false;
            if constexpr (sizeof(BT) == 1 || sizeof(BT) == 2)
            {
                using Vec4            = typename PosterizeVec4Type<BT>::type;
                constexpr int NGROUP  = 4;
                const int64_t planes  = static_cast<int64_t>(numSamples) * numPlanes;
                const int64_t sStride = srcAccess->sampleStride(), pStride = srcAccess->planeStride(),
                              rStride  = srcAccess->rowStride();
                const int64_t dsStride = dstAccess->sampleStride(), dpStride = dstAccess->planeStride(),
                              drStride = dstAccess->rowStride();
                const bool aligned
                    = planes <= 65535 && reinterpret_cast<uintptr_t>(srcData.basePtr()) % sizeof(Vec4) == 0
                   && reinterpret_cast<uintptr_t>(dstData.basePtr()) % sizeof(Vec4) == 0 && sStride % sizeof(Vec4) == 0
                   && pStride % sizeof(Vec4) == 0 && rStride % sizeof(Vec4) == 0 && dsStride % sizeof(Vec4) == 0
                   && dpStride % sizeof(Vec4) == 0 && drStride % sizeof(Vec4) == 0;
                if (aligned)
                {
                    auto srcV = cuda::Tensor4DWrap<const BT, StrideType>(srcData.basePtr(), static_cast<int>(sStride),
                                                                         static_cast<int>(pStride),
                                                                         static_cast<int>(rStride));
                    auto dstV
                        = cuda::Tensor4DWrap<BT, StrideType>(dstData.basePtr(), static_cast<int>(dsStride),
                                                             static_cast<int>(dpStride), static_cast<int>(drStride));
                    dim3 vgrid(util::DivUp(util::DivUp(size.x, 4), static_cast<int>(block.x) * NGROUP),
                               util::DivUp(size.y, block.y), static_cast<unsigned int>(planes));
                    PosterizePlanarVec4Kernel<NGROUP, BT>
                        <<<vgrid, block, 0, stream>>>(srcV, dstV, int4{numSamples, numPlanes, size.y, size.x}, mask);
                    launchedVec = true;
                }
            }
            if (!launchedVec)
            {
                auto src = cuda::Tensor4DWrap<const ValueT, StrideType>(
                    srcData.basePtr(), static_cast<int>(srcAccess->sampleStride()),
                    static_cast<int>(srcAccess->planeStride()), static_cast<int>(srcAccess->rowStride()));
                auto dst = cuda::Tensor4DWrap<ValueT, StrideType>(
                    dstData.basePtr(), static_cast<int>(dstAccess->sampleStride()),
                    static_cast<int>(dstAccess->planeStride()), static_cast<int>(dstAccess->rowStride()));
                Posterize<isPlanar><<<grid, block, 0, stream>>>(src, dst, size, numPlanes, mask);
            }
        }
        NVCV_CHECK_THROW(cudaGetLastError());
    }
    else
    {
        static_assert(std::is_same_v<SrcData, nvcv::ImageBatchVarShapeDataStridedCuda>);
        if (dstData.numImages() > 65535)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Batch size exceeds the CUDA grid.z limit of 65535");
        }
        int3 dstMaxSize{dstData.maxSize().w, dstData.maxSize().h, dstData.numImages()};
        dim3 grid(util::DivUp(dstMaxSize.x, block.x), util::DivUp(dstMaxSize.y, block.y), dstMaxSize.z);

        const int numPlanes = dstData.uniqueFormat().numPlanes();

        // Vectorized planar var-shape path for 1-byte planes (uchar4; NVCV row pitch is >= 4-byte
        // aligned). u16 var-shape stays scalar (ushort4 8-byte alignment not guaranteed on var-shape).
        if constexpr (isPlanar && sizeof(BT) == 1)
        {
            constexpr int NGROUP = 4;
            const int64_t planes = static_cast<int64_t>(dstData.numImages()) * numPlanes;
            if (planes <= 65535)
            {
                cuda::ImageBatchVarShapeWrap<const BT> srcV(srcData);
                cuda::ImageBatchVarShapeWrap<BT>       dstV(dstData);
                dim3 vgrid(util::DivUp(util::DivUp(dstMaxSize.x, 4), static_cast<int>(block.x) * NGROUP),
                           util::DivUp(dstMaxSize.y, block.y), static_cast<unsigned int>(planes));
                PosterizePlanarVarShapeVec4Kernel<NGROUP, BT><<<vgrid, block, 0, stream>>>(srcV, dstV, numPlanes, mask);
                NVCV_CHECK_THROW(cudaGetLastError());
                return;
            }
        }

        cuda::ImageBatchVarShapeWrap<const ValueT> src(srcData);
        cuda::ImageBatchVarShapeWrap<ValueT>       dst(dstData);
        Posterize<isPlanar><<<grid, block, 0, stream>>>(src, dst, numPlanes, mask);
        NVCV_CHECK_THROW(cudaGetLastError());
    }
}

// Dispatch over base data type (u8 / u16 only — integer) and channel count (1 / 3 / 4) ---

template<typename Cb>
inline void RunTypeSwitch(nvcv::DataType dType, const Cb &cb)
{
    using uchar  = unsigned char;
    using ushort = unsigned short;

#define NVCV_POSTERIZE_RUN_TYPED(DYN_BASE_TYPE, STATIC_BASE_TYPE)                     \
    ((dType == nvcv::TYPE_4##DYN_BASE_TYPE) || (dType == nvcv::TYPE_3##DYN_BASE_TYPE) \
     || (dType == nvcv::TYPE_2##DYN_BASE_TYPE) || (dType == nvcv::TYPE_##DYN_BASE_TYPE)) cb(STATIC_BASE_TYPE{});

    // clang-format off
    if NVCV_POSTERIZE_RUN_TYPED(U8, uchar)
    else if NVCV_POSTERIZE_RUN_TYPED(U16, ushort)
    else
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Invalid data type: Posterize supports 8-bit and 16-bit unsigned integers only");
    }
        // clang-format on

#undef NVCV_POSTERIZE_RUN_TYPED
}

template<typename Cb>
inline void RunChannelSwitch(int numChannels, int numPlanes, nvcv::DataType dType, const Cb &cb)
{
    RunTypeSwitch(dType,
                  [&numChannels, &numPlanes, &cb](auto dummyVal)
                  {
                      using ValBase = decltype(dummyVal);
                      // clang-format off
            if (numChannels == 1)
            {
                using Val = cuda::MakeType<ValBase, 1>;
                if (numPlanes == 1)
                {
                    cb(Val{}, std::integral_constant<bool, false>{});
                }
                else
                {
                    cb(Val{}, std::integral_constant<bool, true>{});
                }
            }
            else if (numChannels == 3)
            {
                cb(cuda::MakeType<ValBase, 3>{}, std::integral_constant<bool, false>{});
            }
            else if (numChannels == 4)
            {
                cb(cuda::MakeType<ValBase, 4>{}, std::integral_constant<bool, false>{});
            }
            else
            {
                throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                      "Invalid number of channels: Posterize supports 1, 3 or 4 channels");
            }
                      // clang-format on
                  });
}

// Validation ------------------------------------------------------------------------------

inline void ValidateSrcDstTensors(int &numInterleavedChannels, int &numPlanes, nvcv::DataType &dtype,
                                  const nvcv::Optional<nvcv::TensorDataStridedCuda> &srcData,
                                  const nvcv::Optional<nvcv::TensorDataStridedCuda> &dstData)
{
    if (!srcData)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input must be cuda-accessible, pitch-linear tensor");
    }
    if (!dstData)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Output must be cuda-accessible, pitch-linear tensor");
    }
    if (srcData->layout() != dstData->layout())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input and output must have the same layout");
    }
    if (!(srcData->layout() == nvcv::TENSOR_HWC || srcData->layout() == nvcv::TENSOR_NHWC
          || srcData->layout() == nvcv::TENSOR_CHW || srcData->layout() == nvcv::TENSOR_NCHW))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input must have (N)HWC or (N)CHW layout");
    }
    if (srcData->dtype() != dstData->dtype())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input and output must have the same data type");
    }

    auto srcAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*srcData);
    auto dstAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*dstData);
    NVCV_ASSERT(srcAccess && dstAccess);

    if (srcAccess->numSamples() != dstAccess->numSamples())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Incompatible input/output number of samples");
    }

    int numChannels = srcAccess->numChannels();
    if (numChannels != dstAccess->numChannels())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Incompatible input/output number of channels");
    }

    numPlanes = srcAccess->numPlanes();
    if (numPlanes != dstAccess->numPlanes())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Incompatible input/output number of planes");
    }
    if (numPlanes > 1 && numChannels == 2)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "2-channel planar images are not supported");
    }

    if (srcAccess->numCols() != dstAccess->numCols() || srcAccess->numRows() != dstAccess->numRows())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input and output must have matching width and height");
    }

    dtype                  = srcData->dtype();
    numInterleavedChannels = srcAccess->infoLayout().isChannelLast() ? numChannels : 1;
}

inline auto ValidateSrcDstVarBatch(int &numInterleavedChannels, int &numPlanes, nvcv::DataType &dtype,
                                   cudaStream_t stream, const nvcv::ImageBatchVarShape &src,
                                   const nvcv::ImageBatchVarShape &dst)
{
    using maybeVarShape = nvcv::Optional<nvcv::ImageBatchVarShapeDataStridedCuda>;
    std::tuple<maybeVarShape, maybeVarShape> srcDstData{
        src.exportData<nvcv::ImageBatchVarShapeDataStridedCuda>(stream),
        dst.exportData<nvcv::ImageBatchVarShapeDataStridedCuda>(stream)};
    auto &[srcData, dstData] = srcDstData;

    if (!srcData)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input must be cuda-accessible, varshape pitch-linear image batch");
    }
    if (!dstData)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Output must be cuda-accessible, varshape pitch-linear image batch");
    }

    int numSamples = srcData->numImages();
    if (numSamples != dstData->numImages())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Incompatible input/output number of samples");
    }

    const auto &srcFormat = srcData->uniqueFormat();
    const auto &dstFormat = dstData->uniqueFormat();
    if (!srcFormat || !dstFormat)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "All images in a batch must have the same format");
    }
    if (srcFormat != dstFormat)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input and output must have the same format");
    }

    int numChannels = srcFormat.numChannels();
    numPlanes       = srcFormat.numPlanes();
    if (numPlanes > 1 && numChannels == 2)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "2-channel planar images are not supported");
    }

    dtype = srcFormat.planeDataType(0);
    for (int i = 1; i < numPlanes; ++i)
    {
        if (dtype != srcFormat.planeDataType(i))
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "All planes in the input image must have the same data type");
        }
    }

    numInterleavedChannels = dtype.numChannels();

    for (int i = 0; i < numSamples; i++)
    {
        if (src[i].size() != dst[i].size())
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Input and output must have matching width and height");
        }
    }

    return srcDstData;
}

} // anonymous namespace

namespace cvcuda::priv {

Posterize::Posterize() {}

// Tensor input variant
void Posterize::operator()(cudaStream_t stream, const nvcv::Tensor &src, const nvcv::Tensor &dst, int32_t bits) const
{
    CVCUDA_NVTX_RANGE("cvcuda::Posterize::operator()[Tensor]");
    int            numInterleavedChannels;
    int            numPlanes;
    nvcv::DataType dtype;
    auto           srcData = src.exportData<nvcv::TensorDataStridedCuda>();
    auto           dstData = dst.exportData<nvcv::TensorDataStridedCuda>();
    ValidateSrcDstTensors(numInterleavedChannels, numPlanes, dtype, srcData, dstData);

    RunChannelSwitch(numInterleavedChannels, numPlanes, dtype,
                     [&stream, &srcData, &dstData, bits](auto dummyVal, auto isPlanar)
                     {
                         using ValueT   = decltype(dummyVal);
                         using IsPlanar = decltype(isPlanar);
                         RunPosterize<IsPlanar::value, ValueT>(stream, *srcData, *dstData, bits);
                     });
}

// VarShape input variant
void Posterize::operator()(cudaStream_t stream, const nvcv::ImageBatchVarShape &src,
                           const nvcv::ImageBatchVarShape &dst, int32_t bits) const
{
    CVCUDA_NVTX_RANGE("cvcuda::Posterize::operator()[ImageBatchVarShape]");
    int            numInterleavedChannels;
    int            numPlanes;
    nvcv::DataType dtype;
    auto           srcDstData = ValidateSrcDstVarBatch(numInterleavedChannels, numPlanes, dtype, stream, src, dst);

    RunChannelSwitch(numInterleavedChannels, numPlanes, dtype,
                     [&stream, &srcDstData, bits](auto dummyVal, auto isPlanar)
                     {
                         using ValueT             = decltype(dummyVal);
                         using IsPlanar           = decltype(isPlanar);
                         auto &[srcData, dstData] = srcDstData;
                         RunPosterize<IsPlanar::value, ValueT>(stream, *srcData, *dstData, bits);
                     });
}

} // namespace cvcuda::priv
