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

#include "Nvtx.hpp"
#include "OpColorTwist.hpp"

#include <cvcuda/cuda_tools/Compat.hpp>
#include <cvcuda/cuda_tools/DropCast.hpp>
#include <cvcuda/cuda_tools/ImageBatchVarShapeWrap.hpp>
#include <cvcuda/cuda_tools/MathOps.hpp>
#include <cvcuda/cuda_tools/SaturateCast.hpp>
#include <cvcuda/cuda_tools/StaticCast.hpp>
#include <cvcuda/cuda_tools/TensorWrap.hpp>
#include <cvcuda/cuda_tools/math/LinAlg.hpp>
#include <nvcv/DataType.hpp>
#include <nvcv/Exception.hpp>
#include <nvcv/TensorData.hpp>
#include <nvcv/TensorDataAccess.hpp>
#include <nvcv/TensorLayout.hpp>
#include <nvcv/util/Assert.h>
#include <nvcv/util/CheckError.hpp>
#include <nvcv/util/Math.hpp>

#include <cstdint>
#include <type_traits>

namespace cuda = nvcv::cuda;
namespace util = nvcv::util;

namespace {

template<typename T, int N>
using Vec = cuda::math::Vector<T, N>;
template<typename T, int N, int M>
using Mat = cuda::math::Matrix<T, N, M>;

static bool IsPlanar(nvcv::TensorLayout layout)
{
    return layout == nvcv::TENSOR_NCHW || layout == nvcv::TENSOR_CHW;
}

constexpr int       kPlanarU8NIX  = 4;
constexpr uintptr_t kPlanarU8Mask = sizeof(uchar4) - 1;

inline __device__ bool IsAlignedForPlanarU8x4(const void *ptr)
{
    return (reinterpret_cast<uintptr_t>(ptr) & kPlanarU8Mask) == 0;
}

// Load explicit affine transform matrix from a tensor
template<class TwistWrap>
inline auto __device__ GetAffineTransform(const TwistWrap &twist)
{
    using ValueType = std::remove_const_t<typename TwistWrap::ValueType>;
    using BT        = cuda::BaseType<ValueType>;
    static_assert(cuda::NumElements<ValueType> == 4);

    Mat<BT, 3, 4> affineTransform;
#pragma unroll
    for (int i = 0; i < 3; i++)
    {
        ValueType row;
        if constexpr (TwistWrap::kNumDimensions == 1)
        {
            row = twist[i];
        }
        else
        {
            static_assert(TwistWrap::kNumDimensions == 2);
            int  z = blockIdx.z;
            int2 coord{i, z};
            row = twist[coord];
        }
#pragma unroll
        for (int j = 0; j < 4; j++)
        {
            affineTransform[i][j] = cuda::GetElement(row, j);
        }
    }
    return affineTransform;
}

// Do actual transformation of a pixel by an affine transform
template<class SrcWrapper, class DstWrapper, int N, typename TwistT>
inline void __device__ DoAffineTransform(SrcWrapper src, DstWrapper dst, const int2 size,
                                         const Mat<TwistT, N, N + 1> transform)
{
    using SrcT                       = typename SrcWrapper::ValueType;
    using DstT                       = typename DstWrapper::ValueType;
    using T                          = cuda::BaseType<DstT>;
    static constexpr int numChannels = cuda::NumElements<SrcT>;
    static_assert(std::is_same_v<T, std::remove_const_t<cuda::BaseType<SrcT>>>);
    static_assert(numChannels == cuda::NumElements<DstT>);
    static_assert(numChannels >= N);

    int3 coord = cuda::StaticCast<int>(blockIdx * blockDim + threadIdx);
    if (coord.x >= size.x || coord.y >= size.y)
    {
        return;
    }

    auto               src_pixel = src[coord];
    Vec<TwistT, N + 1> in_vec;
#pragma unroll
    for (int i = 0; i < N; i++)
    {
        in_vec[i] = cuda::GetElement(src_pixel, i);
    }
    in_vec[N]              = 1;
    Vec<TwistT, 3> out_vec = transform * in_vec;
    DstT           out_pixel;
#pragma unroll
    for (int i = 0; i < N; i++)
    {
        cuda::GetElement(out_pixel, i) = cuda::SaturateCast<T>(out_vec[i]);
    }
    // rewrite the extra channels unaffected
#pragma unroll
    for (int i = N; i < numChannels; i++)
    {
        cuda::GetElement(out_pixel, i) = cuda::GetElement(src_pixel, i);
    }
    dst[coord] = out_pixel;
}

template<class SrcWrapper, class DstWrapper, int N, typename TwistT>
inline void __device__ DoAffineTransformPlanar(SrcWrapper src, DstWrapper dst, const int2 size, int numChannels,
                                               const Mat<TwistT, N, N + 1> transform)
{
    using SrcT = typename SrcWrapper::ValueType;
    using DstT = typename DstWrapper::ValueType;
    using T    = std::remove_const_t<DstT>;
    static_assert(std::is_same_v<T, std::remove_const_t<SrcT>>);
    static_assert(N == 3);

    int3 coord = cuda::StaticCast<int>(blockIdx * blockDim + threadIdx);
    if (coord.x >= size.x || coord.y >= size.y)
    {
        return;
    }

    Vec<TwistT, N + 1> in_vec;
#pragma unroll
    for (int i = 0; i < N; i++)
    {
        in_vec[i] = *src.ptr(coord.z, i, coord.y, coord.x);
    }
    in_vec[N]              = 1;
    Vec<TwistT, N> out_vec = transform * in_vec;

#pragma unroll
    for (int i = 0; i < N; i++)
    {
        *dst.ptr(coord.z, i, coord.y, coord.x) = cuda::SaturateCast<T>(out_vec[i]);
    }
    if (numChannels == 4)
    {
        *dst.ptr(coord.z, 3, coord.y, coord.x) = *src.ptr(coord.z, 3, coord.y, coord.x);
    }
}

template<typename TwistT>
inline void __device__ ApplyColorTwistPlanarU8(uint8_t r, uint8_t g, uint8_t b, const Mat<TwistT, 3, 4> transform,
                                               uint8_t &outR, uint8_t &outG, uint8_t &outB)
{
    Vec<TwistT, 4> inVec;
    inVec[0]              = r;
    inVec[1]              = g;
    inVec[2]              = b;
    inVec[3]              = 1;
    Vec<TwistT, 3> outVec = transform * inVec;
    outR                  = cuda::SaturateCast<uint8_t>(outVec[0]);
    outG                  = cuda::SaturateCast<uint8_t>(outVec[1]);
    outB                  = cuda::SaturateCast<uint8_t>(outVec[2]);
}

template<class SrcWrapper, class DstWrapper, typename TwistT>
inline void __device__ DoAffineTransformPlanarU8x4(SrcWrapper src, DstWrapper dst, const int2 size, int numChannels,
                                                   const Mat<TwistT, 3, 4> transform)
{
    using SrcT = std::remove_const_t<typename SrcWrapper::ValueType>;
    using DstT = typename DstWrapper::ValueType;
    static_assert(std::is_same_v<SrcT, uint8_t>);
    static_assert(std::is_same_v<DstT, uint8_t>);

    int3 coord = cuda::StaticCast<int>(blockIdx * blockDim + threadIdx);
    coord.x *= kPlanarU8NIX;
    if (coord.x >= size.x || coord.y >= size.y)
    {
        return;
    }

    const uint8_t *srcR = src.ptr(coord.z, 0, coord.y, coord.x);
    const uint8_t *srcG = src.ptr(coord.z, 1, coord.y, coord.x);
    const uint8_t *srcB = src.ptr(coord.z, 2, coord.y, coord.x);
    uint8_t       *dstR = dst.ptr(coord.z, 0, coord.y, coord.x);
    uint8_t       *dstG = dst.ptr(coord.z, 1, coord.y, coord.x);
    uint8_t       *dstB = dst.ptr(coord.z, 2, coord.y, coord.x);

    bool canVectorize = coord.x + kPlanarU8NIX - 1 < size.x && IsAlignedForPlanarU8x4(srcR)
                     && IsAlignedForPlanarU8x4(srcG) && IsAlignedForPlanarU8x4(srcB) && IsAlignedForPlanarU8x4(dstR)
                     && IsAlignedForPlanarU8x4(dstG) && IsAlignedForPlanarU8x4(dstB);

    if (numChannels == 4)
    {
        canVectorize = canVectorize && IsAlignedForPlanarU8x4(src.ptr(coord.z, 3, coord.y, coord.x))
                    && IsAlignedForPlanarU8x4(dst.ptr(coord.z, 3, coord.y, coord.x));
    }

    if (canVectorize)
    {
        const uchar4 rIn = *reinterpret_cast<const uchar4 *>(srcR);
        const uchar4 gIn = *reinterpret_cast<const uchar4 *>(srcG);
        const uchar4 bIn = *reinterpret_cast<const uchar4 *>(srcB);
        uchar4       rOut{};
        uchar4       gOut{};
        uchar4       bOut{};
#pragma unroll
        for (int i = 0; i < kPlanarU8NIX; i++)
        {
            uint8_t r;
            uint8_t g;
            uint8_t b;
            ApplyColorTwistPlanarU8(cuda::GetElement(rIn, i), cuda::GetElement(gIn, i), cuda::GetElement(bIn, i),
                                    transform, r, g, b);
            cuda::GetElement(rOut, i) = r;
            cuda::GetElement(gOut, i) = g;
            cuda::GetElement(bOut, i) = b;
        }
        *reinterpret_cast<uchar4 *>(dstR) = rOut;
        *reinterpret_cast<uchar4 *>(dstG) = gOut;
        *reinterpret_cast<uchar4 *>(dstB) = bOut;
        if (numChannels == 4)
        {
            *reinterpret_cast<uchar4 *>(dst.ptr(coord.z, 3, coord.y, coord.x))
                = *reinterpret_cast<const uchar4 *>(src.ptr(coord.z, 3, coord.y, coord.x));
        }
    }
    else
    {
#pragma unroll
        for (int i = 0; i < kPlanarU8NIX; i++)
        {
            const int x = coord.x + i;
            if (x < size.x)
            {
                uint8_t r;
                uint8_t g;
                uint8_t b;
                ApplyColorTwistPlanarU8(*src.ptr(coord.z, 0, coord.y, x), *src.ptr(coord.z, 1, coord.y, x),
                                        *src.ptr(coord.z, 2, coord.y, x), transform, r, g, b);
                *dst.ptr(coord.z, 0, coord.y, x) = r;
                *dst.ptr(coord.z, 1, coord.y, x) = g;
                *dst.ptr(coord.z, 2, coord.y, x) = b;
                if (numChannels == 4)
                {
                    *dst.ptr(coord.z, 3, coord.y, x) = *src.ptr(coord.z, 3, coord.y, x);
                }
            }
        }
    }
}

// Load affine transform ----------------------------------------------------------

template<class SrcWrapper, class DstWrapper, typename ValueType>
inline __device__ void DoColorTwist(SrcWrapper src, DstWrapper dst, const int2 size,
                                    const cuda::Tensor1DWrap<const ValueType> param)
{
    static_assert(cuda::NumElements<ValueType> == 4);
    auto transform = GetAffineTransform(param);
    DoAffineTransform(src, dst, size, transform);
}

template<class SrcWrapper, class DstWrapper, typename ValueType>
inline __device__ void DoColorTwist(SrcWrapper src, DstWrapper dst, const int2 size,
                                    const cuda::Tensor2DWrap<const ValueType> param)
{
    static_assert(cuda::NumElements<ValueType> == 4);
    auto transform = GetAffineTransform(param);
    DoAffineTransform(src, dst, size, transform);
}

template<class SrcWrapper, class DstWrapper, typename ValueType>
inline __device__ void DoColorTwistPlanar(SrcWrapper src, DstWrapper dst, const int2 size, int numChannels,
                                          const cuda::Tensor1DWrap<const ValueType> param)
{
    static_assert(cuda::NumElements<ValueType> == 4);
    auto transform = GetAffineTransform(param);
    DoAffineTransformPlanar(src, dst, size, numChannels, transform);
}

template<class SrcWrapper, class DstWrapper, typename ValueType>
inline __device__ void DoColorTwistPlanar(SrcWrapper src, DstWrapper dst, const int2 size, int numChannels,
                                          const cuda::Tensor2DWrap<const ValueType> param)
{
    static_assert(cuda::NumElements<ValueType> == 4);
    auto transform = GetAffineTransform(param);
    DoAffineTransformPlanar(src, dst, size, numChannels, transform);
}

template<class SrcWrapper, class DstWrapper, class ColorTwistParam>
inline __device__ void DoColorTwistPlanarU8x4(SrcWrapper src, DstWrapper dst, const int2 size, int numChannels,
                                              const ColorTwistParam param)
{
    using ValueType = std::remove_const_t<typename ColorTwistParam::ValueType>;
    static_assert(cuda::NumElements<ValueType> == 4);
    auto transform = GetAffineTransform(param);
    DoAffineTransformPlanarU8x4(src, dst, size, numChannels, transform);
}

// ColorTwist kernel --------------------------------------------------------------

// Tensor variant
template<class SrcWrapper, class DstWrapper, class ColorTwistParam>
__global__ void ColorTwist(SrcWrapper src, DstWrapper dst, int2 size, const ColorTwistParam param)
{
    DoColorTwist(src, dst, size, param);
}

template<class SrcWrapper, class DstWrapper, class ColorTwistParam>
__global__ void ColorTwistPlanarTensor(SrcWrapper src, DstWrapper dst, int2 size, int numChannels,
                                       const ColorTwistParam param)
{
    DoColorTwistPlanar(src, dst, size, numChannels, param);
}

template<class SrcWrapper, class DstWrapper, class ColorTwistParam>
__global__ void ColorTwistPlanarTensorU8x4(SrcWrapper src, DstWrapper dst, int2 size, int numChannels,
                                           const ColorTwistParam param)
{
    DoColorTwistPlanarU8x4(src, dst, size, numChannels, param);
}

// VarBatch variant
template<class SrcWrapper, class DstWrapper, class ColorTwistParam>
__global__ void ColorTwist(SrcWrapper src, DstWrapper dst, const ColorTwistParam param)
{
    int  z = blockIdx.z;
    int2 size{dst.width(z), dst.height(z)};

    DoColorTwist(src, dst, size, param);
}

template<class SrcWrapper, class DstWrapper, class ColorTwistParam>
__global__ void ColorTwistPlanarVarShape(SrcWrapper src, DstWrapper dst, int numChannels, const ColorTwistParam param)
{
    int  z = blockIdx.z;
    int2 size{dst.width(z), dst.height(z)};

    DoColorTwistPlanar(src, dst, size, numChannels, param);
}

template<class SrcWrapper, class DstWrapper, class ColorTwistParam>
__global__ void ColorTwistPlanarVarShapeU8x4(SrcWrapper src, DstWrapper dst, int numChannels,
                                             const ColorTwistParam param)
{
    int  z = blockIdx.z;
    int2 size{dst.width(z), dst.height(z)};

    DoColorTwistPlanarU8x4(src, dst, size, numChannels, param);
}

// Run ColorTwist kernel ----------------------------------------------------------

template<typename T, class SrcData, class DstData, class ColorTwistParam>
inline void RunColorTwist(cudaStream_t stream, const SrcData &srcData, const DstData &dstData,
                          const ColorTwistParam &param)
{
    dim3 block(32, 4, 1);
    if constexpr (std::is_same_v<SrcData, nvcv::TensorDataStridedCuda>)
    {
        auto inAccess = nvcv::TensorDataAccessStridedImage::Create(srcData);
        NVCV_ASSERT(inAccess);
        auto outAccess = nvcv::TensorDataAccessStridedImage::Create(dstData);
        NVCV_ASSERT(outAccess);
        int2 size = cuda::StaticCast<int>(long2{inAccess->numCols(), inAccess->numRows()});
        dim3 grid(util::DivUp(size.x, block.x), util::DivUp(size.y, block.y), inAccess->numSamples());

        int64_t inMaxStride  = inAccess->sampleStride() * inAccess->numSamples();
        int64_t outMaxStride = outAccess->sampleStride() * outAccess->numSamples();
        if (std::max(inMaxStride, outMaxStride) <= cuda::TypeTraits<int32_t>::max)
        {
            auto src = cuda::CreateTensorWrapNHW<const T, int32_t>(srcData);
            auto dst = cuda::CreateTensorWrapNHW<T, int32_t>(dstData);
            ColorTwist<<<grid, block, 0, stream>>>(src, dst, size, param);
        }
        else
        {
            throw nvcv::Exception(nvcv::Status::ERROR_OVERFLOW, "Input or output size exceeds %d. Tensor is too large.",
                                  cuda::TypeTraits<int32_t>::max);
        }
        NVCV_CHECK_THROW(cudaGetLastError());
    }
    else
    {
        static_assert(std::is_same_v<SrcData, nvcv::ImageBatchVarShapeDataStridedCuda>);
        int3 dstMaxSize{dstData.maxSize().w, dstData.maxSize().h, dstData.numImages()};
        dim3 grid(util::DivUp(dstMaxSize.x, block.x), util::DivUp(dstMaxSize.y, block.y), dstMaxSize.z);

        cuda::ImageBatchVarShapeWrap<const T> src(srcData);
        cuda::ImageBatchVarShapeWrap<T>       dst(dstData);

        ColorTwist<<<grid, block, 0, stream>>>(src, dst, param);
        NVCV_CHECK_THROW(cudaGetLastError());
    }
}

template<typename T, class ColorTwistParam>
inline void RunColorTwistPlanar(cudaStream_t stream, const nvcv::TensorDataStridedCuda &srcData,
                                const nvcv::TensorDataStridedCuda &dstData, const ColorTwistParam &param)
{
    using BT = cuda::BaseType<T>;

    dim3 block(32, 4, 1);

    auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(srcData);
    NVCV_ASSERT(inAccess);
    auto outAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(dstData);
    NVCV_ASSERT(outAccess);
    int2 size = cuda::StaticCast<int>(long2{inAccess->numCols(), inAccess->numRows()});

    if (inAccess->numSamples() > 65535)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Planar ColorTwist launch exceeds CUDA grid.z limit: N=%d",
                              static_cast<int>(inAccess->numSamples()));
    }

    dim3 grid(util::DivUp(size.x, block.x), util::DivUp(size.y, block.y), inAccess->numSamples());

    int64_t inMaxStride  = inAccess->sampleStride() * inAccess->numSamples();
    int64_t outMaxStride = outAccess->sampleStride() * outAccess->numSamples();
    if (std::max(inMaxStride, outMaxStride) <= cuda::TypeTraits<int32_t>::max)
    {
        auto src = cuda::CreateTensorWrapNCHW<const BT, int32_t>(srcData);
        auto dst = cuda::CreateTensorWrapNCHW<BT, int32_t>(dstData);
        if constexpr (std::is_same_v<BT, uint8_t>)
        {
            dim3 gridU8(util::DivUp(size.x, block.x * kPlanarU8NIX), util::DivUp(size.y, block.y),
                        inAccess->numSamples());
            ColorTwistPlanarTensorU8x4<<<gridU8, block, 0, stream>>>(src, dst, size, inAccess->numChannels(), param);
        }
        else
        {
            ColorTwistPlanarTensor<<<grid, block, 0, stream>>>(src, dst, size, inAccess->numChannels(), param);
        }
    }
    else
    {
        throw nvcv::Exception(nvcv::Status::ERROR_OVERFLOW, "Input or output size exceeds %d. Tensor is too large.",
                              cuda::TypeTraits<int32_t>::max);
    }
    NVCV_CHECK_THROW(cudaGetLastError());
}

template<typename T, class ColorTwistParam>
inline void RunColorTwistPlanar(cudaStream_t stream, const nvcv::ImageBatchVarShapeDataStridedCuda &srcData,
                                const nvcv::ImageBatchVarShapeDataStridedCuda &dstData, int numChannels,
                                const ColorTwistParam &param)
{
    using BT = cuda::BaseType<T>;

    if (dstData.numImages() > 65535)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Planar ColorTwist launch exceeds CUDA grid.z limit: N=%d", dstData.numImages());
    }

    dim3 block(32, 4, 1);
    int3 dstMaxSize{dstData.maxSize().w, dstData.maxSize().h, dstData.numImages()};
    dim3 grid(util::DivUp(dstMaxSize.x, block.x), util::DivUp(dstMaxSize.y, block.y), dstMaxSize.z);

    cuda::ImageBatchVarShapeWrap<const BT> src(srcData);
    cuda::ImageBatchVarShapeWrap<BT>       dst(dstData);

    if constexpr (std::is_same_v<BT, uint8_t>)
    {
        dim3 gridU8(util::DivUp(dstMaxSize.x, block.x * kPlanarU8NIX), util::DivUp(dstMaxSize.y, block.y),
                    dstMaxSize.z);
        ColorTwistPlanarVarShapeU8x4<<<gridU8, block, 0, stream>>>(src, dst, numChannels, param);
    }
    else
    {
        ColorTwistPlanarVarShape<<<grid, block, 0, stream>>>(src, dst, numChannels, param);
    }
    NVCV_CHECK_THROW(cudaGetLastError());
}

template<typename SrcDestT, typename TwistT, class SrcData, class DstData>
inline void RunColorTwist(cudaStream_t stream, const SrcData &srcData, const DstData &dstData,
                          const nvcv::TensorDataStridedCuda &twistData, bool hasPerSampleTwist, bool isPlanar,
                          int numChannels)
{
    if (!hasPerSampleTwist)
    {
        auto twist = cuda::Tensor1DWrap<const TwistT>(twistData);
        if (isPlanar)
        {
            if constexpr (std::is_same_v<SrcData, nvcv::TensorDataStridedCuda>)
            {
                RunColorTwistPlanar<SrcDestT>(stream, srcData, dstData, twist);
            }
            else
            {
                RunColorTwistPlanar<SrcDestT>(stream, srcData, dstData, numChannels, twist);
            }
        }
        else
        {
            RunColorTwist<SrcDestT>(stream, srcData, dstData, twist);
        }
    }
    else
    {
        auto twist = cuda::Tensor2DWrap<const TwistT>(twistData);
        if (isPlanar)
        {
            if constexpr (std::is_same_v<SrcData, nvcv::TensorDataStridedCuda>)
            {
                RunColorTwistPlanar<SrcDestT>(stream, srcData, dstData, twist);
            }
            else
            {
                RunColorTwistPlanar<SrcDestT>(stream, srcData, dstData, numChannels, twist);
            }
        }
        else
        {
            RunColorTwist<SrcDestT>(stream, srcData, dstData, twist);
        }
    }
}

// Src/twist/dst type-switch
template<typename Cb>
inline void RunSrcTypeSwitch(int numChannels, nvcv::DataType srcType, nvcv::DataType twistType, Cb &&cb)
{
    // The channels of input sample and the width of transform tensor may be baked into the data type.

#define NVCV_RUN_COLOR_TWIST(NUM_CHANNELS, SRC_TYPE, TWIST_TYPE, SRC_VEC_TYPE, TWIST_VEC_TYPE) \
    ((numChannels == NUM_CHANNELS)                                                             \
     && (srcType == nvcv::TYPE_##SRC_TYPE || srcType == nvcv::TYPE_##NUM_CHANNELS##SRC_TYPE)   \
     && (twistType == nvcv::TYPE_##TWIST_TYPE || twistType == nvcv::TYPE_4##TWIST_TYPE))       \
        cb(SRC_VEC_TYPE{}, TWIST_VEC_TYPE{})

    // clang-format off
    if NVCV_RUN_COLOR_TWIST (3, U8, F32, uchar3, float4);
    else if NVCV_RUN_COLOR_TWIST (4, U8, F32, uchar4, float4);
    else if NVCV_RUN_COLOR_TWIST (3, U16, F32, ushort3, float4);
    else if NVCV_RUN_COLOR_TWIST (4, U16, F32, ushort4, float4);
    else if NVCV_RUN_COLOR_TWIST (3, S16, F32, short3, float4);
    else if NVCV_RUN_COLOR_TWIST (4, S16, F32, short4, float4);
    else if NVCV_RUN_COLOR_TWIST (3, U32, F64, uint3, double4_16a);
    else if NVCV_RUN_COLOR_TWIST (4, U32, F64, uint4, double4_16a);
    else if NVCV_RUN_COLOR_TWIST (3, S32, F64, int3, double4_16a);
    else if NVCV_RUN_COLOR_TWIST (4, S32, F64, int4, double4_16a);
    else if NVCV_RUN_COLOR_TWIST (3, F32, F32, float3, float4);
    else if NVCV_RUN_COLOR_TWIST (4, F32, F32, float4, float4);
    else
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid input/twist/output data types");
    }
    // clang-format on

#undef NVCV_RUN_COLOR_TWIST
}

// Argument validation helpers ----------------------------------------------------

inline bool validateSrcDstTensors(int &numSamples, int &numChannels, nvcv::DataType &srcDstDtype,
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

    srcDstDtype = srcData->dtype();

    if (srcDstDtype != dstData->dtype())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input and output data type are different, but must be the same.");
    }

    if (srcData->layout() != dstData->layout())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input and output must have the same layout");
    }

    const bool isPlanar = IsPlanar(srcData->layout());
    if (srcData->layout() != nvcv::TENSOR_HWC && srcData->layout() != nvcv::TENSOR_NHWC && !isPlanar)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input must have (N)HWC or (N)CHW layout");
    }

    auto srcAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*srcData);
    auto dstAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*dstData);
    if (!srcAccess || !dstAccess)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input/output must be accessible as strided images");
    }

    numSamples = srcAccess->numSamples();
    if (numSamples != dstAccess->numSamples())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Incompatible input/output number of samples");
    }

    numChannels = srcAccess->numChannels();
    if (numChannels != dstAccess->numChannels())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Incompatible input/output number of channels");
    }

    if (numChannels != 3 && numChannels != 4)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input must have 3 or 4 channels");
    }

    return isPlanar;
}

inline bool validateSrcDstVarBatch(int &numSamples, int &numChannels, nvcv::DataType &srcDstDtype,
                                   const nvcv::Optional<nvcv::ImageBatchVarShapeDataStridedCuda> &srcData,
                                   const nvcv::Optional<nvcv::ImageBatchVarShapeDataStridedCuda> &dstData)
{
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

    numSamples = srcData->numImages();
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

    srcDstDtype = srcFormat.planeDataType(0);

    numChannels = srcFormat.numChannels();
    if (numChannels != 3 && numChannels != 4)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "The input must have 3 or 4 channels");
    }

    if (srcFormat != dstFormat)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input and output data type are different, but must be the same.");
    }

    return srcFormat.numPlanes() > 1;
}

inline void validateTwistTensor(bool &hasPerSampleTwist, nvcv::DataType &twistDtype, int numImages,
                                const nvcv::Optional<nvcv::TensorDataStridedCuda> &twistData)
{
    if (!twistData)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "The twist argument must be cuda-accessible, pitch-linear tensor");
    }

    twistDtype = twistData->dtype();

    if (twistDtype.numChannels() != 1 && twistDtype.numChannels() != 4)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "The twist transformation must be a 3x4 matrix");
    }

    int  rank               = twistData->rank();
    bool hasBakedInChannels = twistDtype.numChannels() > 1;
    int  numDataDims        = rank + hasBakedInChannels;
    hasPerSampleTwist       = numDataDims == 3;

    if (!hasPerSampleTwist && numDataDims != 2)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "The twist argument must be 2D or 3D tensor");
    }

    int numCols, numRows;
    if (hasPerSampleTwist)
    {
        if (numImages != twistData->shape(0))
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "The twist must be 2D matrix or 3D tensor where the outermost dimenstion matches "
                                  "the input batch size");
        }
        numRows = twistData->shape(1);
        numCols = hasBakedInChannels ? 4 : twistData->shape(2);
    }
    else
    {
        numRows = twistData->shape(0);
        numCols = hasBakedInChannels ? 4 : twistData->shape(1);
    }

    if (numRows != 3 || numCols != 4)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "The twist must matrix must be 3x4");
    }
}

} // anonymous namespace

namespace cvcuda::priv {

// Constructor -----------------------------------------------------------------

ColorTwist::ColorTwist() {}

// Operator --------------------------------------------------------------------

// Tensor input variant
void ColorTwist::operator()(cudaStream_t stream, const nvcv::Tensor &src, const nvcv::Tensor &dst,
                            const nvcv::Tensor &twist) const
{
    CVCUDA_NVTX_RANGE("cvcuda::ColorTwist::operator()[Tensor]");
    int            numSamples;
    int            numChannels;
    nvcv::DataType srcDstDtype;
    auto           srcData  = src.exportData<nvcv::TensorDataStridedCuda>();
    auto           dstData  = dst.exportData<nvcv::TensorDataStridedCuda>();
    bool           isPlanar = validateSrcDstTensors(numSamples, numChannels, srcDstDtype, srcData, dstData);

    bool           hasPerSampleTwist;
    nvcv::DataType twistDtype;
    auto           twistData = twist.exportData<nvcv::TensorDataStridedCuda>();
    validateTwistTensor(hasPerSampleTwist, twistDtype, numSamples, twistData);

    RunSrcTypeSwitch(numChannels, srcDstDtype, twistDtype,
                     [&stream, &srcData, &dstData, &twistData, &hasPerSampleTwist, isPlanar, numChannels](
                         auto srcDummy, auto twistDummy)
                     {
                         using SrcDstT = decltype(srcDummy);
                         using TwistT  = decltype(twistDummy);
                         RunColorTwist<SrcDstT, TwistT>(stream, *srcData, *dstData, *twistData, hasPerSampleTwist,
                                                        isPlanar, numChannels);
                     });
}

// VarShape input variant
void ColorTwist::operator()(cudaStream_t stream, const nvcv::ImageBatchVarShape &src,
                            const nvcv::ImageBatchVarShape &dst, const nvcv::Tensor &twist) const
{
    CVCUDA_NVTX_RANGE("cvcuda::ColorTwist::operator()[ImageBatchVarShape]");
    int            numSamples;
    int            numChannels;
    nvcv::DataType srcDstDtype;
    auto           srcData  = src.exportData<nvcv::ImageBatchVarShapeDataStridedCuda>(stream);
    auto           dstData  = dst.exportData<nvcv::ImageBatchVarShapeDataStridedCuda>(stream);
    bool           isPlanar = validateSrcDstVarBatch(numSamples, numChannels, srcDstDtype, srcData, dstData);

    bool           hasPerSampleTwist;
    nvcv::DataType twistDtype;
    auto           twistData = twist.exportData<nvcv::TensorDataStridedCuda>();
    validateTwistTensor(hasPerSampleTwist, twistDtype, numSamples, twistData);

    RunSrcTypeSwitch(numChannels, srcDstDtype, twistDtype,
                     [&stream, &srcData, &dstData, &twistData, &hasPerSampleTwist, isPlanar, numChannels](
                         auto srcDummy, auto twistDummy)
                     {
                         using SrcDstT = decltype(srcDummy);
                         using TwistT  = decltype(twistDummy);
                         RunColorTwist<SrcDstT, TwistT>(stream, *srcData, *dstData, *twistData, hasPerSampleTwist,
                                                        isPlanar, numChannels);
                     });
}

} // namespace cvcuda::priv
