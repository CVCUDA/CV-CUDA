/* Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * SPDX-License-Identifier: Apache-2.0
 *
 * Copyright (C) 2021-2022, Bytedance Inc. All rights reserved.
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

#include "../CudaDeviceUtils.hpp"
#include "CopyMakeBorderPolicy.hpp"
#include "CvCudaLegacy.h"
#include "CvCudaLegacyHelpers.hpp"

#include "CvCudaUtils.cuh"

#include <type_traits>

#define BLOCK 32

using namespace nvcv;
using namespace nvcv::legacy::cuda_op;
using namespace nvcv::legacy::helpers;

namespace nvcv::legacy::cuda_op {
namespace {

static bool IsPlanar(DataFormat format)
{
    return format == kNCHW || format == kCHW;
}

static bool SameLayoutFamily(DataFormat lhs, DataFormat rhs)
{
    return IsPlanar(lhs) == IsPlanar(rhs);
}

template<class OutType>
DataType GetOutputDataType(const OutType &data_out)
{
    if constexpr (std::is_same_v<OutType, TensorDataStridedCuda>)
    {
        return GetLegacyDataType(data_out.dtype());
    }
    else
    {
        return GetLegacyDataType(data_out.uniqueFormat());
    }
}

static float PlanarBorderComponent(const float4 &borderValue, int plane)
{
    return plane == 0 ? borderValue.x : plane == 1 ? borderValue.y : plane == 2 ? borderValue.z : borderValue.w;
}

static TensorDataStridedCuda PlanarTensorView(const TensorDataStridedCuda &data)
{
    auto access = TensorDataAccessStridedImagePlanar::Create(data);
    NVCV_ASSERT(access);

    TensorDataStridedCuda::Buffer buf;
    buf.basePtr    = reinterpret_cast<NVCVByte *>(data.basePtr());
    buf.strides[0] = access->sampleStride();
    buf.strides[1] = access->chStride();
    buf.strides[2] = access->rowStride();
    buf.strides[3] = access->colStride();

    return TensorDataStridedCuda{
        TensorShape{{access->numSamples(), access->numChannels(), access->numRows(), access->numCols()}, "NCHW"},
        data.dtype(), buf
    };
}

template<typename T>
constexpr int CMBVS_NIX = sizeof(T) == 1 ? 8
                        : std::is_same_v<cuda::BaseType<T>, float> && sizeof(T) == 4
                            ? 2
                            : (cuda::NumElements<T> == 3 ? sizeof(uint3) : sizeof(uint4)) / sizeof(T);

template<typename T, int NIX>
using CMBVS_DPT
    = std::conditional_t<sizeof(T) * NIX == sizeof(uint), uint,
                         std::conditional_t<sizeof(T) * NIX == sizeof(uint2), uint2,
                                            std::conditional_t<sizeof(T) * NIX == sizeof(uint3), uint3, uint4>>>;

template<typename T, int NIX>
constexpr uintptr_t CMBVS_MSK = alignof(CMBVS_DPT<T, NIX>) - 1;

inline int CurrentDeviceSMOrZero()
{
    int sm = 0;
    return cvcuda::priv::GetCurrentDeviceSM(sm) == cudaSuccess ? sm : 0;
}

template<class SrcWrapper, class DstWrapper>
__global__ void copyMakeBorderKernel(const SrcWrapper src, DstWrapper dst, const cuda::Tensor3DWrap<int, int32_t> left_,
                                     const cuda::Tensor3DWrap<int, int32_t> top_, int out_height, int out_width)
{
    const int x         = blockIdx.x * blockDim.x + threadIdx.x;
    const int y         = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();

    int       left    = *left_.ptr(0, 0, batch_idx);
    int       top     = *top_.ptr(0, 0, batch_idx);
    const int x_shift = x - left;
    const int y_shift = y - top;

    if (x < out_width && y < out_height)
    {
        int3 srcCoord             = {x_shift, y_shift, batch_idx};
        *dst.ptr(batch_idx, y, x) = src[srcCoord];
    }
}

template<bool PLANAR, bool FIXED_OUTPUT, class SrcWrapper, class SrcRawWrapper, class DstWrapper, typename T,
         int NIX = CMBVS_NIX<T>>
__global__ void copyMakeBorderKernelVec(const SrcWrapper src, const SrcRawWrapper srcRaw, DstWrapper dst,
                                        const cuda::Tensor3DWrap<int, int32_t> left_,
                                        const cuda::Tensor3DWrap<int, int32_t> top_, int fixedHeight, int fixedWidth,
                                        int plane)
{
    const int y         = blockIdx.y * blockDim.y + threadIdx.y;
    const int x0        = (blockIdx.x * blockDim.x + threadIdx.x) * NIX;
    const int batch_idx = get_batch_idx();

    int outHeight = fixedHeight;
    int outWidth  = fixedWidth;
    if constexpr (!FIXED_OUTPUT)
    {
        if constexpr (PLANAR)
        {
            outHeight = dst.height(batch_idx, plane);
            outWidth  = dst.width(batch_idx, plane);
        }
        else
        {
            outHeight = dst.height(batch_idx);
            outWidth  = dst.width(batch_idx);
        }
    }

    if (x0 >= outWidth || y >= outHeight)
        return;

    const int left = *left_.ptr(0, 0, batch_idx);
    const int top  = *top_.ptr(0, 0, batch_idx);
    const int sx0  = x0 - left;
    const int sy   = y - top;

    int srcHeight;
    int srcWidth;
    if constexpr (PLANAR)
    {
        srcHeight = srcRaw.height(batch_idx, plane);
        srcWidth  = srcRaw.width(batch_idx, plane);
    }
    else
    {
        srcHeight = srcRaw.height(batch_idx);
        srcWidth  = srcRaw.width(batch_idx);
    }

    const bool full         = x0 + NIX - 1 < outWidth;
    const bool interiorRow  = sy >= 0 && sy < srcHeight;
    const bool interiorSpan = sx0 >= 0 && sx0 + NIX - 1 < srcWidth;

    if (full && interiorRow && interiorSpan)
    {
        const T *sp;
        T       *dp;
        if constexpr (PLANAR)
        {
            sp = srcRaw.ptr(batch_idx, plane, sy, sx0);
            dp = dst.ptr(batch_idx, plane, y, x0);
        }
        else
        {
            sp = srcRaw.ptr(batch_idx, sy, sx0);
            dp = dst.ptr(batch_idx, y, x0);
        }

        if ((reinterpret_cast<uintptr_t>(sp) & CMBVS_MSK<T, NIX>) == 0
            && (reinterpret_cast<uintptr_t>(dp) & CMBVS_MSK<T, NIX>) == 0)
        {
            *reinterpret_cast<CMBVS_DPT<T, NIX> *>(dp) = *reinterpret_cast<const CMBVS_DPT<T, NIX> *>(sp);
        }
        else
        {
#pragma unroll
            for (int i = 0; i < NIX; ++i) dp[i] = sp[i];
        }
    }
    else
    {
#pragma unroll
        for (int i = 0; i < NIX; ++i)
        {
            const int x = x0 + i;
            if (x < outWidth)
            {
                if constexpr (PLANAR)
                    *dst.ptr(batch_idx, plane, y, x) = src[int4{x - left, sy, plane, batch_idx}];
                else
                    *dst.ptr(batch_idx, y, x) = src[int3{x - left, sy, batch_idx}];
            }
        }
    }
}

template<class SrcWrapper, class DstWrapper>
__global__ void copyMakeBorderKernel(const SrcWrapper src, DstWrapper dst, const cuda::Tensor3DWrap<int, int32_t> left_,
                                     const cuda::Tensor3DWrap<int, int32_t> top_)
{
    const int x         = blockIdx.x * blockDim.x + threadIdx.x;
    const int y         = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();

    const int left    = *left_.ptr(0, 0, batch_idx);
    const int top     = *top_.ptr(0, 0, batch_idx);
    const int x_shift = x - left;
    const int y_shift = y - top;

    int out_height = dst.height(batch_idx), out_width = dst.width(batch_idx);

    if (x < out_width && y < out_height)
    {
        int3 srcCoord             = {x_shift, y_shift, batch_idx};
        *dst.ptr(batch_idx, y, x) = src[srcCoord];
    }
}

template<class SrcWrapper, typename T>
__global__ void copyMakeBorderPlanarKernel(const SrcWrapper src, cuda::Tensor4DWrap<T> dst,
                                           const cuda::Tensor3DWrap<int, int32_t> left_,
                                           const cuda::Tensor3DWrap<int, int32_t> top_, int out_height, int out_width,
                                           int plane)
{
    const int x         = blockIdx.x * blockDim.x + threadIdx.x;
    const int y         = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();

    const int left    = *left_.ptr(0, 0, batch_idx);
    const int top     = *top_.ptr(0, 0, batch_idx);
    const int x_shift = x - left;
    const int y_shift = y - top;

    if (x < out_width && y < out_height)
    {
        *dst.ptr(batch_idx, plane, y, x) = src[int4{x_shift, y_shift, plane, batch_idx}];
    }
}

template<class SrcWrapper, typename T>
__global__ void copyMakeBorderPlanarKernel(const SrcWrapper src, cuda::ImageBatchVarShapeWrap<T> dst,
                                           const cuda::Tensor3DWrap<int, int32_t> left_,
                                           const cuda::Tensor3DWrap<int, int32_t> top_, int plane)
{
    const int x         = blockIdx.x * blockDim.x + threadIdx.x;
    const int y         = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();

    const int left    = *left_.ptr(0, 0, batch_idx);
    const int top     = *top_.ptr(0, 0, batch_idx);
    const int x_shift = x - left;
    const int y_shift = y - top;

    int out_height = dst.height(batch_idx, plane), out_width = dst.width(batch_idx, plane);

    if (x < out_width && y < out_height)
    {
        *dst.ptr(batch_idx, plane, y, x) = src[int4{x_shift, y_shift, plane, batch_idx}];
    }
}

template<NVCVBorderType B, typename T, bool FORCE_SCALAR = false>
struct copyMakeBorderDispatcher
{
    static void call(const ImageBatchVarShapeDataStridedCuda &src, cuda::Tensor3DWrap<T> dst, const T &borderValue,
                     const cuda::Tensor3DWrap<int, int32_t> &left, const cuda::Tensor3DWrap<int, int32_t> &top,
                     int max_height, int max_width, cudaStream_t stream)
    {
        constexpr int BLOCK_Y
            = FORCE_SCALAR                                ? BLOCK / 4
            : B == NVCV_BORDER_CONSTANT && sizeof(T) == 1 ? 2
            : B == NVCV_BORDER_REFLECT101 && std::is_same_v<cuda::BaseType<T>, uchar> && cuda::NumElements<T> >= 3 ? 2
            : B == NVCV_BORDER_REFLECT101 || B == NVCV_BORDER_REPLICATE                                            ? 4
                                                                                                                   : 8;
        dim3 blockSize(BLOCK, BLOCK_Y, 1);

        cuda::BorderVarShapeWrap<const T, B> brdSrc(src, borderValue);
        constexpr int                        NIX
            = sizeof(T) == 1 && (B == NVCV_BORDER_CONSTANT || B == NVCV_BORDER_REPLICATE) ? 4 : CMBVS_NIX<T>;
        if constexpr (NIX > 1 && !FORCE_SCALAR)
        {
            dim3 gridSize(divUp(max_width, blockSize.x * NIX), divUp(max_height, blockSize.y), src.numImages());
            cuda::ImageBatchVarShapeWrap<const T> srcRaw(src);
            copyMakeBorderKernelVec<false, true, decltype(brdSrc), decltype(srcRaw), decltype(dst), T, NIX>
                <<<gridSize, blockSize, 0, stream>>>(brdSrc, srcRaw, dst, left, top, max_height, max_width, 0);
        }
        else
        {
            dim3 gridSize(divUp(max_width, blockSize.x), divUp(max_height, blockSize.y), src.numImages());
            copyMakeBorderKernel<<<gridSize, blockSize, 0, stream>>>(brdSrc, dst, left, top, max_height, max_width);
        }
        checkKernelErrors();

#ifdef CUDA_DEBUG_LOG
        checkCudaErrors(cudaStreamSynchronize(stream));
        checkCudaErrors(cudaGetLastError());
#endif
    }

    static void call(const ImageBatchVarShapeDataStridedCuda &src, cuda::ImageBatchVarShapeWrap<T> dst,
                     const T &borderValue, const cuda::Tensor3DWrap<int, int32_t> &left,
                     const cuda::Tensor3DWrap<int, int32_t> &top, int max_height, int max_width, cudaStream_t stream)
    {
        constexpr int BLOCK_Y
            = FORCE_SCALAR                                ? BLOCK / 4
            : B == NVCV_BORDER_CONSTANT && sizeof(T) == 1 ? 2
            : B == NVCV_BORDER_REFLECT101 && std::is_same_v<cuda::BaseType<T>, uchar> && cuda::NumElements<T> >= 3 ? 2
            : B == NVCV_BORDER_REFLECT101 || B == NVCV_BORDER_REPLICATE                                            ? 4
                                                                                                                   : 8;
        dim3 blockSize(BLOCK, BLOCK_Y, 1);

        cuda::BorderVarShapeWrap<const T, B> brdSrc(src, borderValue);
        constexpr int                        NIX
            = sizeof(T) == 1 && (B == NVCV_BORDER_CONSTANT || B == NVCV_BORDER_REPLICATE) ? 4 : CMBVS_NIX<T>;
        if constexpr (NIX > 1 && !FORCE_SCALAR)
        {
            dim3 gridSize(divUp(max_width, blockSize.x * NIX), divUp(max_height, blockSize.y), src.numImages());
            cuda::ImageBatchVarShapeWrap<const T> srcRaw(src);
            copyMakeBorderKernelVec<false, false, decltype(brdSrc), decltype(srcRaw), decltype(dst), T, NIX>
                <<<gridSize, blockSize, 0, stream>>>(brdSrc, srcRaw, dst, left, top, 0, 0, 0);
        }
        else
        {
            dim3 gridSize(divUp(max_width, blockSize.x), divUp(max_height, blockSize.y), src.numImages());
            copyMakeBorderKernel<<<gridSize, blockSize, 0, stream>>>(brdSrc, dst, left, top);
        }
        checkKernelErrors();

#ifdef CUDA_DEBUG_LOG
        checkCudaErrors(cudaStreamSynchronize(stream));
        checkCudaErrors(cudaGetLastError());
#endif
    }
};

template<NVCVBorderType B, typename T>
struct copyMakeBorderPlanarDispatcher
{
    static void call(const ImageBatchVarShapeDataStridedCuda &src, cuda::Tensor4DWrap<T> dst, const float4 &borderValue,
                     const cuda::Tensor3DWrap<int, int32_t> &left, const cuda::Tensor3DWrap<int, int32_t> &top,
                     int max_height, int max_width, int channels, cudaStream_t stream)
    {
        dim3 blockSize(BLOCK, BLOCK / 4, 1);

        for (int c = 0; c < channels; ++c)
        {
            const T bVal = static_cast<T>(PlanarBorderComponent(borderValue, c));

            cuda::BorderVarShapeWrap<const T, B> brdSrc(src, bVal);
            constexpr int                        NIX = sizeof(T) == 1 ? 4 : CMBVS_NIX<T>;
            if constexpr (NIX > 1)
            {
                dim3 gridSize(divUp(max_width, blockSize.x * NIX), divUp(max_height, blockSize.y), src.numImages());
                cuda::ImageBatchVarShapeWrap<const T> srcRaw(src);
                copyMakeBorderKernelVec<true, true, decltype(brdSrc), decltype(srcRaw), decltype(dst), T, NIX>
                    <<<gridSize, blockSize, 0, stream>>>(brdSrc, srcRaw, dst, left, top, max_height, max_width, c);
            }
            else
            {
                dim3 gridSize(divUp(max_width, blockSize.x), divUp(max_height, blockSize.y), src.numImages());
                copyMakeBorderPlanarKernel<<<gridSize, blockSize, 0, stream>>>(brdSrc, dst, left, top, max_height,
                                                                               max_width, c);
            }
        }
        checkKernelErrors();
    }

    static void call(const ImageBatchVarShapeDataStridedCuda &src, cuda::ImageBatchVarShapeWrap<T> dst,
                     const float4 &borderValue, const cuda::Tensor3DWrap<int, int32_t> &left,
                     const cuda::Tensor3DWrap<int, int32_t> &top, int max_height, int max_width, int channels,
                     cudaStream_t stream)
    {
        dim3 blockSize(BLOCK, BLOCK / 4, 1);

        for (int c = 0; c < channels; ++c)
        {
            const T bVal = static_cast<T>(PlanarBorderComponent(borderValue, c));

            cuda::BorderVarShapeWrap<const T, B> brdSrc(src, bVal);
            constexpr int                        NIX = sizeof(T) == 1 ? 4 : CMBVS_NIX<T>;
            if constexpr (NIX > 1)
            {
                dim3 gridSize(divUp(max_width, blockSize.x * NIX), divUp(max_height, blockSize.y), src.numImages());
                cuda::ImageBatchVarShapeWrap<const T> srcRaw(src);
                copyMakeBorderKernelVec<true, false, decltype(brdSrc), decltype(srcRaw), decltype(dst), T, NIX>
                    <<<gridSize, blockSize, 0, stream>>>(brdSrc, srcRaw, dst, left, top, 0, 0, c);
            }
            else
            {
                dim3 gridSize(divUp(max_width, blockSize.x), divUp(max_height, blockSize.y), src.numImages());
                copyMakeBorderPlanarKernel<<<gridSize, blockSize, 0, stream>>>(brdSrc, dst, left, top, c);
            }
        }
        checkKernelErrors();
    }
};

template<typename T, int cn, typename OutType> // uchar3 float3 uchar float
void copyMakeBorder(const ImageBatchVarShapeDataStridedCuda &inData, const OutType &outData,
                    const TensorDataStridedCuda &top, const TensorDataStridedCuda &left,
                    const NVCVBorderType borderType, const float4 value, cudaStream_t stream)
{
    typedef cuda::MakeType<T, cn> src_type;
    src_type                      brdVal;
#pragma unroll
    for (int i = 0; i < cn; i++) cuda::GetElement(brdVal, i) = cuda::GetElement(value, i);

    cuda::Tensor3DWrap<int, int32_t> topVec(top);
    cuda::Tensor3DWrap<int, int32_t> leftVec(left);

    auto outSize = GetMaxImageSize(outData);

    using out_type =
        typename std::conditional<std::is_same<OutType, TensorDataStridedCuda>::value, cuda::Tensor3DWrap<src_type>,
                                  cuda::ImageBatchVarShapeWrap<src_type>>::type;

    out_type dstWrap(outData);

    if constexpr (std::is_same_v<src_type, uchar3>)
    {
        if (borderType == NVCV_BORDER_REFLECT101 && !UsePackedRGB8Reflect101ForSM(CurrentDeviceSMOrZero()))
        {
            copyMakeBorderDispatcher<NVCV_BORDER_REFLECT101, src_type, true>::call(
                inData, dstWrap, brdVal, leftVec, topVec, outSize.h, outSize.w, stream);
            return;
        }
    }

    typedef void (*func_t)(const ImageBatchVarShapeDataStridedCuda &src, out_type dst, const src_type &borderValue,
                           const cuda::Tensor3DWrap<int, int32_t> &left, const cuda::Tensor3DWrap<int, int32_t> &top,
                           int max_height, int max_width, cudaStream_t stream);

    static const func_t funcs[] = {copyMakeBorderDispatcher<NVCV_BORDER_CONSTANT, src_type>::call,
                                   copyMakeBorderDispatcher<NVCV_BORDER_REPLICATE, src_type>::call,
                                   copyMakeBorderDispatcher<NVCV_BORDER_REFLECT, src_type>::call,
                                   copyMakeBorderDispatcher<NVCV_BORDER_WRAP, src_type>::call,
                                   copyMakeBorderDispatcher<NVCV_BORDER_REFLECT101, src_type>::call};

    funcs[borderType](inData, dstWrap, brdVal, leftVec, topVec, outSize.h, outSize.w, stream);
}

template<typename T, typename OutType>
void copyMakeBorderPlanar(const ImageBatchVarShapeDataStridedCuda &inData, const OutType &outData,
                          const TensorDataStridedCuda &top, const TensorDataStridedCuda &left,
                          const NVCVBorderType borderType, const float4 value, cudaStream_t stream, int channels)
{
    cuda::Tensor3DWrap<int, int32_t> topVec(top);
    cuda::Tensor3DWrap<int, int32_t> leftVec(left);

    auto outSize = GetMaxImageSize(outData);

    if constexpr (std::is_same_v<OutType, TensorDataStridedCuda>)
    {
        using out_type = cuda::Tensor4DWrap<T>;

        auto     outView = PlanarTensorView(outData);
        out_type dstWrap(outView);

        typedef void (*func_t)(const ImageBatchVarShapeDataStridedCuda &src, out_type dst, const float4 &borderValue,
                               const cuda::Tensor3DWrap<int, int32_t> &left,
                               const cuda::Tensor3DWrap<int, int32_t> &top, int max_height, int max_width, int channels,
                               cudaStream_t stream);

        static const func_t funcs[] = {copyMakeBorderPlanarDispatcher<NVCV_BORDER_CONSTANT, T>::call,
                                       copyMakeBorderPlanarDispatcher<NVCV_BORDER_REPLICATE, T>::call,
                                       copyMakeBorderPlanarDispatcher<NVCV_BORDER_REFLECT, T>::call,
                                       copyMakeBorderPlanarDispatcher<NVCV_BORDER_WRAP, T>::call,
                                       copyMakeBorderPlanarDispatcher<NVCV_BORDER_REFLECT101, T>::call};

        funcs[borderType](inData, dstWrap, value, leftVec, topVec, outSize.h, outSize.w, channels, stream);
    }
    else
    {
        using out_type = cuda::ImageBatchVarShapeWrap<T>;

        out_type dstWrap(outData);

        typedef void (*func_t)(const ImageBatchVarShapeDataStridedCuda &src, out_type dst, const float4 &borderValue,
                               const cuda::Tensor3DWrap<int, int32_t> &left,
                               const cuda::Tensor3DWrap<int, int32_t> &top, int max_height, int max_width, int channels,
                               cudaStream_t stream);

        static const func_t funcs[] = {copyMakeBorderPlanarDispatcher<NVCV_BORDER_CONSTANT, T>::call,
                                       copyMakeBorderPlanarDispatcher<NVCV_BORDER_REPLICATE, T>::call,
                                       copyMakeBorderPlanarDispatcher<NVCV_BORDER_REFLECT, T>::call,
                                       copyMakeBorderPlanarDispatcher<NVCV_BORDER_WRAP, T>::call,
                                       copyMakeBorderPlanarDispatcher<NVCV_BORDER_REFLECT101, T>::call};

        funcs[borderType](inData, dstWrap, value, leftVec, topVec, outSize.h, outSize.w, channels, stream);
    }
}
} // namespace

template<class OutType>
ErrorCode CopyMakeBorderVarShape::inferWarp(const ImageBatchVarShapeDataStridedCuda &data_in, const OutType &data_out,
                                            const TensorDataStridedCuda &top, const TensorDataStridedCuda &left,
                                            const NVCVBorderType borderType, const float4 value, cudaStream_t stream)
{
    DataFormat input_format  = GetLegacyDataFormat(data_in);
    DataFormat output_format = GetLegacyDataFormat(data_out);
    if (std::is_same<decltype(data_in), decltype(data_out)>::value)
    {
        if (input_format != output_format)
        {
            LOG_ERROR("Invalid DataFormat between input (" << input_format << ") and output (" << output_format << ")");
            return ErrorCode::INVALID_DATA_FORMAT;
        }
    }
    else if (!SameLayoutFamily(input_format, output_format))
    {
        LOG_ERROR("Invalid DataFormat between input (" << input_format << ") and output (" << output_format << ")");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    auto format = input_format;
    if (!(format == kNHWC || format == kHWC || format == kNCHW || format == kCHW))
    {
        LOG_ERROR("Invalid input DataFormat " << format
                                              << ", the valid DataFormats are: \"NHWC\", \"HWC\", \"NCHW\", \"CHW\"");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    const bool isPlanar = IsPlanar(format);

    int channels = data_in.uniqueFormat().numChannels();
    if (channels > 4 || (isPlanar && channels == 2))
    {
        LOG_ERROR("Invalid channel number " << channels);
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    DataType data_type     = GetLegacyDataType(data_in.uniqueFormat());
    DataType out_data_type = GetOutputDataType(data_out);

    if (data_type != out_data_type)
    {
        LOG_ERROR("DataType of input and output must be equal, but got " << data_type << " and " << out_data_type);
        return ErrorCode::INVALID_DATA_TYPE;
    }

    if (!isPlanar && channels == 2 && data_type != kCV_8U)
    {
        LOG_ERROR("Invalid channel number " << channels << " for DataType " << data_type
                                            << ", 2 channels only supported for 8bit Unsigned");
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    if (!(data_type == kCV_8U || data_type == kCV_16U || data_type == kCV_16S || data_type == kCV_32F))
    {
        LOG_ERROR("Invalid DataType " << data_type);
        return ErrorCode::INVALID_DATA_TYPE;
    }

    if (!(borderType == NVCV_BORDER_CONSTANT || borderType == NVCV_BORDER_REPLICATE || borderType == NVCV_BORDER_REFLECT
          || borderType == NVCV_BORDER_REFLECT101 || borderType == NVCV_BORDER_WRAP))
    {
        LOG_ERROR("Invalid borderType " << borderType);
        return ErrorCode::INVALID_PARAMETER;
    }

    DataType   left_data_type = GetLegacyDataType(left.dtype());
    DataFormat left_format    = GetLegacyDataFormat(left.layout());
    if (!(left_format == kNHWC || left_format == kHWC))
    {
        LOG_ERROR("Invalid Left DataFormat " << left_format);
        return ErrorCode::INVALID_DATA_FORMAT;
    }
    if (left_data_type != kCV_32S)
    {
        LOG_ERROR("Invalid Left DataType " << left_data_type);
        return ErrorCode::INVALID_DATA_TYPE;
    }

    DataType   top_data_type = GetLegacyDataType(top.dtype());
    DataFormat top_format    = GetLegacyDataFormat(top.layout());
    if (!(top_format == kNHWC || top_format == kHWC))
    {
        LOG_ERROR("Invalid Top DataFormat " << top_format);
        return ErrorCode::INVALID_DATA_FORMAT;
    }
    if (top_data_type != kCV_32S)
    {
        LOG_ERROR("Invalid Top DataType " << top_data_type);
        return ErrorCode::INVALID_DATA_TYPE;
    }

    typedef void (*func_t)(const ImageBatchVarShapeDataStridedCuda &d_in, const OutType &d_out,
                           const TensorDataStridedCuda &top, const TensorDataStridedCuda &left,
                           const NVCVBorderType borderType, const float4 value, cudaStream_t stream);

    // clang-format off
    static const func_t funcs[6][4] = {
        {copyMakeBorder<uchar, 1>,        copyMakeBorder<uchar, 2>,        copyMakeBorder<uchar, 3>,        copyMakeBorder<uchar, 4>       },
        {0 /*copyMakeBorder<schar , 1>*/, 0 /*copyMakeBorder<schar , 2>*/, 0 /*copyMakeBorder<schar , 3>*/, 0 /*copyMakeBorder<schar , 4>*/},
        {copyMakeBorder<ushort, 1>,       0 /*copyMakeBorder<ushort, 2>*/, copyMakeBorder<ushort, 3>,       copyMakeBorder<ushort, 4>      },
        {copyMakeBorder<short, 1>,        0 /*copyMakeBorder<short , 2>*/, copyMakeBorder<short, 3>,        copyMakeBorder<short, 4>       },
        {0 /*copyMakeBorder<int   , 1>*/, 0 /*copyMakeBorder<int   , 2>*/, 0 /*copyMakeBorder<int   , 3>*/, 0 /*copyMakeBorder<int   , 4>*/},
        {copyMakeBorder<float, 1>,        0 /*copyMakeBorder<float , 2>*/, copyMakeBorder<float, 3>,        copyMakeBorder<float, 4>       }
    };
    // clang-format on

    const func_t func = funcs[data_type][channels - 1];

    if (isPlanar)
    {
        typedef void (*planar_func_t)(const ImageBatchVarShapeDataStridedCuda &d_in, const OutType &d_out,
                                      const TensorDataStridedCuda &top, const TensorDataStridedCuda &left,
                                      const NVCVBorderType borderType, const float4 value, cudaStream_t stream,
                                      int channels);

        static const planar_func_t planarFuncs[6] = {
            copyMakeBorderPlanar<uchar>, 0 /*schar*/, copyMakeBorderPlanar<ushort>,
            copyMakeBorderPlanar<short>, 0 /*int*/,   copyMakeBorderPlanar<float>,
        };

        const planar_func_t planarFunc = planarFuncs[data_type];
        NVCV_ASSERT(planarFunc != 0);
        planarFunc(data_in, data_out, top, left, borderType, value, stream, channels);
        return SUCCESS;
    }

    NVCV_ASSERT(func != 0);

    func(data_in, data_out, top, left, borderType, value, stream);

    return SUCCESS;
}

ErrorCode CopyMakeBorderVarShape::infer(const ImageBatchVarShapeDataStridedCuda &data_in,
                                        const ImageBatchVarShapeDataStridedCuda &data_out,
                                        const TensorDataStridedCuda &top, const TensorDataStridedCuda &left,
                                        const NVCVBorderType borderType, const float4 value, cudaStream_t stream)
{
    return inferWarp(data_in, data_out, top, left, borderType, value, stream);
}

ErrorCode CopyMakeBorderVarShape::infer(const ImageBatchVarShapeDataStridedCuda &data_in,
                                        const TensorDataStridedCuda &data_out, const TensorDataStridedCuda &top,
                                        const TensorDataStridedCuda &left, const NVCVBorderType borderType,
                                        const float4 value, cudaStream_t stream)
{
    return inferWarp(data_in, data_out, top, left, borderType, value, stream);
}

} // namespace nvcv::legacy::cuda_op
