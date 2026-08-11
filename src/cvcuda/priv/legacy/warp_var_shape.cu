/* Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * SPDX-License-Identifier: Apache-2.0
 *
 * Copyright (C) 2021-2022, Bytedance Inc. All rights reserved.
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

#include "CvCudaLegacy.h"
#include "CvCudaLegacyHelpers.hpp"

#include "CvCudaUtils.cuh"
#include "warp_cubic.cuh"

#include <type_traits>

#define BLOCK 32

namespace nvcv::legacy::cuda_op {

__global__ void inverseMatWarpPerspective(const int numImages, const cuda::Tensor2DWrap<float, int32_t> in,
                                          cuda::Tensor2DWrap<float, int32_t> out)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= numImages)
    {
        return;
    }

    cuda::math::Matrix<float, 3, 3> transMatrix;

    transMatrix.load(in.ptr(index));

    cuda::math::inv_inplace(transMatrix);

    transMatrix.store(out.ptr(index));
}

__global__ void inverseMatWarpAffine(const int numImages, const cuda::Tensor2DWrap<float, int32_t> in,
                                     cuda::Tensor2DWrap<float, int32_t> out)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= numImages)
    {
        return;
    }

    cuda::math::Matrix<float, 2, 3> M;
    M[0][0] = (float)(*in.ptr(index, 0));
    M[0][1] = (float)(*in.ptr(index, 1));
    M[0][2] = (float)(*in.ptr(index, 2));
    M[1][0] = (float)(*in.ptr(index, 3));
    M[1][1] = (float)(*in.ptr(index, 4));
    M[1][2] = (float)(*in.ptr(index, 5));

    // M is stored in row-major format M[0,0], M[0,1], M[0,2], M[1,0], M[1,1], M[1,2]
    float den          = M[0][0] * M[1][1] - M[0][1] * M[1][0];
    den                = std::abs(den) > 1e-5 ? 1. / den : .0;
    *out.ptr(index, 0) = (float)M[1][1] * den;
    *out.ptr(index, 1) = (float)-M[0][1] * den;
    *out.ptr(index, 2) = (float)(M[0][1] * M[1][2] - M[1][1] * M[0][2]) * den;
    *out.ptr(index, 3) = (float)-M[1][0] * den;
    *out.ptr(index, 4) = (float)M[0][0] * den;
    *out.ptr(index, 5) = (float)(M[1][0] * M[0][2] - M[0][0] * M[1][2]) * den;
}

template<class Transform, class SrcWrapper, class DstWrapper>
__global__ void warp(SrcWrapper src, DstWrapper dst, const cuda::Tensor2DWrap<float, int32_t> coeffs)
{
    using SrcValueT       = std::remove_cv_t<typename SrcWrapper::ValueType>;
    constexpr bool kCubic = SrcWrapper::kInterpolationType == NVCV_INTERP_CUBIC;
    // The restructured sampler pays off for affine maps (the scatter-heavy case); perspective
    // maps keep the wrap path, whose per-tap loads coalesce well and carry no extra registers.
    constexpr bool kFast = kCubic && kCubicFastSampler<SrcValueT> && std::is_same_v<Transform, WarpAffineTransform>;

    int3      dstCoord = cuda::StaticCast<int>(blockDim * blockIdx + threadIdx);
    const int lid      = threadIdx.y * blockDim.x + threadIdx.x;

    extern __shared__ float coeff[];

    if (lid < 9)
    {
        coeff[lid] = *coeffs.ptr(dstCoord.z, lid);
    }

    __syncthreads();

    if (dstCoord.x < dst.width(dstCoord.z) && dstCoord.y < dst.height(dstCoord.z))
    {
        const float2 coord = Transform::calcCoord(coeff, dstCoord.x, dstCoord.y);

        if constexpr (kFast)
        {
            dst[dstCoord] = CubicSampleVarShape(src, dstCoord.z, coord);
        }
        else
        {
            dst[dstCoord] = src[float3{coord.x, coord.y, static_cast<float>(dstCoord.z)}];
        }
    }
}

// LINEAR Perspective benefits from amortizing per-block coefficient staging across two coalesced X outputs. Keep a
// separate kernel so WarpAffine and the other interpolation modes retain their qualified code generation.
template<class Transform, class SrcWrapper, class DstWrapper>
__global__ void warp_x2(SrcWrapper src, DstWrapper dst, const cuda::Tensor2DWrap<float, int32_t> coeffs)
{
    constexpr int kNIX  = 2;
    const int     dstX0 = blockIdx.x * blockDim.x * kNIX + threadIdx.x;
    const int     dstY  = blockIdx.y * blockDim.y + threadIdx.y;
    const int     batch = blockIdx.z;
    const int     lid   = threadIdx.y * blockDim.x + threadIdx.x;

    extern __shared__ float coeff[];

    if (lid < 9)
    {
        coeff[lid] = *coeffs.ptr(batch, lid);
    }

    __syncthreads();

    const int dstWidth  = dst.width(batch);
    const int dstHeight = dst.height(batch);
    if (dstY < dstHeight)
    {
#pragma unroll
        for (int i = 0; i < kNIX; ++i)
        {
            const int dstX = dstX0 + i * static_cast<int>(blockDim.x);
            if (dstX < dstWidth)
            {
                const float2 coord = Transform::calcCoord(coeff, dstX, dstY);
                const float3 srcCoord{coord.x, coord.y, static_cast<float>(batch)};

                dst[int3{dstX, dstY, batch}] = src[srcCoord];
            }
        }
    }
}

template<class Transform, typename T, NVCVBorderType B, NVCVInterpolationType I>
struct WarpDispatcher
{
    static void call(const ImageBatchVarShapeDataStridedCuda &inData, const ImageBatchVarShapeDataStridedCuda &outData,
                     const cuda::Tensor2DWrap<float, int32_t> transform, const float4 &borderValue, cudaStream_t stream)
    {
        Size2D outMaxSize = outData.maxSize();

        constexpr bool useX2 = std::is_same_v<Transform, PerspectiveTransform> && I == NVCV_INTERP_LINEAR;
        constexpr int  kNIX  = useX2 ? 2 : 1;
        dim3           block(BLOCK, BLOCK / 4);
        dim3           grid(divUp(outMaxSize.w, block.x * kNIX), divUp(outMaxSize.h, block.y), outData.numImages());

        auto bVal = cuda::StaticCast<cuda::BaseType<T>>(cuda::DropCast<cuda::NumElements<T>>(borderValue));

        cuda::InterpolationVarShapeWrap<const T, B, I> src(inData, bVal);
        cuda::ImageBatchVarShapeWrap<T>                dst(outData);

        size_t smem_size = 9 * sizeof(float);

        if constexpr (useX2)
        {
            warp_x2<Transform><<<grid, block, smem_size, stream>>>(src, dst, transform);
        }
        else
        {
            warp<Transform><<<grid, block, smem_size, stream>>>(src, dst, transform);
        }
        checkKernelErrors();
    }
};

template<class Transform, typename T>
void warp_caller(const ImageBatchVarShapeDataStridedCuda &inData, const ImageBatchVarShapeDataStridedCuda &outData,
                 cuda::Tensor2DWrap<float, int32_t> transform, const int interpolation, const int borderMode,
                 const float4 &borderValue, cudaStream_t stream)
{
    typedef void (*func_t)(
        const ImageBatchVarShapeDataStridedCuda &inData, const ImageBatchVarShapeDataStridedCuda &outData,
        cuda::Tensor2DWrap<float, int32_t> transform, const float4 &borderValue, cudaStream_t stream);

    static const func_t funcs[3][5] = {
        {WarpDispatcher<Transform, T, NVCV_BORDER_CONSTANT, NVCV_INTERP_NEAREST>::call,
         WarpDispatcher<Transform, T, NVCV_BORDER_REPLICATE, NVCV_INTERP_NEAREST>::call,
         WarpDispatcher<Transform, T, NVCV_BORDER_REFLECT, NVCV_INTERP_NEAREST>::call,
         WarpDispatcher<Transform, T, NVCV_BORDER_WRAP, NVCV_INTERP_NEAREST>::call,
         WarpDispatcher<Transform, T, NVCV_BORDER_REFLECT101, NVCV_INTERP_NEAREST>::call},
        {WarpDispatcher<Transform, T, NVCV_BORDER_CONSTANT,  NVCV_INTERP_LINEAR>::call,
         WarpDispatcher<Transform, T, NVCV_BORDER_REPLICATE,  NVCV_INTERP_LINEAR>::call,
         WarpDispatcher<Transform, T, NVCV_BORDER_REFLECT,  NVCV_INTERP_LINEAR>::call,
         WarpDispatcher<Transform, T, NVCV_BORDER_WRAP,  NVCV_INTERP_LINEAR>::call,
         WarpDispatcher<Transform, T, NVCV_BORDER_REFLECT101,  NVCV_INTERP_LINEAR>::call},
        {WarpDispatcher<Transform, T, NVCV_BORDER_CONSTANT,   NVCV_INTERP_CUBIC>::call,
         WarpDispatcher<Transform, T, NVCV_BORDER_REPLICATE,   NVCV_INTERP_CUBIC>::call,
         WarpDispatcher<Transform, T, NVCV_BORDER_REFLECT,   NVCV_INTERP_CUBIC>::call,
         WarpDispatcher<Transform, T, NVCV_BORDER_WRAP,   NVCV_INTERP_CUBIC>::call,
         WarpDispatcher<Transform, T, NVCV_BORDER_REFLECT101,   NVCV_INTERP_CUBIC>::call},
    };

    funcs[interpolation][borderMode](inData, outData, transform, borderValue, stream);
}

template<typename T>
void warpAffine(const ImageBatchVarShapeDataStridedCuda &inData, const ImageBatchVarShapeDataStridedCuda &outData,
                cuda::Tensor2DWrap<float, int32_t> transform, const int interpolation, const int borderMode,
                const float4 &borderValue, cudaStream_t stream)
{
    warp_caller<WarpAffineTransform, T>(inData, outData, transform, interpolation, borderMode, borderValue, stream);
}

template<typename T>
void warpPerspective(const ImageBatchVarShapeDataStridedCuda &inData, const ImageBatchVarShapeDataStridedCuda &outData,
                     cuda::Tensor2DWrap<float, int32_t> transform, const int interpolation, const int borderMode,
                     const float4 &borderValue, cudaStream_t stream)
{
    warp_caller<PerspectiveTransform, T>(inData, outData, transform, interpolation, borderMode, borderValue, stream);
}

// Planar (NCHW/CHW) var-shape warp. Warp samples each channel at the same transformed coordinate, so
// each plane is an independent single-channel image: this kernel warps one plane (selected by the
// plane-indexed (sample, plane, y, x) coordinate), reusing the interpolation/border wrap. One launch
// per plane lets each plane carry its own constant-border component. Bit-exact with the interleaved
// single-channel warp.
// Fused planar var-shape CUBIC warp: one launch covers every channel plane, sharing the
// transformed coordinate, cubic weights, and border resolution across planes instead of
// relaunching the single-plane kernel per plane.
template<class Transform, typename BT, NVCVBorderType B, int NP, class SrcWrapper, class DstWrapper>
__global__ void warp_planar_fused(SrcWrapper src, DstWrapper dst, const cuda::Tensor2DWrap<float, int32_t> coeffs,
                                  float4 borderValue)
{
    const int3 dstCoord = cuda::StaticCast<int>(blockDim * blockIdx + threadIdx);
    const int  lid      = threadIdx.y * blockDim.x + threadIdx.x;

    extern __shared__ float coeff[];

    if (lid < 9)
    {
        coeff[lid] = *coeffs.ptr(dstCoord.z, lid);
    }

    __syncthreads();

    const int z = dstCoord.z;
    if (dstCoord.x < dst.width(z) && dstCoord.y < dst.height(z))
    {
        const float2 coord = Transform::calcCoord(coeff, dstCoord.x, dstCoord.y);

        CubicWarpPlanes<BT, B, NP, int>(
            [&](int p, int yy) { return reinterpret_cast<const BT *>(src.ptr(z, p, yy, 0)); },
            [&](int p, BT v) { *reinterpret_cast<BT *>(dst.ptr(z, p, dstCoord.y, dstCoord.x)) = v; }, borderValue,
            int2{src.width(z), src.height(z)}, coord);
    }
}

template<class Transform, class SrcWrapper, class DstWrapper>
__global__ void warp_planar(SrcWrapper src, DstWrapper dst, const cuda::Tensor2DWrap<float, int32_t> coeffs, int plane)
{
    int3      dstCoord = cuda::StaticCast<int>(blockDim * blockIdx + threadIdx);
    const int lid      = threadIdx.y * blockDim.x + threadIdx.x;

    extern __shared__ float coeff[];

    if (lid < 9)
    {
        coeff[lid] = *coeffs.ptr(dstCoord.z, lid);
    }

    __syncthreads();

    const int batch = dstCoord.z;
    if (dstCoord.x < dst.width(batch) && dstCoord.y < dst.height(batch))
    {
        const float2 coord = Transform::calcCoord(coeff, dstCoord.x, dstCoord.y);

        constexpr bool kFast
            = SrcWrapper::kInterpolationType == NVCV_INTERP_CUBIC
           && kCubicFastSampler<
                  std::remove_cv_t<typename SrcWrapper::ValueType>> && std::is_same_v<Transform, WarpAffineTransform>;
        if constexpr (kFast)
        {
            dst[int4{dstCoord.x, dstCoord.y, plane, batch}] = CubicSampleVarShapePlane(src, batch, plane, coord);
        }
        else
        {
            dst[int4{dstCoord.x, dstCoord.y, plane, batch}]
                = src[float4{coord.x, coord.y, static_cast<float>(plane), static_cast<float>(batch)}];
        }
    }
}

template<class Transform, typename T, NVCVBorderType B, NVCVInterpolationType I>
struct WarpPlanarDispatcher
{
    static void call(const ImageBatchVarShapeDataStridedCuda &inData, const ImageBatchVarShapeDataStridedCuda &outData,
                     const cuda::Tensor2DWrap<float, int32_t> transform, const float4 &borderValue, int channels,
                     cudaStream_t stream)
    {
        Size2D outMaxSize = outData.maxSize();

        dim3   block(BLOCK, BLOCK / 4);
        dim3   grid(divUp(outMaxSize.w, block.x), divUp(outMaxSize.h, block.y), outData.numImages());
        size_t smem_size = 9 * sizeof(float);

        cuda::ImageBatchVarShapeWrap<T> dst(outData);

        // Fused planar only pays for byte-based types; float planes keep per-plane launches
        // (each already using the fast CUBIC sampler).
        if constexpr (I == NVCV_INTERP_CUBIC && sizeof(cuda::BaseType<T>) == 1)
        {
            cuda::ImageBatchVarShapeWrap<const T> src(inData);

            switch (channels)
            {
            case 1:
                warp_planar_fused<Transform, cuda::BaseType<T>, B, 1>
                    <<<grid, block, smem_size, stream>>>(src, dst, transform, borderValue);
                break;
            case 3:
                warp_planar_fused<Transform, cuda::BaseType<T>, B, 3>
                    <<<grid, block, smem_size, stream>>>(src, dst, transform, borderValue);
                break;
            case 4:
                warp_planar_fused<Transform, cuda::BaseType<T>, B, 4>
                    <<<grid, block, smem_size, stream>>>(src, dst, transform, borderValue);
                break;
            default:
                break;
            }
        }
        else
        {
            for (int c = 0; c < channels; ++c)
            {
                const float bc = c == 0 ? borderValue.x
                               : c == 1 ? borderValue.y
                               : c == 2 ? borderValue.z
                                        : borderValue.w;
                auto        bVal
                    = cuda::StaticCast<cuda::BaseType<T>>(cuda::DropCast<cuda::NumElements<T>>(float4{bc, bc, bc, bc}));

                cuda::InterpolationVarShapeWrap<const T, B, I> src(inData, bVal);

                warp_planar<Transform><<<grid, block, smem_size, stream>>>(src, dst, transform, c);
            }
        }
        checkKernelErrors();
    }
};

template<class Transform, typename T>
void warp_planar_caller(const ImageBatchVarShapeDataStridedCuda &inData,
                        const ImageBatchVarShapeDataStridedCuda &outData, cuda::Tensor2DWrap<float, int32_t> transform,
                        const int interpolation, const int borderMode, int channels, const float4 &borderValue,
                        cudaStream_t stream)
{
    typedef void (*func_t)(const ImageBatchVarShapeDataStridedCuda &inData,
                           const ImageBatchVarShapeDataStridedCuda &outData,
                           const cuda::Tensor2DWrap<float, int32_t> transform, const float4 &borderValue, int channels,
                           cudaStream_t stream);

    static const func_t funcs[3][5] = {
        {WarpPlanarDispatcher<Transform, T, NVCV_BORDER_CONSTANT, NVCV_INTERP_NEAREST>::call,
         WarpPlanarDispatcher<Transform, T, NVCV_BORDER_REPLICATE, NVCV_INTERP_NEAREST>::call,
         WarpPlanarDispatcher<Transform, T, NVCV_BORDER_REFLECT, NVCV_INTERP_NEAREST>::call,
         WarpPlanarDispatcher<Transform, T, NVCV_BORDER_WRAP, NVCV_INTERP_NEAREST>::call,
         WarpPlanarDispatcher<Transform, T, NVCV_BORDER_REFLECT101, NVCV_INTERP_NEAREST>::call},
        {WarpPlanarDispatcher<Transform, T, NVCV_BORDER_CONSTANT,  NVCV_INTERP_LINEAR>::call,
         WarpPlanarDispatcher<Transform, T, NVCV_BORDER_REPLICATE,  NVCV_INTERP_LINEAR>::call,
         WarpPlanarDispatcher<Transform, T, NVCV_BORDER_REFLECT,  NVCV_INTERP_LINEAR>::call,
         WarpPlanarDispatcher<Transform, T, NVCV_BORDER_WRAP,  NVCV_INTERP_LINEAR>::call,
         WarpPlanarDispatcher<Transform, T, NVCV_BORDER_REFLECT101,  NVCV_INTERP_LINEAR>::call},
        {WarpPlanarDispatcher<Transform, T, NVCV_BORDER_CONSTANT,   NVCV_INTERP_CUBIC>::call,
         WarpPlanarDispatcher<Transform, T, NVCV_BORDER_REPLICATE,   NVCV_INTERP_CUBIC>::call,
         WarpPlanarDispatcher<Transform, T, NVCV_BORDER_REFLECT,   NVCV_INTERP_CUBIC>::call,
         WarpPlanarDispatcher<Transform, T, NVCV_BORDER_WRAP,   NVCV_INTERP_CUBIC>::call,
         WarpPlanarDispatcher<Transform, T, NVCV_BORDER_REFLECT101,   NVCV_INTERP_CUBIC>::call},
    };

    funcs[interpolation][borderMode](inData, outData, transform, borderValue, channels, stream);
}

template<typename T>
void warpAffine_planar(const ImageBatchVarShapeDataStridedCuda &inData,
                       const ImageBatchVarShapeDataStridedCuda &outData, cuda::Tensor2DWrap<float, int32_t> transform,
                       const int interpolation, const int borderMode, int channels, const float4 &borderValue,
                       cudaStream_t stream)
{
    warp_planar_caller<WarpAffineTransform, T>(inData, outData, transform, interpolation, borderMode, channels,
                                               borderValue, stream);
}

template<typename T>
void warpPerspective_planar(const ImageBatchVarShapeDataStridedCuda &inData,
                            const ImageBatchVarShapeDataStridedCuda &outData,
                            cuda::Tensor2DWrap<float, int32_t> transform, const int interpolation, const int borderMode,
                            int channels, const float4 &borderValue, cudaStream_t stream)
{
    warp_planar_caller<PerspectiveTransform, T>(inData, outData, transform, interpolation, borderMode, channels,
                                                borderValue, stream);
}

WarpAffineVarShape::WarpAffineVarShape(const int32_t maxBatchSize)
    : CudaBaseOp()
    , m_maxBatchSize(maxBatchSize)
{
    if (m_maxBatchSize > 0)
    {
        // Allocating for 9 floats even though only 6 are required because the CUDA kernel
        // is shared between affine & perspective
        size_t bufferSize = sizeof(float) * 9 * m_maxBatchSize;
        NVCV_CHECK_LOG(cudaMalloc(&m_transformationMatrix, bufferSize));
    }
}

WarpAffineVarShape::~WarpAffineVarShape()
{
    if (m_transformationMatrix != nullptr)
    {
        NVCV_CHECK_LOG(cudaFree(m_transformationMatrix));
    }
    m_transformationMatrix = nullptr;
}

ErrorCode WarpAffineVarShape::infer(const ImageBatchVarShapeDataStridedCuda &inData,
                                    const ImageBatchVarShapeDataStridedCuda &outData,
                                    const TensorDataStridedCuda &transMatrix, const int32_t flags,
                                    const NVCVBorderType borderMode, const float4 borderValue, cudaStream_t stream)
{
    if (m_maxBatchSize <= 0)
    {
        LOG_ERROR("Operator warp perspective var shape is not initialized properly, maxVarShapeBatchSize: "
                  << m_maxBatchSize);
        return ErrorCode::INVALID_PARAMETER;
    }

    if (!inData.uniqueFormat())
    {
        LOG_ERROR("Images in the input varshape must all have the same format");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    if (m_maxBatchSize < inData.numImages())
    {
        LOG_ERROR("Invalid number of images, it should not exceed " << m_maxBatchSize);
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    DataFormat input_format  = helpers::GetLegacyDataFormat(inData);
    DataFormat output_format = helpers::GetLegacyDataFormat(outData);

    if (input_format != output_format)
    {
        LOG_ERROR("Invalid DataFormat between input (" << input_format << ") and output (" << output_format << ")");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    DataFormat format = input_format;

    if (!(format == kNHWC || format == kHWC || format == kNCHW || format == kCHW))
    {
        LOG_ERROR("Invalid input DataFormat " << format
                                              << ", the valid DataFormats are: \"NHWC\", \"HWC\", \"NCHW\", \"CHW\"");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    const bool isPlanar = (format == kNCHW || format == kCHW);

    int channels = inData.uniqueFormat().numChannels();

    if (channels > 4 || channels == 2)
    {
        LOG_ERROR("Invalid channel number " << channels);
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    const int interpolation = flags & NVCV_INTERP_MAX;

    DataType data_type = helpers::GetLegacyDataType(inData.uniqueFormat());

    if (!(data_type == kCV_8U || data_type == kCV_16U || data_type == kCV_16S || data_type == kCV_32F))
    {
        LOG_ERROR("Invalid DataType " << data_type);
        return ErrorCode::INVALID_DATA_TYPE;
    }

    if (!(interpolation == NVCV_INTERP_NEAREST || interpolation == NVCV_INTERP_LINEAR
          || interpolation == NVCV_INTERP_CUBIC))
    {
        LOG_ERROR("Invalid interpolation " << interpolation);
        return ErrorCode::INVALID_PARAMETER;
    }
    if (!(borderMode == NVCV_BORDER_CONSTANT || borderMode == NVCV_BORDER_REPLICATE || borderMode == NVCV_BORDER_REFLECT
          || borderMode == NVCV_BORDER_WRAP || borderMode == NVCV_BORDER_REFLECT101))
    {
        LOG_ERROR("Invalid borderMode " << borderMode);
        return ErrorCode::INVALID_PARAMETER;
    }

    // Check if inverse op is needed
    bool performInverse = !(flags & NVCV_WARP_INVERSE_MAP);

    // Wrap the matrix in 2D wrappers with proper pitch
    cuda::Tensor2DWrap<float, int32_t> transMatrixInput(transMatrix);
    cuda::Tensor2DWrap<float, int32_t> transMatrixOutput(m_transformationMatrix, static_cast<int>(sizeof(float) * 9));

    if (performInverse)
    {
        inverseMatWarpAffine<<<1, inData.numImages(), 0, stream>>>(inData.numImages(), transMatrixInput,
                                                                   transMatrixOutput);
        checkKernelErrors();
    }
    else
    {
        NVCV_CHECK_LOG(cudaMemcpy2DAsync(m_transformationMatrix, sizeof(float) * 9, transMatrixInput.ptr(0, 0),
                                         transMatrixInput.strides()[0], sizeof(float) * 6, inData.numImages(),
                                         cudaMemcpyDeviceToDevice, stream));
    }

    typedef void (*func_t)(const ImageBatchVarShapeDataStridedCuda &inData,
                           const ImageBatchVarShapeDataStridedCuda &outData,
                           const cuda::Tensor2DWrap<float, int32_t> transform, const int interpolation,
                           const int borderMode, const float4 &borderValue, cudaStream_t stream);

    static const func_t funcs[6][4] = {
        {     warpAffine<uchar1>,  0 /*warpAffine<uchar2>*/,      warpAffine<uchar3>,      warpAffine<uchar4>},
        {0 /*warpAffine<schar>*/,   0 /*warpAffine<char2>*/, 0 /*warpAffine<char3>*/, 0 /*warpAffine<char4>*/},
        {    warpAffine<ushort1>, 0 /*warpAffine<ushort2>*/,     warpAffine<ushort3>,     warpAffine<ushort4>},
        {     warpAffine<short1>,  0 /*warpAffine<short2>*/,      warpAffine<short3>,      warpAffine<short4>},
        {  0 /*warpAffine<int>*/,    0 /*warpAffine<int2>*/,  0 /*warpAffine<int3>*/,  0 /*warpAffine<int4>*/},
        {     warpAffine<float1>,  0 /*warpAffine<float2>*/,      warpAffine<float3>,      warpAffine<float4>}
    };

    if (isPlanar)
    {
        // Planar dispatch indexes by dtype only: each channel is warped as a single-channel plane
        // (per-channel border value handled inside the planar caller).
        typedef void (*planar_func_t)(
            const ImageBatchVarShapeDataStridedCuda &inData, const ImageBatchVarShapeDataStridedCuda &outData,
            cuda::Tensor2DWrap<float, int32_t> transform, const int interpolation, const int borderMode, int channels,
            const float4 &borderValue, cudaStream_t stream);

        static const planar_func_t planar_funcs[6] = {
            warpAffine_planar<uchar1>, 0 /*schar*/, warpAffine_planar<ushort1>,
            warpAffine_planar<short1>, 0 /*int*/,   warpAffine_planar<float1>,
        };

        const planar_func_t planarFunc = planar_funcs[data_type];
        NVCV_ASSERT(planarFunc != 0);
        planarFunc(inData, outData, transMatrixOutput, interpolation, borderMode, channels, borderValue, stream);
        return SUCCESS;
    }

    const func_t func = funcs[data_type][channels - 1];
    NVCV_ASSERT(func != 0);

    func(inData, outData, transMatrixOutput, interpolation, borderMode, borderValue, stream);
    return SUCCESS;
}

WarpPerspectiveVarShape::WarpPerspectiveVarShape(const int32_t maxBatchSize)
    : CudaBaseOp()
    , m_maxBatchSize(maxBatchSize)
{
    if (m_maxBatchSize > 0)
    {
        size_t bufferSize = sizeof(float) * 9 * m_maxBatchSize;
        NVCV_CHECK_LOG(cudaMalloc(&m_transformationMatrix, bufferSize));
    }
}

WarpPerspectiveVarShape::~WarpPerspectiveVarShape()
{
    if (m_transformationMatrix != nullptr)
    {
        NVCV_CHECK_LOG(cudaFree(m_transformationMatrix));
    }
    m_transformationMatrix = nullptr;
}

ErrorCode WarpPerspectiveVarShape::infer(const ImageBatchVarShapeDataStridedCuda &inData,
                                         const ImageBatchVarShapeDataStridedCuda &outData,
                                         const TensorDataStridedCuda &transMatrix, const int32_t flags,
                                         const NVCVBorderType borderMode, const float4 borderValue, cudaStream_t stream)
{
    if (m_maxBatchSize <= 0)
    {
        LOG_ERROR("Operator warp perspective var shape is not initialized properly, maxVarShapeBatchSize: "
                  << m_maxBatchSize);
        return ErrorCode::INVALID_PARAMETER;
    }

    if (m_maxBatchSize < inData.numImages())
    {
        LOG_ERROR("Invalid number of images, it should not exceed " << m_maxBatchSize);
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    if (!inData.uniqueFormat())
    {
        LOG_ERROR("Images in the input varshape must all have the same format");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    DataFormat input_format  = helpers::GetLegacyDataFormat(inData);
    DataFormat output_format = helpers::GetLegacyDataFormat(outData);

    if (input_format != output_format)
    {
        LOG_ERROR("Invalid DataFormat between input (" << input_format << ") and output (" << output_format << ")");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    DataFormat format = input_format;

    if (!(format == kNHWC || format == kHWC || format == kNCHW || format == kCHW))
    {
        LOG_ERROR("Invalid input DataFormat " << format
                                              << ", the valid DataFormats are: \"NHWC\", \"HWC\", \"NCHW\", \"CHW\"");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    const bool isPlanar = (format == kNCHW || format == kCHW);

    int channels = inData.uniqueFormat().numChannels();

    if (channels > 4 || channels == 2)
    {
        LOG_ERROR("Invalid channel number " << channels);
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    const int interpolation = flags & NVCV_INTERP_MAX;

    DataType data_type = helpers::GetLegacyDataType(inData.uniqueFormat());

    if (!(data_type == kCV_8U || data_type == kCV_16U || data_type == kCV_16S || data_type == kCV_32F))
    {
        LOG_ERROR("Invalid DataType " << data_type);
        return ErrorCode::INVALID_DATA_TYPE;
    }

    if (!(interpolation == NVCV_INTERP_NEAREST || interpolation == NVCV_INTERP_LINEAR
          || interpolation == NVCV_INTERP_CUBIC))
    {
        LOG_ERROR("Invalid interpolation " << interpolation);
        return ErrorCode::INVALID_PARAMETER;
    }
    if (!(borderMode == NVCV_BORDER_CONSTANT || borderMode == NVCV_BORDER_REPLICATE || borderMode == NVCV_BORDER_REFLECT
          || borderMode == NVCV_BORDER_WRAP || borderMode == NVCV_BORDER_REFLECT101))
    {
        LOG_ERROR("Invalid borderMode " << borderMode);
        return ErrorCode::INVALID_PARAMETER;
    }

    // Check if inverse op is needed
    bool performInverse = flags & NVCV_WARP_INVERSE_MAP;

    // Wrap the matrix in 2D wrappers with proper pitch
    cuda::Tensor2DWrap<float, int32_t> transMatrixInput(transMatrix);
    cuda::Tensor2DWrap<float, int32_t> transMatrixOutput(m_transformationMatrix, static_cast<int>(sizeof(float) * 9));

    if (!performInverse)
    {
        inverseMatWarpPerspective<<<1, inData.numImages(), 0, stream>>>(inData.numImages(), transMatrixInput,
                                                                        transMatrixOutput);
        checkKernelErrors();
    }
    else
    {
        NVCV_CHECK_LOG(cudaMemcpy2DAsync(m_transformationMatrix, sizeof(float) * 9, transMatrixInput.ptr(0, 0),
                                         transMatrixInput.strides()[0], sizeof(float) * 9, inData.numImages(),
                                         cudaMemcpyDeviceToDevice, stream));
    }

    typedef void (*func_t)(const ImageBatchVarShapeDataStridedCuda &inData,
                           const ImageBatchVarShapeDataStridedCuda &outData,
                           cuda::Tensor2DWrap<float, int32_t> transform, const int interpolation, const int borderMode,
                           const float4 &borderValue, cudaStream_t stream);

    static const func_t funcs[6][4] = {
        {     warpPerspective<uchar1>,  0 /*warpPerspective<uchar2>*/,      warpPerspective<uchar3>,warpPerspective<uchar4>                                                                                                    },
        {0 /*warpPerspective<schar>*/,   0 /*warpPerspective<char2>*/, 0 /*warpPerspective<char3>*/,
         0 /*warpPerspective<char4>*/                                                                                        },
        {    warpPerspective<ushort1>, 0 /*warpPerspective<ushort2>*/,     warpPerspective<ushort3>, warpPerspective<ushort4>},
        {     warpPerspective<short1>,  0 /*warpPerspective<short2>*/,      warpPerspective<short3>,  warpPerspective<short4>},
        {  0 /*warpPerspective<int>*/,    0 /*warpPerspective<int2>*/,  0 /*warpPerspective<int3>*/,
         0 /*warpPerspective<int4>*/                                                                                         },
        {     warpPerspective<float1>,  0 /*warpPerspective<float2>*/,      warpPerspective<float3>,  warpPerspective<float4>}
    };

    if (isPlanar)
    {
        // Planar dispatch indexes by dtype only: each channel is warped as a single-channel plane
        // (per-channel border value handled inside the planar caller). Matches WarpAffineVarShape.
        typedef void (*planar_func_t)(
            const ImageBatchVarShapeDataStridedCuda &inData, const ImageBatchVarShapeDataStridedCuda &outData,
            cuda::Tensor2DWrap<float, int32_t> transform, const int interpolation, const int borderMode, int channels,
            const float4 &borderValue, cudaStream_t stream);

        static const planar_func_t planar_funcs[6] = {
            warpPerspective_planar<uchar1>, 0 /*schar*/, warpPerspective_planar<ushort1>,
            warpPerspective_planar<short1>, 0 /*int*/,   warpPerspective_planar<float1>,
        };

        const planar_func_t planarFunc = planar_funcs[data_type];
        NVCV_ASSERT(planarFunc != 0);
        planarFunc(inData, outData, transMatrixOutput, interpolation, borderMode, channels, borderValue, stream);
        return SUCCESS;
    }

    const func_t func = funcs[data_type][channels - 1];
    NVCV_ASSERT(func != 0);

    func(inData, outData, transMatrixOutput, interpolation, borderMode, borderValue, stream);
    return SUCCESS;
}

} // namespace nvcv::legacy::cuda_op
