/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "CvCudaLegacy.h"
#include "CvCudaLegacyHelpers.hpp"

#include "CvCudaUtils.cuh"
#include "warp_cubic.cuh"

#include <type_traits>

#define BLOCK 32

namespace nvcv::legacy::cuda_op {

template<bool UseSharedTransform, class Transform, class SrcWrapper, class DstWrapper>
__global__ void warp(SrcWrapper src, DstWrapper dst, int2 dstSize, int2 srcSize, Transform transform)
{
    using SrcValueT       = std::remove_cv_t<typename SrcWrapper::ValueType>;
    constexpr bool kCubic = SrcWrapper::kInterpolationType == NVCV_INTERP_CUBIC;
    // The restructured sampler pays off for affine maps (the scatter-heavy case); perspective
    // maps keep the wrap path, whose per-tap loads coalesce well and carry no extra registers.
    constexpr bool kFast = kCubic && kCubicFastSampler<SrcValueT> && std::is_same_v<Transform, WarpAffineTransform>;

    int3 dstCoord = cuda::StaticCast<int>(blockDim * blockIdx + threadIdx);

    if constexpr (UseSharedTransform)
    {
        extern __shared__ float coeff[];

        const int lid = threadIdx.y * blockDim.x + threadIdx.x;
        if (lid < 9)
        {
            coeff[lid] = transform.xform[lid];
        }

        __syncthreads();

        if (dstCoord.x < dstSize.x && dstCoord.y < dstSize.y)
        {
            const float2 coord = Transform::calcCoord(coeff, dstCoord.x, dstCoord.y);

            if constexpr (kFast)
            {
                dst[dstCoord] = CubicSampleTensor(src, dstCoord.z, coord, srcSize);
            }
            else
            {
                dst[dstCoord] = src[float3{coord.x, coord.y, static_cast<float>(dstCoord.z)}];
            }
        }
    }
    else if (dstCoord.x < dstSize.x && dstCoord.y < dstSize.y)
    {
        const float2 coord = Transform::calcCoord(transform.xform, dstCoord.x, dstCoord.y);

        if constexpr (kFast)
        {
            dst[dstCoord] = CubicSampleTensor(src, dstCoord.z, coord, srcSize);
        }
        else
        {
            dst[dstCoord] = src[float3{coord.x, coord.y, static_cast<float>(dstCoord.z)}];
        }
    }
}

// Fused planar (NCHW/CHW) CUBIC warp: one launch covers every channel plane, sharing the
// transformed coordinate, cubic weights, and border resolution across planes instead of
// relaunching the single-channel kernel per plane. src/dst wrap plane 0; the other planes are
// addressed by the plane byte stride.
template<bool UseSharedTransform, class Transform, typename BT, NVCVBorderType B, int NP, class SrcWrapper,
         class DstWrapper>
__global__ void warp_planar_fused(SrcWrapper src, DstWrapper dst, int2 dstSize, int2 srcSize, int64_t srcPlaneStride,
                                  int64_t dstPlaneStride, float4 borderValue, Transform transform)
{
    const int3 dstCoord = cuda::StaticCast<int>(blockDim * blockIdx + threadIdx);

    const float *xform = transform.xform;
    if constexpr (UseSharedTransform)
    {
        extern __shared__ float coeff[];

        const int lid = threadIdx.y * blockDim.x + threadIdx.x;
        if (lid < 9)
        {
            coeff[lid] = transform.xform[lid];
        }

        __syncthreads();
        xform = coeff;
    }

    if (dstCoord.x < dstSize.x && dstCoord.y < dstSize.y)
    {
        const float2 coord = Transform::calcCoord(xform, dstCoord.x, dstCoord.y);

        CubicWarpPlanes<BT, B, NP, int32_t>(
            [&](int p, int yy) {
                return reinterpret_cast<const BT *>(reinterpret_cast<const char *>(src.ptr(dstCoord.z, yy))
                                                    + p * srcPlaneStride);
            },
            [&](int p, BT v)
            {
                *(reinterpret_cast<BT *>(reinterpret_cast<char *>(dst.ptr(dstCoord.z, dstCoord.y)) + p * dstPlaneStride)
                  + dstCoord.x)
                    = v;
            },
            borderValue, srcSize, coord);
    }
}

template<class Transform, typename T, NVCVBorderType B, NVCVInterpolationType I>
struct WarpDispatcher
{
    static ErrorCode call(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData,
                          Transform transform, const float4 &borderValue, cudaStream_t stream)
    {
        auto outAccess = TensorDataAccessStridedImagePlanar::Create(outData);
        NVCV_ASSERT(outAccess);

        auto inAccess = TensorDataAccessStridedImagePlanar::Create(inData);
        NVCV_ASSERT(inAccess);

        const int2 dstSize{outAccess->numCols(), outAccess->numRows()};
        const int  batchSize{static_cast<int>(outAccess->numSamples())};

        dim3 block(BLOCK, BLOCK / 4);
        dim3 grid(divUp(dstSize.x, block.x), divUp(dstSize.y, block.y), batchSize);

        auto bVal = cuda::StaticCast<cuda::BaseType<T>>(cuda::DropCast<cuda::NumElements<T>>(borderValue));

        int64_t srcMaxStride = inAccess->sampleStride() * batchSize;
        int64_t dstMaxStride = outAccess->sampleStride() * batchSize;

        if (std::max(srcMaxStride, dstMaxStride) <= cuda::TypeTraits<int32_t>::max)
        {
            auto src = cuda::CreateInterpolationWrapNHW<const T, B, I, int32_t>(inData, bVal);
            auto dst = cuda::CreateTensorWrapNHW<T, int32_t>(outData);

            const int2 srcSize{inAccess->numCols(), inAccess->numRows()};

            // PerspectiveTransform is already passed by value, so only WarpAffine's measured
            // specializations need the shared-memory copy.
            constexpr bool useSharedTransform
                = std::is_same_v<Transform,
                                 WarpAffineTransform> && (cuda::NumElements<T> == 3 || I == NVCV_INTERP_CUBIC);
            constexpr int smemSize = useSharedTransform ? 9 * sizeof(float) : 0;
            warp<useSharedTransform, Transform>
                <<<grid, block, smemSize, stream>>>(src, dst, dstSize, srcSize, transform);
        }
        else
        {
            LOG_ERROR("Input or output size exceeds " << cuda::TypeTraits<int32_t>::max << ". Tensor is too large.");
            return ErrorCode::INVALID_PARAMETER;
        }
        checkKernelErrors();
        return ErrorCode::SUCCESS;
    }
};

template<class Transform, typename T>
ErrorCode warp_caller(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData, Transform transform,
                      int interpolation, int borderMode, const float4 &borderValue, cudaStream_t stream)
{
    typedef ErrorCode (*func_t)(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData,
                                Transform transform, const float4 &borderValue, cudaStream_t stream);

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

    return funcs[interpolation][borderMode](inData, outData, transform, borderValue, stream);
}

template<typename T>
ErrorCode warpAffine(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData,
                     WarpAffineTransform transform, const int interpolation, int borderMode, const float4 &borderValue,
                     cudaStream_t stream)
{
    return warp_caller<WarpAffineTransform, T>(inData, outData, transform, interpolation, borderMode, borderValue,
                                               stream);
}

template<typename T>
ErrorCode warpPerspective(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData,
                          PerspectiveTransform transform, const int interpolation, int borderMode,
                          const float4 &borderValue, cudaStream_t stream)
{
    return warp_caller<PerspectiveTransform, T>(inData, outData, transform, interpolation, borderMode, borderValue,
                                                stream);
}

static void invertMat(const float *M, float *h_aCoeffs)
{
    // M is stored in row-major format M[0,0], M[0,1], M[0,2], M[1,0], M[1,1], M[1,2]
    float den    = M[0] * M[4] - M[1] * M[3];
    den          = std::abs(den) > 1e-5 ? 1. / den : .0;
    h_aCoeffs[0] = (float)M[4] * den;
    h_aCoeffs[1] = (float)-M[1] * den;
    h_aCoeffs[2] = (float)(M[1] * M[5] - M[4] * M[2]) * den;
    h_aCoeffs[3] = (float)-M[3] * den;
    h_aCoeffs[4] = (float)M[0] * den;
    h_aCoeffs[5] = (float)(M[3] * M[2] - M[0] * M[5]) * den;
}

// Build a single-channel (N, H, W, 1) NHWC view of channel plane `plane` of a packed planar
// (NCHW/CHW) tensor. Warp samples each channel at the same transformed coordinate, so plane (n, c)
// is an independent single-channel image: viewing one plane across all N samples (sample stride
// unchanged, base offset by plane*chStride) lets the existing single-channel warp kernel handle it,
// bit-exact with the equivalent NHWC single-channel warp. One such view is processed per plane.
static nvcv::TensorDataStridedCuda PlanarChannelView(const nvcv::TensorDataStridedCuda              &data,
                                                     const nvcv::TensorDataAccessStridedImagePlanar &access, int plane)
{
    nvcv::TensorDataStridedCuda::Buffer buf;
    buf.basePtr    = reinterpret_cast<NVCVByte *>(data.basePtr()) + plane * access.chStride();
    buf.strides[0] = access.sampleStride(); // N
    buf.strides[1] = access.rowStride();    // H
    buf.strides[2] = access.colStride();    // W
    buf.strides[3] = access.colStride();    // C == 1
    return nvcv::TensorDataStridedCuda{
        nvcv::TensorShape{{access.numSamples(), access.numRows(), access.numCols(), 1}, "NHWC"},
        data.dtype(), buf
    };
}

// Select the constant-border component for a given channel plane. The interleaved path fills the
// border with (x, y, z, w) per channel; a single-channel plane uses only component `plane` (the
// single-channel kernel reads borderValue.x), so it is replicated into .x.
static float4 PlanarBorderValue(const float4 &borderValue, int plane)
{
    const float c = plane == 0 ? borderValue.x
                  : plane == 1 ? borderValue.y
                  : plane == 2 ? borderValue.z
                               : borderValue.w;
    return float4{c, c, c, c};
}

template<class Transform, typename BT, NVCVBorderType B>
struct WarpPlanarFusedDispatcher
{
    static ErrorCode call(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData,
                          Transform transform, const float4 &borderValue, int numPlanes, cudaStream_t stream)
    {
        auto inAccess = TensorDataAccessStridedImagePlanar::Create(inData);
        NVCV_ASSERT(inAccess);
        auto outAccess = TensorDataAccessStridedImagePlanar::Create(outData);
        NVCV_ASSERT(outAccess);

        const int2 dstSize{outAccess->numCols(), outAccess->numRows()};
        const int2 srcSize{inAccess->numCols(), inAccess->numRows()};
        const int  batchSize{static_cast<int>(outAccess->numSamples())};

        dim3 block(BLOCK, BLOCK / 4);
        dim3 grid(divUp(dstSize.x, block.x), divUp(dstSize.y, block.y), batchSize);

        const int64_t srcMaxStride = inAccess->sampleStride() * batchSize;
        const int64_t dstMaxStride = outAccess->sampleStride() * batchSize;
        if (std::max(srcMaxStride, dstMaxStride) > cuda::TypeTraits<int32_t>::max)
        {
            LOG_ERROR("Input or output size exceeds " << cuda::TypeTraits<int32_t>::max << ". Tensor is too large.");
            return ErrorCode::INVALID_PARAMETER;
        }

        auto planeIn  = PlanarChannelView(inData, *inAccess, 0);
        auto planeOut = PlanarChannelView(outData, *outAccess, 0);

        auto src = cuda::CreateTensorWrapNHW<const BT, int32_t>(planeIn);
        auto dst = cuda::CreateTensorWrapNHW<BT, int32_t>(planeOut);

        constexpr bool useSharedTransform = std::is_same_v<Transform, WarpAffineTransform>;
        constexpr int  smemSize           = useSharedTransform ? 9 * sizeof(float) : 0;

        switch (numPlanes)
        {
        case 1:
            warp_planar_fused<useSharedTransform, Transform, BT, B, 1><<<grid, block, smemSize, stream>>>(
                src, dst, dstSize, srcSize, inAccess->chStride(), outAccess->chStride(), borderValue, transform);
            break;
        case 3:
            warp_planar_fused<useSharedTransform, Transform, BT, B, 3><<<grid, block, smemSize, stream>>>(
                src, dst, dstSize, srcSize, inAccess->chStride(), outAccess->chStride(), borderValue, transform);
            break;
        case 4:
            warp_planar_fused<useSharedTransform, Transform, BT, B, 4><<<grid, block, smemSize, stream>>>(
                src, dst, dstSize, srcSize, inAccess->chStride(), outAccess->chStride(), borderValue, transform);
            break;
        default:
            LOG_ERROR("Invalid planar channel number " << numPlanes);
            return ErrorCode::INVALID_DATA_SHAPE;
        }
        checkKernelErrors();
        return ErrorCode::SUCCESS;
    }
};

template<class Transform, typename BT>
ErrorCode warpPlanarFusedCubic(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData,
                               Transform transform, int borderMode, const float4 &borderValue, int numPlanes,
                               cudaStream_t stream)
{
    typedef ErrorCode (*func_t)(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData,
                                Transform transform, const float4 &borderValue, int numPlanes, cudaStream_t stream);

    static const func_t funcs[5] = {
        WarpPlanarFusedDispatcher<Transform, BT, NVCV_BORDER_CONSTANT>::call,
        WarpPlanarFusedDispatcher<Transform, BT, NVCV_BORDER_REPLICATE>::call,
        WarpPlanarFusedDispatcher<Transform, BT, NVCV_BORDER_REFLECT>::call,
        WarpPlanarFusedDispatcher<Transform, BT, NVCV_BORDER_WRAP>::call,
        WarpPlanarFusedDispatcher<Transform, BT, NVCV_BORDER_REFLECT101>::call,
    };

    return funcs[borderMode](inData, outData, transform, borderValue, numPlanes, stream);
}

template<class Transform>
ErrorCode warpPlanarFusedCubicCaller(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData,
                                     Transform transform, int dataType, int borderMode, const float4 &borderValue,
                                     int numPlanes, cudaStream_t stream)
{
    typedef ErrorCode (*func_t)(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData,
                                Transform transform, int borderMode, const float4 &borderValue, int numPlanes,
                                cudaStream_t stream);

    static const func_t funcs[6] = {warpPlanarFusedCubic<Transform, uchar>, 0, warpPlanarFusedCubic<Transform, ushort>,
                                    warpPlanarFusedCubic<Transform, short>, 0, warpPlanarFusedCubic<Transform, float>};

    const func_t func = funcs[dataType];
    NVCV_ASSERT(func != 0);

    return func(inData, outData, transform, borderMode, borderValue, numPlanes, stream);
}

ErrorCode WarpAffine::infer(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData,
                            const float *xform, const int32_t flags, const NVCVBorderType borderMode,
                            const float4 borderValue, cudaStream_t stream)
{
    DataFormat input_format  = helpers::GetLegacyDataFormat(inData.layout());
    DataFormat output_format = helpers::GetLegacyDataFormat(outData.layout());

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

    auto inAccess = TensorDataAccessStridedImagePlanar::Create(inData);
    NVCV_ASSERT(inAccess);

    DataType  data_type   = helpers::GetLegacyDataType(inData.dtype());
    DataShape input_shape = helpers::GetLegacyDataShape(inAccess->infoShape());

    int       channels      = input_shape.C;
    const int interpolation = flags & NVCV_INTERP_MAX;

    if (channels > 4 || channels == 2)
    {
        LOG_ERROR("Invalid channel number " << channels);
        return ErrorCode::INVALID_DATA_SHAPE;
    }

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

    typedef ErrorCode (*func_t)(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData,
                                WarpAffineTransform transform, const int interpolation, int borderMode,
                                const float4 &borderValue, cudaStream_t stream);

    static const func_t funcs[6][4] = {
        { warpAffine<uchar1>, 0,  warpAffine<uchar3>,  warpAffine<uchar4>},
        {                  0, 0,                   0,                   0},
        {warpAffine<ushort1>, 0, warpAffine<ushort3>, warpAffine<ushort4>},
        { warpAffine<short1>, 0,  warpAffine<short3>,  warpAffine<short4>},
        {                  0, 0,                   0,                   0},
        { warpAffine<float1>, 0,  warpAffine<float3>,  warpAffine<float4>}
    };

    WarpAffineTransform transform;

    if (flags & NVCV_WARP_INVERSE_MAP)
    {
        for (int i = 0; i < 9; i++)
        {
            transform.xform[i] = i < 6 ? (float)(xform[i]) : 0.0f;
        }
    }
    else
    {
        invertMat(xform, transform.xform);
    }

    if (isPlanar)
    {
        // Fused planar only pays for byte-based types; other dtypes keep per-plane launches
        // (each already using the fast CUBIC sampler).
        if (interpolation == NVCV_INTERP_CUBIC && data_type == kCV_8U)
        {
            return warpPlanarFusedCubicCaller(inData, outData, transform, data_type, borderMode, borderValue, channels,
                                              stream);
        }

        // Warp each channel plane as an independent single-channel image (same transform, per-channel
        // border value), reusing the interleaved single-channel kernel (funcs column 0). This is
        // bit-exact with warping the equivalent NHWC single-channel data. The planar layout carries a
        // per-channel border value, so each plane is given its own.
        const func_t planarFunc = funcs[data_type][0];
        NVCV_ASSERT(planarFunc != 0);

        auto outAccess = TensorDataAccessStridedImagePlanar::Create(outData);
        NVCV_ASSERT(outAccess);

        for (int c = 0; c < channels; ++c)
        {
            auto      planeIn  = PlanarChannelView(inData, *inAccess, c);
            auto      planeOut = PlanarChannelView(outData, *outAccess, c);
            ErrorCode ec       = planarFunc(planeIn, planeOut, transform, interpolation, borderMode,
                                            PlanarBorderValue(borderValue, c), stream);
            if (ec != ErrorCode::SUCCESS)
            {
                return ec;
            }
        }
        return ErrorCode::SUCCESS;
    }

    const func_t func = funcs[data_type][channels - 1];
    NVCV_ASSERT(func != 0);

    return func(inData, outData, transform, interpolation, borderMode, borderValue, stream);
}

ErrorCode WarpPerspective::infer(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData,
                                 const float *transMatrix, const int32_t flags, const NVCVBorderType borderMode,
                                 const float4 borderValue, cudaStream_t stream)
{
    DataFormat input_format  = helpers::GetLegacyDataFormat(inData.layout());
    DataFormat output_format = helpers::GetLegacyDataFormat(outData.layout());

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

    auto inAccess = TensorDataAccessStridedImagePlanar::Create(inData);
    NVCV_ASSERT(inAccess);

    DataType  data_type   = helpers::GetLegacyDataType(inData.dtype());
    DataShape input_shape = helpers::GetLegacyDataShape(inAccess->infoShape());

    int       channels      = input_shape.C;
    const int interpolation = flags & NVCV_INTERP_MAX;

    if (channels > 4 || channels == 2)
    {
        LOG_ERROR("Invalid channel number " << channels);
        return ErrorCode::INVALID_DATA_SHAPE;
    }

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

    typedef ErrorCode (*func_t)(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData,
                                PerspectiveTransform transform, const int interpolation, int borderMode,
                                const float4 &borderValue, cudaStream_t stream);

    static const func_t funcs[6][4] = {
        {      warpPerspective<uchar1>,  0 /*warpPerspective<uchar2>*/,      warpPerspective<uchar3>,warpPerspective<uchar4>                                                                                                     },
        {0 /*warpPerspective<schar1>*/,   0 /*warpPerspective<char2>*/, 0 /*warpPerspective<char3>*/,
         0 /*warpPerspective<char4>*/                                                                                         },
        {     warpPerspective<ushort1>, 0 /*warpPerspective<ushort2>*/,     warpPerspective<ushort3>, warpPerspective<ushort4>},
        {      warpPerspective<short1>,  0 /*warpPerspective<short2>*/,      warpPerspective<short3>,  warpPerspective<short4>},
        {  0 /*warpPerspective<int1>*/,    0 /*warpPerspective<int2>*/,  0 /*warpPerspective<int3>*/,
         0 /*warpPerspective<int4>*/                                                                                          },
        {      warpPerspective<float1>,  0 /*warpPerspective<float2>*/,      warpPerspective<float3>,  warpPerspective<float4>}
    };

    PerspectiveTransform transform(transMatrix);

    if (!(flags & NVCV_WARP_INVERSE_MAP))
    {
        cuda::math::Matrix<float, 3, 3> tempMatrixForInverse;

        tempMatrixForInverse.load(transMatrix);

        cuda::math::inv_inplace(tempMatrixForInverse);

        tempMatrixForInverse.store(transform.xform);
    }

    if (isPlanar)
    {
        // Fused planar only pays for byte-based types; other dtypes keep per-plane launches
        // (each already using the fast CUBIC sampler).
        if (interpolation == NVCV_INTERP_CUBIC && data_type == kCV_8U)
        {
            return warpPlanarFusedCubicCaller(inData, outData, transform, data_type, borderMode, borderValue, channels,
                                              stream);
        }

        // Warp each channel plane as an independent single-channel image (same transform, per-channel
        // border value), reusing the interleaved single-channel kernel (funcs column 0). This is
        // bit-exact with warping the equivalent NHWC single-channel data, and matches WarpAffine.
        const func_t planarFunc = funcs[data_type][0];
        NVCV_ASSERT(planarFunc != 0);

        auto outAccess = TensorDataAccessStridedImagePlanar::Create(outData);
        NVCV_ASSERT(outAccess);

        for (int c = 0; c < channels; ++c)
        {
            auto      planeIn  = PlanarChannelView(inData, *inAccess, c);
            auto      planeOut = PlanarChannelView(outData, *outAccess, c);
            ErrorCode ec       = planarFunc(planeIn, planeOut, transform, interpolation, borderMode,
                                            PlanarBorderValue(borderValue, c), stream);
            if (ec != ErrorCode::SUCCESS)
            {
                return ec;
            }
        }
        return ErrorCode::SUCCESS;
    }

    const func_t func = funcs[data_type][channels - 1];
    NVCV_ASSERT(func != 0);

    return func(inData, outData, transform, interpolation, borderMode, borderValue, stream);
}

} // namespace nvcv::legacy::cuda_op
