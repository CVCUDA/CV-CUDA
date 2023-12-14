/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#define BLOCK 32

namespace nvcv::legacy::cuda_op {

template<class Transform, class SrcWrapper, class DstWrapper, typename T = typename DstWrapper::ValueType>
__global__ void warp(SrcWrapper src, DstWrapper dst, int2 dstSize, Transform transform)
{
    int3      dstCoord = cuda::StaticCast<int>(blockDim * blockIdx + threadIdx);
    const int lid      = threadIdx.y * blockDim.x + threadIdx.x;

    extern __shared__ float coeff[];

    if (lid < 9)
    {
        coeff[lid] = transform.xform[lid];
    }

    __syncthreads();

    if (dstCoord.x < dstSize.x && dstCoord.y < dstSize.y)
    {
        const float2 coord = Transform::calcCoord(coeff, dstCoord.x, dstCoord.y);
        const float3 srcCoord{coord.x, coord.y, static_cast<float>(dstCoord.z)};

        dst[dstCoord] = src[srcCoord];
    }
}

template<class Transform, typename T, NVCVBorderType B, NVCVInterpolationType I>
struct WarpDispatcher
{
    static void call(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData, Transform transform,
                     const float4 &borderValue, cudaStream_t stream)
    {
        auto outAccess = TensorDataAccessStridedImagePlanar::Create(outData);
        NVCV_ASSERT(outAccess);

        const int2 dstSize{outAccess->numCols(), outAccess->numRows()};
        const int  batchSize{static_cast<int>(outAccess->numSamples())};

        dim3 block(BLOCK, BLOCK / 4);
        dim3 grid(divUp(dstSize.x, block.x), divUp(dstSize.y, block.y), batchSize);

        auto bVal = cuda::StaticCast<cuda::BaseType<T>>(cuda::DropCast<cuda::NumElements<T>>(borderValue));

        auto src = cuda::CreateInterpolationWrapNHW<const T, B, I>(inData, bVal);
        auto dst = cuda::CreateTensorWrapNHW<T>(outData);

        int smem_size = 9 * sizeof(float);

        warp<Transform><<<grid, block, smem_size, stream>>>(src, dst, dstSize, transform);
        checkKernelErrors();
    }
};

template<class Transform, typename T>
void warp_caller(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData, Transform transform,
                 int interpolation, int borderMode, const float4 &borderValue, cudaStream_t stream)
{
    typedef void (*func_t)(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData,
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

    funcs[interpolation][borderMode](inData, outData, transform, borderValue, stream);
}

template<typename T>
void warpAffine(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData,
                WarpAffineTransform transform, const int interpolation, int borderMode, const float4 &borderValue,
                cudaStream_t stream)
{
    warp_caller<WarpAffineTransform, T>(inData, outData, transform, interpolation, borderMode, borderValue, stream);
}

template<typename T>
void warpPerspective(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData,
                     PerspectiveTransform transform, const int interpolation, int borderMode, const float4 &borderValue,
                     cudaStream_t stream)
{
    warp_caller<PerspectiveTransform, T>(inData, outData, transform, interpolation, borderMode, borderValue, stream);
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

    if (!(format == kNHWC || format == kHWC))
    {
        LOG_ERROR("Invalid DataFormat " << format);
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    auto inAccess = TensorDataAccessStridedImagePlanar::Create(inData);
    NVCV_ASSERT(inAccess);

    DataType  data_type   = helpers::GetLegacyDataType(inData.dtype());
    DataShape input_shape = helpers::GetLegacyDataShape(inAccess->infoShape());

    int       channels      = input_shape.C;
    const int interpolation = flags & NVCV_INTERP_MAX;

    if (channels > 4)
    {
        LOG_ERROR("Invalid channel number " << channels);
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    if (!(data_type == kCV_8U || data_type == kCV_8S || data_type == kCV_16U || data_type == kCV_16S
          || data_type == kCV_32S || data_type == kCV_32F))
    {
        LOG_ERROR("Invalid DataType " << data_type);
        return ErrorCode::INVALID_DATA_TYPE;
    }

    NVCV_ASSERT(interpolation == NVCV_INTERP_NEAREST || interpolation == NVCV_INTERP_LINEAR
                || interpolation == NVCV_INTERP_CUBIC);
    NVCV_ASSERT(borderMode == NVCV_BORDER_REFLECT101 || borderMode == NVCV_BORDER_REPLICATE
                || borderMode == NVCV_BORDER_CONSTANT || borderMode == NVCV_BORDER_REFLECT
                || borderMode == NVCV_BORDER_WRAP);

    typedef void (*func_t)(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData,
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

    const func_t func = funcs[data_type][channels - 1];
    NVCV_ASSERT(func != 0);

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

    func(inData, outData, transform, interpolation, borderMode, borderValue, stream);

    return ErrorCode::SUCCESS;
}

size_t WarpPerspective::calBufferSize(DataShape max_input_shape, DataShape max_output_shape, DataType max_data_type)
{
    return 9 * sizeof(float);
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

    if (!(format == kNHWC || format == kHWC))
    {
        LOG_ERROR("Invalid DataFormat " << format);
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    auto inAccess = TensorDataAccessStridedImagePlanar::Create(inData);
    NVCV_ASSERT(inAccess);

    DataType  data_type   = helpers::GetLegacyDataType(inData.dtype());
    DataShape input_shape = helpers::GetLegacyDataShape(inAccess->infoShape());

    int       channels      = input_shape.C;
    const int interpolation = flags & NVCV_INTERP_MAX;

    if (channels > 4)
    {
        LOG_ERROR("Invalid channel number " << channels);
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    if (!(data_type == kCV_8U || data_type == kCV_8S || data_type == kCV_16U || data_type == kCV_16S
          || data_type == kCV_32S || data_type == kCV_32F))
    {
        LOG_ERROR("Invalid DataType " << data_type);
        return ErrorCode::INVALID_DATA_TYPE;
    }

    NVCV_ASSERT(interpolation == NVCV_INTERP_NEAREST || interpolation == NVCV_INTERP_LINEAR
                || interpolation == NVCV_INTERP_CUBIC);
    NVCV_ASSERT(borderMode == NVCV_BORDER_REFLECT101 || borderMode == NVCV_BORDER_REPLICATE
                || borderMode == NVCV_BORDER_CONSTANT || borderMode == NVCV_BORDER_REFLECT
                || borderMode == NVCV_BORDER_WRAP);

    typedef void (*func_t)(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData,
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

    const func_t func = funcs[data_type][channels - 1];
    NVCV_ASSERT(func != 0);

    PerspectiveTransform transform(transMatrix);

    if (flags & NVCV_WARP_INVERSE_MAP)
    {
        cuda::math::Matrix<float, 3, 3> tempMatrixForInverse;

        tempMatrixForInverse.load(transMatrix);

        cuda::math::inv_inplace(tempMatrixForInverse);

        tempMatrixForInverse.store(transform.xform);
    }

    func(inData, outData, transform, interpolation, borderMode, borderValue, stream);

    return ErrorCode::SUCCESS;
}

} // namespace nvcv::legacy::cuda_op
