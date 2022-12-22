/* Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#define BLOCK 32

using namespace nvcv::legacy::cuda_op;
using namespace nvcv::legacy::helpers;
using namespace nvcv::cuda;

namespace nvcv::legacy::cuda_op {

__global__ void inverseMatWarpPerspective(const int numImages, const cuda::Tensor2DWrap<float> in,
                                          cuda::Tensor2DWrap<float> out)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= numImages)
    {
        return;
    }

    cuda::math::Matrix<float, 3, 3> transMatrix;
    transMatrix[0][0] = (float)(*in.ptr(index, 0));
    transMatrix[0][1] = (float)(*in.ptr(index, 1));
    transMatrix[0][2] = (float)(*in.ptr(index, 2));
    transMatrix[1][0] = (float)(*in.ptr(index, 3));
    transMatrix[1][1] = (float)(*in.ptr(index, 4));
    transMatrix[1][2] = (float)(*in.ptr(index, 5));
    transMatrix[2][0] = (float)(*in.ptr(index, 6));
    transMatrix[2][1] = (float)(*in.ptr(index, 7));
    transMatrix[2][2] = (float)(*in.ptr(index, 8));

    cuda::math::inv_inplace(transMatrix);

    *out.ptr(index, 0) = transMatrix[0][0];
    *out.ptr(index, 1) = transMatrix[0][1];
    *out.ptr(index, 2) = transMatrix[0][2];
    *out.ptr(index, 3) = transMatrix[1][0];
    *out.ptr(index, 4) = transMatrix[1][1];
    *out.ptr(index, 5) = transMatrix[1][2];
    *out.ptr(index, 6) = transMatrix[2][0];
    *out.ptr(index, 7) = transMatrix[2][1];
    *out.ptr(index, 8) = transMatrix[2][2];
}

__global__ void inverseMatWarpAffine(const int numImages, const cuda::Tensor2DWrap<float> in,
                                     cuda::Tensor2DWrap<float> out)
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
    *out.ptr(index, 0) = (float)M[1][2] * den;
    *out.ptr(index, 1) = (float)-M[0][1] * den;
    *out.ptr(index, 2) = (float)(M[0][1] * M[1][2] - M[1][1] * M[0][2]) * den;
    *out.ptr(index, 3) = (float)-M[1][0] * den;
    *out.ptr(index, 4) = (float)M[0][0] * den;
    *out.ptr(index, 5) = (float)(M[1][0] * M[0][2] - M[0][0] * M[1][2]) * den;
}

template<class Transform, class Filter, typename T>
__global__ void warp(const Filter src, Ptr2dVarShapeNHWC<T> dst, const cuda::Tensor2DWrap<float> d_coeffs_)
{
    const int x         = blockDim.x * blockIdx.x + threadIdx.x;
    const int y         = blockDim.y * blockIdx.y + threadIdx.y;
    const int batch_idx = get_batch_idx();
    const int lid       = get_lid();

    extern __shared__ float coeff[];
    if (lid < 9)
    {
        coeff[lid] = *d_coeffs_.ptr(batch_idx, lid);
    }
    __syncthreads();

    if (x < dst.at_cols(batch_idx) && y < dst.at_rows(batch_idx))
    {
        const float2 coord        = Transform::calcCoord(coeff, x, y);
        *dst.ptr(batch_idx, y, x) = nvcv::cuda::SaturateCast<nvcv::cuda::BaseType<T>>(src(batch_idx, coord.y, coord.x));
    }
}

template<class Transform, template<typename> class Filter, template<typename> class B, typename T>
struct WarpDispatcher
{
    static void call(const Ptr2dVarShapeNHWC<T> src, Ptr2dVarShapeNHWC<T> dst, const cuda::Tensor2DWrap<float> d_coeffs,
                     const int max_height, const int max_width, const float4 borderValue, cudaStream_t stream)
    {
        using work_type = nvcv::cuda::ConvertBaseTypeTo<float, T>;

        dim3 block(BLOCK, BLOCK / 4);
        dim3 grid(divUp(max_width, block.x), divUp(max_height, block.y), dst.batches);

        work_type    borderVal = nvcv::cuda::DropCast<NumComponents<T>>(borderValue);
        B<work_type> brd(0, 0, borderVal);
        // B<work_type> brd(max_height, max_width, borderVal);
        BorderReader<Ptr2dVarShapeNHWC<T>, B<work_type>>         brdSrc(src, brd);
        Filter<BorderReader<Ptr2dVarShapeNHWC<T>, B<work_type>>> filter_src(brdSrc);
        size_t                                                   smem_size = 9 * sizeof(float);
        warp<Transform><<<grid, block, smem_size, stream>>>(filter_src, dst, d_coeffs);
        checkKernelErrors();
    }
};

template<class Transform, typename T>
void warp_caller(const Ptr2dVarShapeNHWC<T> src, Ptr2dVarShapeNHWC<T> dst, const cuda::Tensor2DWrap<float> transform,
                 const int max_height, const int max_width, const int interpolation, const int borderMode,
                 const float4 borderValue, cudaStream_t stream)
{
    typedef void (*func_t)(const Ptr2dVarShapeNHWC<T> src, Ptr2dVarShapeNHWC<T> dst,
                           const cuda::Tensor2DWrap<float> transform, const int max_height, const int max_width,
                           const float4 borderValue, cudaStream_t stream);

    static const func_t funcs[3][5] = {
        {WarpDispatcher<Transform,  PointFilter, BrdConstant, T>::call,
         WarpDispatcher<Transform,  PointFilter, BrdReplicate, T>::call,
         WarpDispatcher<Transform,  PointFilter, BrdReflect, T>::call,
         WarpDispatcher<Transform,  PointFilter, BrdWrap, T>::call,
         WarpDispatcher<Transform,  PointFilter, BrdReflect101, T>::call},
        {WarpDispatcher<Transform, LinearFilter, BrdConstant, T>::call,
         WarpDispatcher<Transform, LinearFilter, BrdReplicate, T>::call,
         WarpDispatcher<Transform, LinearFilter, BrdReflect, T>::call,
         WarpDispatcher<Transform, LinearFilter, BrdWrap, T>::call,
         WarpDispatcher<Transform, LinearFilter, BrdReflect101, T>::call},
        {WarpDispatcher<Transform,  CubicFilter, BrdConstant, T>::call,
         WarpDispatcher<Transform,  CubicFilter, BrdReplicate, T>::call,
         WarpDispatcher<Transform,  CubicFilter, BrdReflect, T>::call,
         WarpDispatcher<Transform,  CubicFilter, BrdWrap, T>::call,
         WarpDispatcher<Transform,  CubicFilter, BrdReflect101, T>::call}
    };

    funcs[interpolation][borderMode](src, dst, transform, max_height, max_width, borderValue, stream);
}

template<typename T>
void warpAffine(const nvcv::IImageBatchVarShapeDataStridedCuda &inData,
                const nvcv::IImageBatchVarShapeDataStridedCuda &outData, const cuda::Tensor2DWrap<float> transform,
                const int interpolation, const int borderMode, const float4 borderValue, cudaStream_t stream)
{
    cuda_op::Ptr2dVarShapeNHWC<T> src_ptr(inData);
    cuda_op::Ptr2dVarShapeNHWC<T> dst_ptr(outData);

    Size2D outMaxSize = outData.maxSize();

    warp_caller<WarpAffineTransform, T>(src_ptr, dst_ptr, transform, outMaxSize.h, outMaxSize.w, interpolation,
                                        borderMode, borderValue, stream);
}

template<typename T>
void warpPerspective(const nvcv::IImageBatchVarShapeDataStridedCuda &inData,
                     const nvcv::IImageBatchVarShapeDataStridedCuda &outData, const cuda::Tensor2DWrap<float> transform,
                     const int interpolation, const int borderMode, const float4 borderValue, cudaStream_t stream)
{
    Ptr2dVarShapeNHWC<T> src_ptr(inData);
    Ptr2dVarShapeNHWC<T> dst_ptr(outData);

    Size2D outMaxSize = outData.maxSize();

    warp_caller<PerspectiveTransform, T>(src_ptr, dst_ptr, transform, outMaxSize.h, outMaxSize.w, interpolation,
                                         borderMode, borderValue, stream);
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

ErrorCode WarpAffineVarShape::infer(const IImageBatchVarShapeDataStridedCuda &inData,
                                    const IImageBatchVarShapeDataStridedCuda &outData,
                                    const ITensorDataStridedCuda &transMatrix, const int32_t flags,
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

    DataFormat input_format  = helpers::GetLegacyDataFormat(inData);
    DataFormat output_format = helpers::GetLegacyDataFormat(outData);

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

    if (!inData.uniqueFormat())
    {
        LOG_ERROR("Images in the input varshape must all have the same format");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    int channels = inData.uniqueFormat().numChannels();

    if (channels > 4)
    {
        LOG_ERROR("Invalid channel number " << channels);
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    const int interpolation = flags & NVCV_INTERP_MAX;

    DataType data_type = helpers::GetLegacyDataType(inData.uniqueFormat());

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

    // Check if inverse op is needed
    bool performInverse = flags & NVCV_WARP_INVERSE_MAP;

    // Wrap the matrix in 2D wrappers with proper pitch
    cuda::Tensor2DWrap<float> transMatrixInput(transMatrix);
    cuda::Tensor2DWrap<float> transMatrixOutput(m_transformationMatrix, static_cast<int>(sizeof(float) * 9));

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

    typedef void (*func_t)(const nvcv::IImageBatchVarShapeDataStridedCuda &inData,
                           const nvcv::IImageBatchVarShapeDataStridedCuda &outData,
                           const cuda::Tensor2DWrap<float> transform, const int interpolation, const int borderMode,
                           const float4 borderValue, cudaStream_t stream);

    static const func_t funcs[6][4] = {
        {      warpAffine<uchar>,  0 /*warpAffine<uchar2>*/,      warpAffine<uchar3>,      warpAffine<uchar4>},
        {0 /*warpAffine<schar>*/,   0 /*warpAffine<char2>*/, 0 /*warpAffine<char3>*/, 0 /*warpAffine<char4>*/},
        {     warpAffine<ushort>, 0 /*warpAffine<ushort2>*/,     warpAffine<ushort3>,     warpAffine<ushort4>},
        {      warpAffine<short>,  0 /*warpAffine<short2>*/,      warpAffine<short3>,      warpAffine<short4>},
        {  0 /*warpAffine<int>*/,    0 /*warpAffine<int2>*/,  0 /*warpAffine<int3>*/,  0 /*warpAffine<int4>*/},
        {      warpAffine<float>,  0 /*warpAffine<float2>*/,      warpAffine<float3>,      warpAffine<float4>}
    };

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

ErrorCode WarpPerspectiveVarShape::infer(const IImageBatchVarShapeDataStridedCuda &inData,
                                         const IImageBatchVarShapeDataStridedCuda &outData,
                                         const ITensorDataStridedCuda &transMatrix, const int32_t flags,
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

    DataFormat input_format  = helpers::GetLegacyDataFormat(inData);
    DataFormat output_format = helpers::GetLegacyDataFormat(outData);

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

    if (!inData.uniqueFormat())
    {
        LOG_ERROR("Images in the input varshape must all have the same format");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    int channels = inData.uniqueFormat().numChannels();

    if (channels > 4)
    {
        LOG_ERROR("Invalid channel number " << channels);
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    const int interpolation = flags & NVCV_INTERP_MAX;

    DataType data_type = helpers::GetLegacyDataType(inData.uniqueFormat());

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

    // Check if inverse op is needed
    bool performInverse = flags & NVCV_WARP_INVERSE_MAP;

    // Wrap the matrix in 2D wrappers with proper pitch
    cuda::Tensor2DWrap<float> transMatrixInput(transMatrix);
    cuda::Tensor2DWrap<float> transMatrixOutput(m_transformationMatrix, static_cast<int>(sizeof(float) * 9));

    if (performInverse)
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

    typedef void (*func_t)(const nvcv::IImageBatchVarShapeDataStridedCuda &inData,
                           const nvcv::IImageBatchVarShapeDataStridedCuda &outData,
                           const cuda::Tensor2DWrap<float> transform, const int interpolation, const int borderMode,
                           const float4 borderValue, cudaStream_t stream);

    static const func_t funcs[6][4] = {
        {      warpPerspective<uchar>,  0 /*warpPerspective<uchar2>*/,      warpPerspective<uchar3>,warpPerspective<uchar4>                                                                                                    },
        {0 /*warpPerspective<schar>*/,   0 /*warpPerspective<char2>*/, 0 /*warpPerspective<char3>*/,
         0 /*warpPerspective<char4>*/                                                                                        },
        {     warpPerspective<ushort>, 0 /*warpPerspective<ushort2>*/,     warpPerspective<ushort3>, warpPerspective<ushort4>},
        {      warpPerspective<short>,  0 /*warpPerspective<short2>*/,      warpPerspective<short3>,  warpPerspective<short4>},
        {  0 /*warpPerspective<int>*/,    0 /*warpPerspective<int2>*/,  0 /*warpPerspective<int3>*/,
         0 /*warpPerspective<int4>*/                                                                                         },
        {      warpPerspective<float>,  0 /*warpPerspective<float2>*/,      warpPerspective<float3>,  warpPerspective<float4>}
    };

    const func_t func = funcs[data_type][channels - 1];
    NVCV_ASSERT(func != 0);

    func(inData, outData, transMatrixOutput, interpolation, borderMode, borderValue, stream);
    return SUCCESS;
}

} // namespace nvcv::legacy::cuda_op
