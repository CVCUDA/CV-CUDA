/* Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "CvCudaLegacy.h"
#include "CvCudaLegacyHelpers.hpp"

#include "CvCudaUtils.cuh"

namespace cuda = nvcv::cuda;
using namespace nvcv::legacy::helpers;

namespace nvcv::legacy::cuda_op {

#define BLOCK                      32
#define PI                         3.1415926535897932384626433832795
//tl and bt points coords + area + angle
#define _MIN_AREA_EACH_ANGLE_STRID 6
// max rotate degree of contour points
#define _MAX_ROTATE_DEGREES        90

void calculateRotateCoefCUDA(cuda::Tensor2DWrap<float> rotateCoefBuf, const int degrees, const cudaStream_t &stream);

template<typename T>
__global__ void resetRotatedPointsBuf(cuda::Tensor3DWrap<T> rotatedPointsTensor, const int numOfDegrees)
{
    // int pointIdx    = blockIdx.x * blockDim.x + threadIdx.x;
    int contourIdx = blockIdx.x;
    int angleIdx   = threadIdx.x;
    if (angleIdx < numOfDegrees)
    {
        *rotatedPointsTensor.ptr(contourIdx, angleIdx, 0) = INT_MAX;
        *rotatedPointsTensor.ptr(contourIdx, angleIdx, 1) = INT_MAX;
        *rotatedPointsTensor.ptr(contourIdx, angleIdx, 2) = INT_MIN;
        *rotatedPointsTensor.ptr(contourIdx, angleIdx, 3) = INT_MIN;
        *rotatedPointsTensor.ptr(contourIdx, angleIdx, 4) = -1;
        *rotatedPointsTensor.ptr(contourIdx, angleIdx, 5) = -1;
    }
}

__global__ void calculateRotateCoef(cuda::Tensor2DWrap<float> aCoeffs, const int degrees)
{
    int angle = blockIdx.x * blockDim.x + threadIdx.x;
    if (angle < degrees)
    {
        *aCoeffs.ptr(angle, 0) = cos(angle * PI / 180);
        *aCoeffs.ptr(angle, 1) = sin(angle * PI / 180);
    }
}

void calculateRotateCoefCUDA(cuda::Tensor2DWrap<float> rotateCoefBuf, const int degrees, const cudaStream_t &stream)
{
    dim3 block(BLOCK * 8);
    dim3 grid(divUp(degrees, block.x));
    calculateRotateCoef<<<grid, block, 0, stream>>>(rotateCoefBuf, degrees);
}

template<typename T>
__global__ void calculateRotateArea(cuda::Tensor3DWrap<T>   inContourPointsData,
                                    cuda::Tensor3DWrap<int> rotatedPointsTensor, cuda::Tensor2DWrap<float> rotateCoeffs,
                                    cuda::Tensor2DWrap<int> numPointsInContourBuf)
{
    int pointIdx   = blockIdx.x * blockDim.x + threadIdx.x;
    int contourIdx = blockIdx.y;

    int                     angleIdx = blockIdx.z;
    extern __shared__ float rotateCoeffs_sm[];
    rotateCoeffs_sm[2 * angleIdx]     = *rotateCoeffs.ptr(angleIdx, 0);
    rotateCoeffs_sm[2 * angleIdx + 1] = *rotateCoeffs.ptr(angleIdx, 1);

    __syncthreads();

    if (pointIdx < *numPointsInContourBuf.ptr(0, contourIdx))
    {
        T     px        = *inContourPointsData.ptr(contourIdx, pointIdx, 0);
        T     py        = *inContourPointsData.ptr(contourIdx, pointIdx, 1);
        float cos_coeff = rotateCoeffs_sm[2 * angleIdx];
        float sin_coeff = rotateCoeffs_sm[2 * angleIdx + 1];
        int   px_rot    = (px * cos_coeff) - (py * sin_coeff);
        int   py_rot    = (px * sin_coeff) + (py * cos_coeff);
        //xmin
        atomicMin(rotatedPointsTensor.ptr(contourIdx, angleIdx, 0), px_rot);
        //ymin
        atomicMin(rotatedPointsTensor.ptr(contourIdx, angleIdx, 1), py_rot);
        //xmax
        atomicMax(rotatedPointsTensor.ptr(contourIdx, angleIdx, 2), px_rot);
        //ymax
        atomicMax(rotatedPointsTensor.ptr(contourIdx, angleIdx, 3), py_rot);

        __threadfence();
        int rectWidth
            = *rotatedPointsTensor.ptr(contourIdx, angleIdx, 2) - *rotatedPointsTensor.ptr(contourIdx, angleIdx, 0);
        int rectHeight
            = *rotatedPointsTensor.ptr(contourIdx, angleIdx, 3) - *rotatedPointsTensor.ptr(contourIdx, angleIdx, 1);
        *rotatedPointsTensor.ptr(contourIdx, angleIdx, 4) = rectWidth * rectHeight;
        *rotatedPointsTensor.ptr(contourIdx, angleIdx, 5) = angleIdx;
    }
}

/**
 * Find the min area of the contours' bounding box and the related rotated degress
 * To use this function, the grid should be set as the same number of contour batch size.
 * each thread in blocks will process one degress, and calculate the original rotated bounding box.
 */
template<typename TensorWrapper>
__global__ void findMinAreaAndAngle(TensorWrapper rotatedPointsTensor, cuda::Tensor2DWrap<float> outMinAreaRectBox,
                                    const int numOfDegrees)
{
    int angleIdx = threadIdx.x;
    if (angleIdx > numOfDegrees)
    {
        return;
    }

    int                   rectIdx = blockIdx.x;
    extern __shared__ int areaAngleBuf_sm[];
    areaAngleBuf_sm[2 * angleIdx]     = *rotatedPointsTensor.ptr(rectIdx, angleIdx, 4);
    areaAngleBuf_sm[2 * angleIdx + 1] = *rotatedPointsTensor.ptr(rectIdx, angleIdx, 5);
    __syncthreads();

    for (int stride = numOfDegrees / 2; stride > 0; stride >>= 1)
    {
        if (angleIdx < stride)
        {
            int *curAreaIdx   = &areaAngleBuf_sm[2 * angleIdx];
            int *nextAreaIdx  = &areaAngleBuf_sm[2 * (angleIdx + stride)];
            int *curAngleIdx  = &areaAngleBuf_sm[2 * angleIdx + 1];
            int *nextAngleIdx = &areaAngleBuf_sm[2 * (angleIdx + stride) + 1];
            if (*curAreaIdx > *nextAreaIdx)
            {
                *curAreaIdx  = *nextAreaIdx;
                *curAngleIdx = *nextAngleIdx;
            }
        }
        __syncthreads();

        if (stride % 2 == 1 && areaAngleBuf_sm[0] > areaAngleBuf_sm[2 * (stride - 1)])
        {
            areaAngleBuf_sm[0] = areaAngleBuf_sm[2 * (stride - 1)];
            areaAngleBuf_sm[1] = areaAngleBuf_sm[2 * (stride - 1) + 1];
        }
        __syncthreads();
    }
    if (numOfDegrees % 2 == 1 && areaAngleBuf_sm[0] > areaAngleBuf_sm[2 * (numOfDegrees - 1)])
    {
        areaAngleBuf_sm[0] = areaAngleBuf_sm[2 * (numOfDegrees - 1)];
        areaAngleBuf_sm[1] = areaAngleBuf_sm[2 * (numOfDegrees - 1) + 1];
    }
    if (threadIdx.x == 0)
    {
        int   minRotateAngle = areaAngleBuf_sm[1];
        float cos_coeff      = cos(-minRotateAngle * PI / 180);
        float sin_coeff      = sin(-minRotateAngle * PI / 180);
        float xmin           = *rotatedPointsTensor.ptr(rectIdx, areaAngleBuf_sm[1], 0);
        float ymin           = *rotatedPointsTensor.ptr(rectIdx, areaAngleBuf_sm[1], 1);
        float xmax           = *rotatedPointsTensor.ptr(rectIdx, areaAngleBuf_sm[1], 2);
        float ymax           = *rotatedPointsTensor.ptr(rectIdx, areaAngleBuf_sm[1], 3);

        float tl_x = (xmin * cos_coeff) - (ymin * sin_coeff);
        float tl_y = (xmin * sin_coeff) + (ymin * cos_coeff);
        float br_x = (xmax * cos_coeff) - (ymax * sin_coeff);
        float br_y = (xmax * sin_coeff) + (ymax * cos_coeff);
        float tr_x = (xmax * cos_coeff) - (ymin * sin_coeff);
        float tr_y = (xmax * sin_coeff) + (ymin * cos_coeff);
        float bl_x = (xmin * cos_coeff) - (ymax * sin_coeff);
        float bl_y = (xmin * sin_coeff) + (ymax * cos_coeff);

        *outMinAreaRectBox.ptr(rectIdx, 0) = bl_x;
        *outMinAreaRectBox.ptr(rectIdx, 1) = bl_y;
        *outMinAreaRectBox.ptr(rectIdx, 2) = tl_x;
        *outMinAreaRectBox.ptr(rectIdx, 3) = tl_y;
        *outMinAreaRectBox.ptr(rectIdx, 4) = tr_x;
        *outMinAreaRectBox.ptr(rectIdx, 5) = tr_y;
        *outMinAreaRectBox.ptr(rectIdx, 6) = br_x;
        *outMinAreaRectBox.ptr(rectIdx, 7) = br_y;
    }
}

template<typename T>
void minAreaRect(const TensorDataStridedCuda &inData, void *rotatedPointsDev,
                 const cuda::Tensor2DWrap<float> rotateCoeffsData, const TensorDataStridedCuda &numPointsInContour,
                 const TensorDataStridedCuda &outData, int contourBatch, int maxNumPointsInContour, cudaStream_t stream)
{
    auto inAccess = nvcv::TensorDataAccessStrided::Create(inData);
    NVCV_ASSERT(inAccess);

    cuda::Tensor3DWrap<T> inContourPointsData(inData);

    int                       kernelPitch2 = static_cast<int>(_MIN_AREA_EACH_ANGLE_STRID * sizeof(int));
    int                       kernelPitch1 = _MAX_ROTATE_DEGREES * kernelPitch2;
    cuda::Tensor3DWrap<int>   rotatedPointsTensor(rotatedPointsDev, kernelPitch1, kernelPitch2);
    cuda::Tensor2DWrap<float> outMinAreaRectData(outData);
    cuda::Tensor2DWrap<int>   pointsInContourData(numPointsInContour);

    dim3 block1(128);
    dim3 grid1(contourBatch);
    resetRotatedPointsBuf<<<grid1, block1, 0, stream>>>(rotatedPointsTensor, _MAX_ROTATE_DEGREES);
    checkKernelErrors();

    dim3   block2(256);
    dim3   grid2(divUp(maxNumPointsInContour, block2.x), contourBatch, _MAX_ROTATE_DEGREES);
    size_t smem_size = 2 * _MAX_ROTATE_DEGREES * sizeof(float);
    calculateRotateArea<<<grid2, block2, smem_size, stream>>>(inContourPointsData, rotatedPointsTensor,
                                                              rotateCoeffsData, pointsInContourData);
    checkKernelErrors();
    cudaStreamSynchronize(stream);

    dim3 grid3(contourBatch);
    findMinAreaAndAngle<<<grid3, block2, smem_size, stream>>>(rotatedPointsTensor, outMinAreaRectData,
                                                              _MAX_ROTATE_DEGREES);
    checkKernelErrors();
}

MinAreaRect::MinAreaRect(DataShape max_input_shape, DataShape max_output_shape, int maxContourNum)
    : CudaBaseOp(max_input_shape, max_output_shape)
    , mMaxContourNum(maxContourNum)
{
    NVCV_CHECK_THROW(cudaMalloc(&mRotateCoeffsBufDev, _MAX_ROTATE_DEGREES * 2 * sizeof(float)));
    NVCV_CHECK_THROW(
        cudaMalloc(&mRotatedPointsDev, maxContourNum * _MAX_ROTATE_DEGREES * _MIN_AREA_EACH_ANGLE_STRID * sizeof(int)));
}

MinAreaRect::~MinAreaRect()
{
    NVCV_CHECK_LOG(cudaFree(mRotateCoeffsBufDev));
    NVCV_CHECK_LOG(cudaFree(mRotatedPointsDev));
}

size_t MinAreaRect::calBufferSize(DataShape max_input_shape, DataShape max_output_shape, int maxContourNum)
{
    return maxContourNum * _MAX_ROTATE_DEGREES * _MIN_AREA_EACH_ANGLE_STRID * sizeof(int);
}

ErrorCode MinAreaRect::infer(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData,
                             const TensorDataStridedCuda &numPointsInContour, const int totalContours,
                             cudaStream_t stream)
{
    cuda_op::DataType input_datatype  = GetLegacyDataType(inData.dtype());
    cuda_op::DataType output_datatype = GetLegacyDataType(outData.dtype());

    auto inAccess = nvcv::TensorDataAccessStrided::Create(inData);
    NVCV_ASSERT(inAccess);
    auto outAccess = nvcv::TensorDataAccessStrided::Create(outData);
    NVCV_ASSERT(outAccess);

    auto inShape               = inAccess->shape();
    int  contourBatch          = inShape[0];
    int  maxNumPointsInContour = inShape[1];
    if ((contourBatch > mMaxContourNum))
    {
        LOG_ERROR("Invalid contour number " << contourBatch);
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    cuda::Tensor2DWrap<float> rotateCoeffsData(mRotateCoeffsBufDev, static_cast<int>(2 * sizeof(float)));
    calculateRotateCoefCUDA(rotateCoeffsData, _MAX_ROTATE_DEGREES, stream);

    typedef void (*minAreaRect_t)(const TensorDataStridedCuda &inData, void *rotatedPointsDev,
                                  const cuda::Tensor2DWrap<float> rotateCoeffsData,
                                  const TensorDataStridedCuda &numPointsInContour, const TensorDataStridedCuda &outData,
                                  int batch, int maxNumPointsInContour, cudaStream_t stream);
    static const minAreaRect_t funcs[5] = {0, 0, minAreaRect<ushort>, minAreaRect<short>, minAreaRect<int>};

    funcs[input_datatype](inData, mRotatedPointsDev, rotateCoeffsData, numPointsInContour, outData, contourBatch,
                          maxNumPointsInContour, stream);

    return ErrorCode::SUCCESS;
}
} // namespace nvcv::legacy::cuda_op
