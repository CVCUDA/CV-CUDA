/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "Nvtx.hpp"
#include "OpMinAreaRect.hpp"

#include <cvcuda/cuda_tools/TensorWrap.hpp>
#include <nvcv/DataType.hpp>
#include <nvcv/Exception.hpp>
#include <nvcv/TensorData.hpp>
#include <nvcv/TensorLayout.hpp>
#include <nvcv/util/CheckError.hpp>
#include <nvcv/util/Math.hpp>

#include <climits>
#include <cstddef>
#include <memory>

namespace cuda = nvcv::cuda;
namespace util = nvcv::util;

namespace {

constexpr double kPi = 3.1415926535897932384626433832795;

// tl and br point coords + area + angle
constexpr int kMinAreaEachAngleStride = 6;

constexpr int kMaxRotateDegrees = 90;

// Byte strides of the two device scratch buffers. They live here because DeviceBuffers' allocation
// and the TensorWrap views the kernels index with have to agree on the layout.
constexpr int kRotateCoeffPitch = static_cast<int>(2 * sizeof(float));
constexpr int kAnglePitch       = static_cast<int>(kMinAreaEachAngleStride * sizeof(int));
// The sweep only fills angles 0..kMaxRotateDegrees-1, so the last slot is never read. The extra
// slot is kept so the allocation stays byte-identical to the legacy operator's.
constexpr int kContourPitch = (kMaxRotateDegrees + 1) * kAnglePitch;

__global__ void ResetRotatedPointsBuf(cuda::Tensor3DWrap<int> rotatedPointsTensor, const int numOfDegrees)
{
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

__global__ void CalculateRotateCoef(cuda::Tensor2DWrap<float> aCoeffs, const int degrees)
{
    int angle = blockIdx.x * blockDim.x + threadIdx.x;
    if (angle < degrees)
    {
        *aCoeffs.ptr(angle, 0) = cos(angle * kPi / 180);
        *aCoeffs.ptr(angle, 1) = sin(angle * kPi / 180);
    }
}

template<typename T>
__global__ void CalculateRotateArea(cuda::Tensor3DWrap<T>   inContourPointsData,
                                    cuda::Tensor3DWrap<int> rotatedPointsTensor, cuda::Tensor2DWrap<float> rotateCoeffs,
                                    cuda::Tensor2DWrap<int> numPointsInContourBuf, int maxNumPointsInContour)
{
    int pointIdx   = blockIdx.x * blockDim.x + threadIdx.x;
    int contourIdx = blockIdx.y;

    if (pointIdx < min(*numPointsInContourBuf.ptr(0, contourIdx), maxNumPointsInContour))
    {
        int   angleIdx  = blockIdx.z;
        T     px        = *inContourPointsData.ptr(contourIdx, pointIdx, 0);
        T     py        = *inContourPointsData.ptr(contourIdx, pointIdx, 1);
        float cos_coeff = *rotateCoeffs.ptr(angleIdx, 0);
        float sin_coeff = *rotateCoeffs.ptr(angleIdx, 1);
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
__global__ void FindMinAreaAndAngle(cuda::Tensor3DWrap<int>   rotatedPointsTensor,
                                    cuda::Tensor2DWrap<float> outMinAreaRectBox, const int numOfDegrees)
{
    int angleIdx = threadIdx.x;

    int                   rectIdx = blockIdx.x;
    extern __shared__ int areaAngleBuf_sm[];
    if (angleIdx < numOfDegrees)
    {
        areaAngleBuf_sm[2 * angleIdx]       = *rotatedPointsTensor.ptr(rectIdx, angleIdx, 4);
        areaAngleBuf_sm[(2 * angleIdx) + 1] = *rotatedPointsTensor.ptr(rectIdx, angleIdx, 5);
    }

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

        // Halving an odd stride drops the top element from the next round; this folds it back in.
        if (angleIdx == 0 && stride % 2 == 1 && areaAngleBuf_sm[0] > areaAngleBuf_sm[2 * (stride - 1)])
        {
            areaAngleBuf_sm[0] = areaAngleBuf_sm[2 * (stride - 1)];
            areaAngleBuf_sm[1] = areaAngleBuf_sm[2 * (stride - 1) + 1];
        }

        __syncthreads();
    }

    // Same fold-back for an odd initial count, which the halving loop never visits.
    if (angleIdx == 0 && numOfDegrees % 2 == 1 && areaAngleBuf_sm[0] > areaAngleBuf_sm[2 * (numOfDegrees - 1)])
    {
        areaAngleBuf_sm[0] = areaAngleBuf_sm[2 * (numOfDegrees - 1)];
        areaAngleBuf_sm[1] = areaAngleBuf_sm[2 * (numOfDegrees - 1) + 1];
    }

    if (threadIdx.x == 0)
    {
        int minRotateAngle = areaAngleBuf_sm[1];

        float cos_coeff = cos(-minRotateAngle * kPi / 180);
        float sin_coeff = sin(-minRotateAngle * kPi / 180);
        float xmin      = *rotatedPointsTensor.ptr(rectIdx, minRotateAngle, 0);
        float ymin      = *rotatedPointsTensor.ptr(rectIdx, minRotateAngle, 1);
        float xmax      = *rotatedPointsTensor.ptr(rectIdx, minRotateAngle, 2);
        float ymax      = *rotatedPointsTensor.ptr(rectIdx, minRotateAngle, 3);

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

inline void LaunchCalculateRotateCoef(cuda::Tensor2DWrap<float> rotateCoeffsData, const int degrees,
                                      cudaStream_t stream)
{
    dim3 block(256);
    dim3 grid(util::DivUp(degrees, static_cast<int>(block.x)));
    CalculateRotateCoef<<<grid, block, 0, stream>>>(rotateCoeffsData, degrees);

    NVCV_CHECK_THROW(cudaGetLastError());
}

template<typename T>
void RunMinAreaRect(const nvcv::TensorDataStridedCuda &inData, int *rotatedPointsDev,
                    cuda::Tensor2DWrap<float> rotateCoeffsData, const nvcv::TensorDataStridedCuda &numPointsInContour,
                    const nvcv::TensorDataStridedCuda &outData, int contourBatch, int maxNumPointsInContour,
                    cudaStream_t stream)
{
    cuda::Tensor3DWrap<T> inContourPointsData(inData);

    cuda::Tensor3DWrap<int>   rotatedPointsTensor(rotatedPointsDev, kContourPitch, kAnglePitch);
    cuda::Tensor2DWrap<float> outMinAreaRectData(outData);
    cuda::Tensor2DWrap<int>   pointsInContourData(numPointsInContour);

    dim3 block1(128);
    dim3 grid1(contourBatch);
    ResetRotatedPointsBuf<<<grid1, block1, 0, stream>>>(rotatedPointsTensor, kMaxRotateDegrees);
    NVCV_CHECK_THROW(cudaGetLastError());

    dim3 block2(maxNumPointsInContour <= 512 ? 128 : 256);
    dim3 grid2(util::DivUp(maxNumPointsInContour, static_cast<int>(block2.x)), contourBatch, kMaxRotateDegrees);
    CalculateRotateArea<<<grid2, block2, 0, stream>>>(inContourPointsData, rotatedPointsTensor, rotateCoeffsData,
                                                      pointsInContourData, maxNumPointsInContour);
    NVCV_CHECK_THROW(cudaGetLastError());

    // Redundant for correctness: same-stream ordering already retires CalculateRotateArea before
    // FindMinAreaAndAngle starts, and it does not repair the intra-kernel race on slots 4/5. Kept
    // because dropping it changes the operator's host-side blocking behavior, which this
    // parity-preserving migration does not do.
    NVCV_CHECK_THROW(cudaStreamSynchronize(stream));

    dim3 block3(128);
    dim3 grid3(contourBatch);

    // The reduction stores an area and angle for every produced angle.
    size_t smem_size = 2 * kMaxRotateDegrees * sizeof(int);
    FindMinAreaAndAngle<<<grid3, block3, smem_size, stream>>>(rotatedPointsTensor, outMinAreaRectData,
                                                              kMaxRotateDegrees);
    NVCV_CHECK_THROW(cudaGetLastError());
}

} // namespace

namespace cvcuda::priv {

MinAreaRect::DeviceBuffers::DeviceBuffers(int maxContourNum)
{
    try
    {
        NVCV_CHECK_THROW(cudaMalloc(&rotateCoeffs, static_cast<size_t>(kMaxRotateDegrees) * kRotateCoeffPitch));
        NVCV_CHECK_THROW(cudaMalloc(&rotatedPoints, static_cast<size_t>(maxContourNum) * kContourPitch));
    }
    catch (...)
    {
        // rotatedPoints scales with maxContourNum, so it is the allocation that realistically fails;
        // without this the earlier rotateCoeffs buffer would leak with no handle left to free it.
        cleanup();
        throw;
    }
}

MinAreaRect::DeviceBuffers::~DeviceBuffers()
{
    cleanup();
}

void MinAreaRect::DeviceBuffers::cleanup() noexcept
{
    NVCV_CHECK_LOG(cudaFree(rotateCoeffs));
    rotateCoeffs = nullptr;
    NVCV_CHECK_LOG(cudaFree(rotatedPoints));
    rotatedPoints = nullptr;
}

MinAreaRect::MinAreaRect(int maxContourNum)
    : m_maxContourNum(maxContourNum)
    , m_buffers([maxContourNum](int) { return std::make_unique<DeviceBuffers>(maxContourNum); })
{
}

void MinAreaRect::operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out,
                             const nvcv::Tensor &numPointsInContour, const int totalContours) const
{
    CVCUDA_NVTX_RANGE("cvcuda::MinAreaRect::operator()[Tensor]");
    auto inData = in.exportData<nvcv::TensorDataStridedCuda>();
    if (inData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input must be cuda-accessible, pitch-linear tensor");
    }

    auto outData = out.exportData<nvcv::TensorDataStridedCuda>();
    if (outData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Output must be cuda-accessible, pitch-linear tensor");
    }
    auto numPointsInContourData = numPointsInContour.exportData<nvcv::TensorDataStridedCuda>();
    if (numPointsInContourData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "numPointsInContour must be cuda-accessible, pitch-linear tensor");
    }

    if (inData->layout() != nvcv::TENSOR_NWC)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input must have NWC layout");
    }
    if (numPointsInContourData->layout() != nvcv::TENSOR_NW)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input must have NW layout");
    }
    if (outData->layout() != nvcv::TENSOR_NW)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Output must have NW layout");
    }

    auto inShape = inData->shape();
    if (auto channels = static_cast<int>(inShape[inShape.rank() - 1]); channels != 2)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input must have 2 channels (x, y coordinates per point)");
    }

    nvcv::DataType inDtype = inData->dtype();
    if (inDtype != nvcv::TYPE_U16 && inDtype != nvcv::TYPE_S16 && inDtype != nvcv::TYPE_S32)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input must have TYPE_U16, TYPE_S16, or TYPE_S32 data type");
    }
    if (outData->dtype() != nvcv::TYPE_F32)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Output must have TYPE_F32 data type");
    }
    if (numPointsInContourData->dtype() != nvcv::TYPE_S32)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "numPointsInContourData must have TYPE_S32 data type");
    }

    int contourBatch          = static_cast<int>(inShape[0]);
    int maxNumPointsInContour = static_cast<int>(inShape[1]);
    if (contourBatch > m_maxContourNum)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid contour number %d", contourBatch);
    }

    DeviceBuffers &buffers = m_buffers.get();

    cuda::Tensor2DWrap<float> rotateCoeffsData(buffers.rotateCoeffs, kRotateCoeffPitch);

    LaunchCalculateRotateCoef(rotateCoeffsData, kMaxRotateDegrees, stream);

    if (inDtype == nvcv::TYPE_U16)
    {
        RunMinAreaRect<unsigned short>(*inData, buffers.rotatedPoints, rotateCoeffsData, *numPointsInContourData,
                                       *outData, contourBatch, maxNumPointsInContour, stream);
    }
    else if (inDtype == nvcv::TYPE_S16)
    {
        RunMinAreaRect<short>(*inData, buffers.rotatedPoints, rotateCoeffsData, *numPointsInContourData, *outData,
                              contourBatch, maxNumPointsInContour, stream);
    }
    else
    {
        RunMinAreaRect<int>(*inData, buffers.rotatedPoints, rotateCoeffsData, *numPointsInContourData, *outData,
                            contourBatch, maxNumPointsInContour, stream);
    }
}

} // namespace cvcuda::priv
