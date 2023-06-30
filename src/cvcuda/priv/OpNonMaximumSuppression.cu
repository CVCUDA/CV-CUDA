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
**/

#include "OpNonMaximumSuppression.hpp"

#include <nvcv/DataType.hpp>
#include <nvcv/Exception.hpp>
#include <nvcv/TensorData.hpp>
#include <nvcv/TensorLayout.hpp>
#include <nvcv/cuda/DropCast.hpp>
#include <nvcv/cuda/MathOps.hpp>
#include <nvcv/cuda/MathWrappers.hpp>
#include <nvcv/cuda/StaticCast.hpp>
#include <nvcv/cuda/TensorWrap.hpp>
#include <util/CheckError.hpp>
#include <util/Math.hpp>

namespace cuda = nvcv::cuda;
namespace util = nvcv::util;

namespace {

template<typename T>
inline __device__ float ComputeArea(const T &bbox)
{
    return bbox.z * bbox.w;
}

template<typename T>
inline __device__ float ComputeIoU(const T &box1, const T &box2)
{
    int   xInterLeft   = cuda::max(box1.x, box2.x);
    int   yInterTop    = cuda::max(box1.y, box2.y);
    int   xInterRight  = cuda::min(box1.x + box1.z, box2.x + box2.z);
    int   yInterBottom = cuda::min(box1.y + box1.w, box2.y + box2.w);
    int   widthInter   = xInterRight - xInterLeft;
    int   heightInter  = yInterBottom - yInterTop;
    float interArea    = widthInter * heightInter;
    float iou          = 0.f;
    if (widthInter > 0.f && heightInter > 0.f)
    {
        float unionArea = ComputeArea(box1) + ComputeArea(box2) - interArea;
        if (unionArea > 0.f)
        {
            iou = interArea / unionArea;
        }
    }
    return iou;
}

template<typename T, typename U>
__global__ void NonMaximumSuppression(cuda::Tensor2DWrap<const T> inBBoxes, cuda::Tensor2DWrap<U> outMask,
                                      cuda::Tensor2DWrap<const float> inScores, int numBBoxes, float scoreThreshold,
                                      float iouThreshold)
{
    const int bboxX = blockDim.x * blockIdx.x + threadIdx.x;
    if (bboxX >= numBBoxes)
    {
        return;
    }

    const int   batchIdx = blockIdx.z;
    const int2  coordX{bboxX, batchIdx};
    const float scoreX = inScores[coordX];

    U &dst = outMask[coordX];

    if (scoreX < scoreThreshold)
    {
        dst = 0;
        return;
    }

    const T srcX    = inBBoxes[coordX];
    bool    discard = false;

    for (int bboxY = 0; bboxY < numBBoxes; ++bboxY)
    {
        if (bboxX == bboxY)
        {
            continue;
        }

        const int2 coordY{bboxY, batchIdx};
        const T    srcY = inBBoxes[coordY];

        if (ComputeIoU(srcX, srcY) > iouThreshold)
        {
            const float scoreY = inScores[coordY];

            if (scoreX < scoreY || (scoreX == scoreY && ComputeArea(srcX) < ComputeArea(srcY)))
            {
                discard = true;
                break;
            }
        }
    }

    dst = discard ? 0 : 1;
}

inline __host__ void RunNonMaximumSuppresion(const nvcv::TensorDataStridedCuda &in,
                                             const nvcv::TensorDataStridedCuda &out,
                                             const nvcv::TensorDataStridedCuda &scores, float scThresh, float iouThresh,
                                             cudaStream_t stream)
{
    cuda::Tensor2DWrap<const short4> inWrap(in);
    cuda::Tensor2DWrap<uint8_t>      outWrap(out);
    cuda::Tensor2DWrap<const float>  scoresWrap(scores);

    int numSamples = in.shape(0);
    int numBBoxes  = in.shape(1);

    dim3 block(256, 1, 1);
    dim3 grid((numBBoxes + block.x - 1) / block.x, 1, numSamples);

    NonMaximumSuppression<<<grid, block, 0, stream>>>(inWrap, outWrap, scoresWrap, numBBoxes, scThresh, iouThresh);
}

} // namespace

// =============================================================================
// NonMaximumSuppression Class Definition
// =============================================================================

namespace cvcuda::priv {

NonMaximumSuppression::NonMaximumSuppression() {}

void NonMaximumSuppression::operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out,
                                       const nvcv::Tensor &scores, float scoreThreshold, float iouThreshold) const
{
    auto inData = in.exportData<nvcv::TensorDataStridedCuda>();
    if (!inData)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input must be cuda-accessible, pitch-linear tensor");
    }

    auto outData = out.exportData<nvcv::TensorDataStridedCuda>();
    if (!outData)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Output must be cuda-accessible, pitch-linear tensor");
    }

    auto scoreData = scores.exportData<nvcv::TensorDataStridedCuda>();
    if (!scoreData)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Scores must be cuda-accessible, pitch-linear tensor");
    }

    if (!((inData->rank() == 3 && inData->dtype() == nvcv::TYPE_S16 && inData->shape(2) == 4)
          || (inData->rank() == 3 && inData->dtype() == nvcv::TYPE_4S16 && inData->shape(2) == 1)
          || (inData->rank() == 2 && inData->dtype() == nvcv::TYPE_4S16)))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input tensor must have rank 2 or 3 and 4xS16 or 4S16 data type");
    }
    if (!((outData->rank() == 3 && outData->dtype() == nvcv::TYPE_U8 && outData->shape(2) == 1)
          || (outData->rank() == 2 && outData->dtype() == nvcv::TYPE_U8)))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Output tensor must have rank 2 or 3 and U8 data type");
    }
    if (!((scoreData->rank() == 3 && scoreData->dtype() == nvcv::TYPE_F32 && scoreData->shape(2) == 1)
          || (scoreData->rank() == 2 && scoreData->dtype() == nvcv::TYPE_F32)))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Scores tensor must have rank 2 or 3 and 1xF32 data type");
    }
    if (inData->shape(0) != outData->shape(0) || inData->shape(0) != scoreData->shape(0))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input, output and scores number of batches (first shape) must be equal");
    }
    if (inData->shape(1) != outData->shape(1) || inData->shape(1) != scoreData->shape(1))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input, output and scores number of boxes (second shape) must be equal");
    }

    if (iouThreshold <= 0.f || iouThreshold > 1.f)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "IoU threshold must be in (0, 1]");
    }

    RunNonMaximumSuppresion(*inData, *outData, *scoreData, scoreThreshold, iouThreshold, stream);
}

} // namespace cvcuda::priv
