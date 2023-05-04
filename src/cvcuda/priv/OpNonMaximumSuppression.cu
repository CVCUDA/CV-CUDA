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
#include <nvcv/cuda/StaticCast.hpp>
#include <nvcv/cuda/TensorWrap.hpp>
#include <util/Math.hpp>

namespace cuda = nvcv::cuda;
namespace util = nvcv::util;

__device__ float compute_bbox_iou(int batch, int bboxX, int bboxY, const cuda::Tensor3DWrap<int> &bBoxX,
                                  const cuda::Tensor3DWrap<int> &bBoxY);

__device__ bool bbox_eq(int batch, int idxX, int idxY, const cuda::Tensor3DWrap<int> &bBoxX,
                        const cuda::Tensor3DWrap<int> &bBoxY);

__device__ __forceinline__ float bbox_area(int batch, int idxX, const cuda::Tensor3DWrap<int> &bBoxX);

__device__ bool bbox_nonzero(int batch, int idxX, const cuda::Tensor3DWrap<int> &bBoxX);

__host__ void non_maximum_suppresion(const nvcv::TensorDataStridedCuda &in, const nvcv::TensorDataStridedCuda &out,
                                     const nvcv::TensorDataStridedCuda &scores, float score_threshold,
                                     float iou_threshold, cudaStream_t stream);

__device__ __forceinline__ float bbox_area(int batch, int idxX, const cuda::Tensor3DWrap<int> &bBoxX)
{
    return static_cast<float>(*bBoxX.ptr(batch, idxX, 2) * *bBoxX.ptr(batch, idxX, 3));
}

__device__ bool bbox_eq(int batch, int idxX, int idxY, const cuda::Tensor3DWrap<int> &bBoxX,
                        const cuda::Tensor3DWrap<int> &bBoxY)
{
    return (*bBoxX.ptr(batch, idxX, 0) == *bBoxY.ptr(batch, idxY, 0))
        && (*bBoxX.ptr(batch, idxX, 1) == *bBoxY.ptr(batch, idxY, 1))
        && (*bBoxX.ptr(batch, idxX, 2) == *bBoxY.ptr(batch, idxY, 2))
        && (*bBoxX.ptr(batch, idxX, 3) == *bBoxY.ptr(batch, idxY, 3));
}

__device__ bool bbox_nonzero(int batch, int idxX, const cuda::Tensor3DWrap<int> &bBoxX)
{
    return (*bBoxX.ptr(batch, idxX, 2) > 0) && (*bBoxX.ptr(batch, idxX, 3) > 0);
}

__device__ float compute_bbox_iou(int batch, int bboxX, int bboxY, const cuda::Tensor3DWrap<int> &bBoxX,
                                  const cuda::Tensor3DWrap<int> &bBoxY)
{
    auto box1 = make_int4(*bBoxX.ptr(batch, bboxX, 0), *bBoxX.ptr(batch, bboxX, 1), *bBoxX.ptr(batch, bboxX, 2),
                          *bBoxX.ptr(batch, bboxX, 3));
    auto box2 = make_int4(*bBoxY.ptr(batch, bboxY, 0), *bBoxY.ptr(batch, bboxY, 1), *bBoxY.ptr(batch, bboxY, 2),
                          *bBoxY.ptr(batch, bboxY, 3));

    float box1Area = box1.z * box1.w;
    float box2Area = box2.z * box2.w;

    auto xInterLeft   = max(box1.x, box2.x);
    auto yInterTop    = max(box1.y, box2.y);
    auto xInterRight  = min(box1.x + box1.z, box2.x + box2.z);
    auto yInterBottom = min(box1.y + box1.w, box2.y + box2.w);

    auto widthInter  = xInterRight - xInterLeft;
    auto heightInter = yInterBottom - yInterTop;

    float interArea = widthInter * heightInter;

    float iou = 0.0f;
    if (widthInter > 0.0f && heightInter > 0.0f)
    {
        iou = interArea / (box1Area + box2Area - interArea);
    }

    return iou;
}

//template<typename SrcWrapper, typename DstWrapper, typename ScoreWrapper>
__global__ void copy_bbox(int batch, int numProposals, const cuda::Tensor3DWrap<int> bBoxProposals,
                          const cuda::Tensor3DWrap<int> bBoxSelected, const cuda::Tensor2DWrap<float> bBoxScores,
                          float scoreThreshold, float iouThreshold)
{
    int bboxX = blockDim.x * blockIdx.x + threadIdx.x;

    if (bboxX < numProposals)
    {
        if (*bBoxScores.ptr(batch, bboxX) >= scoreThreshold)
        {
            *bBoxSelected.ptr(batch, bboxX, 0) = *bBoxProposals.ptr(batch, bboxX, 0);
            *bBoxSelected.ptr(batch, bboxX, 1) = *bBoxProposals.ptr(batch, bboxX, 1);
            *bBoxSelected.ptr(batch, bboxX, 2) = *bBoxProposals.ptr(batch, bboxX, 2);
            *bBoxSelected.ptr(batch, bboxX, 3) = *bBoxProposals.ptr(batch, bboxX, 3);
        }
        else
        {
            *bBoxSelected.ptr(batch, bboxX, 0) = 0;
            *bBoxSelected.ptr(batch, bboxX, 1) = 0;
            *bBoxSelected.ptr(batch, bboxX, 2) = 0;
            *bBoxSelected.ptr(batch, bboxX, 3) = 0;
        }
        __threadfence();

        for (auto bboxY = 0; bboxY < numProposals; ++bboxY)
        {
            if (bbox_eq(batch, bboxX, bboxY, bBoxSelected, bBoxSelected)
                && (*bBoxScores.ptr(batch, bboxX) <= *bBoxScores.ptr(batch, bboxY)) && (bboxY < bboxX))
            {
                *bBoxSelected.ptr(batch, bboxX, 0) = 0;
                *bBoxSelected.ptr(batch, bboxX, 1) = 0;
                *bBoxSelected.ptr(batch, bboxX, 2) = 0;
                *bBoxSelected.ptr(batch, bboxX, 3) = 0;
                __threadfence();
            }
        }

        for (auto bboxY = 0; bboxY < numProposals; ++bboxY)
        {
            if (bbox_nonzero(batch, bboxX, bBoxSelected) && bbox_nonzero(batch, bboxY, bBoxSelected)
                && (compute_bbox_iou(batch, bboxX, bboxY, bBoxSelected, bBoxSelected) > iouThreshold))
            {
                if ((*bBoxScores.ptr(batch, bboxX) < *bBoxScores.ptr(batch, bboxY))
                    || ((*bBoxScores.ptr(batch, bboxX) == *bBoxScores.ptr(batch, bboxY))
                        && bbox_area(batch, bboxX, bBoxSelected) < bbox_area(batch, bboxY, bBoxSelected)))
                {
                    *bBoxSelected.ptr(batch, bboxX, 0) = 0;
                    *bBoxSelected.ptr(batch, bboxX, 1) = 0;
                    *bBoxSelected.ptr(batch, bboxX, 2) = 0;
                    *bBoxSelected.ptr(batch, bboxX, 3) = 0;
                    __threadfence();
                }
            }
        }
    }
}

__host__ void non_maximum_suppresion(const nvcv::TensorDataStridedCuda &in, const nvcv::TensorDataStridedCuda &out,
                                     const nvcv::TensorDataStridedCuda &scores, float score_threshold,
                                     float iou_threshold, cudaStream_t stream)
{
    constexpr int TILE_DIM = 16;

    auto inAccess = nvcv::TensorDataAccessStrided::Create(in);
    NVCV_ASSERT(inAccess);
    auto inShape = inAccess->shape();

    cuda::Tensor3DWrap<int>   bBoxProposals(in);
    cuda::Tensor3DWrap<int>   bBoxSelected(out);
    cuda::Tensor2DWrap<float> bBoxScores(scores);

    auto batchSize  = inShape[0];
    auto bboxLength = inShape[1];

    // Tiling threads over image at tiling dimension block size
    auto blockSize = dim3(TILE_DIM * TILE_DIM, 1);
    auto gridSize  = dim3((bboxLength + blockSize.x - 1) / blockSize.x, 1);

    for (auto batch = 0; batch < batchSize; ++batch)
    {
        copy_bbox<<<gridSize, blockSize, 0, stream>>>(batch, inShape[1], bBoxProposals, bBoxSelected, bBoxScores,
                                                      score_threshold, iou_threshold);
    }
}

// =============================================================================
// NonMaximumSuppression Class Definition
// =============================================================================

namespace cvcuda::priv {

NonMaximumSuppression::NonMaximumSuppression() {}

void NonMaximumSuppression::operator()(cudaStream_t stream, nvcv::ITensor &in, nvcv::ITensor &out,
                                       nvcv::ITensor &scores, float score_threshold, float iou_threshold) const
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

    if (score_threshold <= 0.0f)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Score threshold must be greater than zero");
    }

    if (iou_threshold <= 0.0f)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "IoU threshold must be greater than zero");
    }

    non_maximum_suppresion(*inData, *outData, *scoreData, score_threshold, iou_threshold, stream);
}

} // namespace cvcuda::priv
