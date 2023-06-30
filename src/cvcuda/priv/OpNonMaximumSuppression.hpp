/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

/**
 * @file OpRemap.hpp
 *
 * @brief Defines the private C++ Class for the Remap operation.
 */

#ifndef CVCUDA_PRIV__NON_MAXIMUM_SUPPRESSION_HPP
#define CVCUDA_PRIV__NON_MAXIMUM_SUPPRESSION_HPP

#include "IOperator.hpp"

#include <cuda_runtime.h>
#include <nvcv/Tensor.hpp>

namespace cvcuda::priv {

class NonMaximumSuppression : public IOperator
{
public:
    explicit NonMaximumSuppression();

    /**
     * @brief Reduces the number of bounding boxes based on score and overlap.
     *
     * @param in GPU tensor, in[i, j, k] is the set of input bounding box proposals for an image where i ranges
     *           from 0 to batch-1, j ranges from 0 to number of bounding box proposals, and k == 4 (x, y, width,
     *           height) anchored at the top-left of the bounding box area
     *
     * @param out GPU tensor, out[i, j] is the output boolean mask marking the selected bounding boxes for an image
     *            where i ranges from 0 to batch-1, j ranges from 0 to the number of bounding box proposals
     *
     * @param scores GPU tensor, scores[i, j] are the associated scores for each bounding box proposal in ``in``
     *               considered during the reduce operation
     *
     * @param scoreThreshold Minimum score of a bounding box proposals
     *
     * @param iouThreshold Maximum overlap between bounding box proposals covering the same effective image region
     *                     as calculated by Intersection- over-Union (IoU)
     *
     * @param stream for the asynchronous execution.
     */
    void operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out, const nvcv::Tensor &scores,
                    float scoreThreshold, float iouThreshold) const;
};

} // namespace cvcuda::priv

#endif // CVCUDA_PRIV__NON_MAXIMUM_SUPPRESSION_HPP
