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

#include "OpFindContours.hpp"

#include "legacy/CvCudaLegacy.h"
#include "legacy/CvCudaLegacyHelpers.hpp"

#include <nvcv/Exception.hpp>
#include <util/CheckError.hpp>

namespace cvcuda::priv {

namespace legacy = nvcv::legacy::cuda_op;

FindContours::FindContours(nvcv::Size2D maxSize, int maxBatchSize)
{
    legacy::DataShape maxIn, maxOut;
    // maxIn/maxOut not used by op.
    maxIn.N = maxBatchSize;
    maxIn.C = 1;
    maxIn.H = maxSize.h;
    maxIn.W = maxSize.w;

    m_legacyOp = std::make_unique<legacy::FindContours>(maxIn, maxOut);
}

void FindContours::operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &points,
                              const nvcv::Tensor &numPoints) const
{
    auto inData = in.exportData<nvcv::TensorDataStridedCuda>();
    if (inData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input must be cuda-accessible, pitch-linear tensor");
    }

    auto pointCoords = points.exportData<nvcv::TensorDataStridedCuda>();
    if (pointCoords == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Output must be cuda-accessible, pitch-linear tensor");
    }

    auto pointCounts = numPoints.exportData<nvcv::TensorDataStridedCuda>();
    if (pointCounts == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Output must be cuda-accessible, pitch-linear tensor");
    }

    NVCV_CHECK_THROW(m_legacyOp->infer(*inData, *pointCoords, *pointCounts, stream));
}

} // namespace cvcuda::priv
