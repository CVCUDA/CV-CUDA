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
 * @file FindHomography.hpp
 *
 * @brief Defines the private C++ Class for the FindHomography operation.
 */

#ifndef CVCUDA_PRIV__FIND_HOMOGRAPHY_HPP
#define CVCUDA_PRIV__FIND_HOMOGRAPHY_HPP
#include "IOperator.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cvcuda/OpFindHomography.h>
#include <library_types.h>
#include <nvcv/Array.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorBatch.hpp>

typedef struct
{
    float2 *srcMean;
    float2 *dstMean;
    float2 *srcShiftSum;
    float2 *dstShiftSum;
    float  *LtL;
    float  *W;
    float  *r;
    float  *J;
    float  *calc_buffer;
} BufferOffsets;

typedef struct
{
    int               *cusolverInfo;
    float             *cusolverBuffer;
    cusolverDnHandle_t cusolverH;
    syevjInfo_t        syevj_params;
    int                lwork;
} cuSolver;

namespace cvcuda::priv {

class FindHomography final : public IOperator
{
public:
    explicit FindHomography(int batchSize, int numPoints);
    ~FindHomography();
    void operator()(cudaStream_t stream, const nvcv::Tensor &src, const nvcv::Tensor &dst,
                    const nvcv::Tensor &models) const;
    void operator()(cudaStream_t stream, const nvcv::TensorBatch &src, const nvcv::TensorBatch &dst,
                    const nvcv::TensorBatch &models) const;

private:
    BufferOffsets bufferOffset;
    cuSolver      cusolverData;
};

} // namespace cvcuda::priv

#endif // CVCUDA_PRIV__FIND_HOMOGRAPHY_HPP
