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

/**
 * @file WarpTransformMatrix.hpp
 *
 * @brief Device scratch for the per-image warp coefficients, shared by WarpAffine and
 * WarpPerspective.
 *
 * Host-only, so the two operators' class headers can include it; the kernels that read the buffer
 * live in WarpCommon.cuh, which includes this header for the layout constant.
 */

#ifndef CVCUDA_PRIV_WARP_TRANSFORM_MATRIX_HPP
#define CVCUDA_PRIV_WARP_TRANSFORM_MATRIX_HPP

#include <cuda_runtime.h>
#include <nvcv/util/CheckError.hpp>

#include <cstdint>

namespace cvcuda::priv::warp {

// Nine floats per image even though an affine map only needs six: the var-shape kernels are shared
// between WarpAffine and WarpPerspective and always stage nine.
constexpr int kNumCoeffs = 9;

// Persistent device scratch for the per-image transform coefficients the var-shape kernels read.
// This is the allocation strategy the legacy var-shape operators used, kept as-is rather than
// moved to the workspace API or made per call: one buffer sized from the operator's maximum batch
// size, allocated at construction and freed at destruction. A non-positive maximum allocates
// nothing, and every var-shape submit is then rejected, exactly as the legacy operators behaved.
//
// Both operators hold one of these behind a PerDeviceResource, which creates one instance per CUDA
// device -- the buffer is device memory, so it cannot be shared across devices.
struct TransformMatrix
{
    explicit TransformMatrix(int32_t maxBatchSize)
    {
        if (maxBatchSize > 0)
        {
            NVCV_CHECK_THROW(cudaMalloc(&matrix, sizeof(float) * kNumCoeffs * maxBatchSize));
        }
    }

    ~TransformMatrix()
    {
        if (matrix != nullptr)
        {
            NVCV_CHECK_LOG(cudaFree(matrix));
        }
    }

    TransformMatrix(const TransformMatrix &)            = delete;
    TransformMatrix &operator=(const TransformMatrix &) = delete;

    float *matrix = nullptr;
};

} // namespace cvcuda::priv::warp

#endif // CVCUDA_PRIV_WARP_TRANSFORM_MATRIX_HPP
