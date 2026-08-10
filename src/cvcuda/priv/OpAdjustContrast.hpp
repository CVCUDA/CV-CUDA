/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * @file OpAdjustContrast.hpp
 *
 * @brief Defines the private C++ Class for the AdjustContrast operation.
 */

#ifndef CVCUDA_PRIV__ADJUST_CONTRAST_HPP
#define CVCUDA_PRIV__ADJUST_CONTRAST_HPP

#include "IOperator.hpp"
#include "PerDeviceResource.hpp"

#include <cuda_runtime.h>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>

#include <cstddef>
#include <mutex>

namespace cvcuda::priv {

// Bounded reduction workspace, retained per device and grown on demand. Host submissions are
// serialized while the completion event orders asynchronous reuse across CUDA streams.
class AdjustContrastWorkspace
{
public:
    AdjustContrastWorkspace();
    ~AdjustContrastWorkspace();

    AdjustContrastWorkspace(const AdjustContrastWorkspace &)            = delete;
    AdjustContrastWorkspace &operator=(const AdjustContrastWorkspace &) = delete;

    [[nodiscard]] std::unique_lock<std::mutex> lock();

    void *acquire(size_t sizeBytes, cudaStream_t stream);
    void  release(cudaStream_t stream);
    void  releaseNoThrow(cudaStream_t stream) noexcept;

private:
    std::mutex  m_mutex;
    void       *m_data     = nullptr;
    size_t      m_capacity = 0; // bytes
    cudaEvent_t m_ready    = nullptr;
    bool        m_pending  = false;
};

class AdjustContrast final : public IOperator
{
public:
    explicit AdjustContrast();

    void operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out, double contrastFactor) const;

    void operator()(cudaStream_t stream, const nvcv::ImageBatchVarShape &in, const nvcv::ImageBatchVarShape &out,
                    double contrastFactor) const;

private:
    mutable PerDeviceResource<AdjustContrastWorkspace> m_workspace;
};

} // namespace cvcuda::priv

#endif // CVCUDA_PRIV__ADJUST_CONTRAST_HPP
