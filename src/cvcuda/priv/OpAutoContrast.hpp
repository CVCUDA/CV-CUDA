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
 * @file OpAutoContrast.hpp
 *
 * @brief Defines the private C++ Class for the AutoContrast operation.
 */

#ifndef CVCUDA_PRIV__AUTO_CONTRAST_HPP
#define CVCUDA_PRIV__AUTO_CONTRAST_HPP

#include "IOperator.hpp"
#include "PerDeviceResource.hpp"

#include <cuda_runtime.h>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>

#include <cstddef>
#include <cstdint>
#include <mutex>

namespace cvcuda::priv {

namespace detail {

constexpr int64_t AutoContrastDivUp(int64_t value, int64_t divisor)
{
    return value / divisor + (value % divisor != 0 ? 1 : 0);
}

constexpr int64_t AutoContrastGridX(int64_t width, int blockWidth, int xSteps, int numPlanes = 1)
{
    return AutoContrastDivUp(width, static_cast<int64_t>(blockWidth) * xSteps) * numPlanes;
}

} // namespace detail

// Bounded per-CTA partial and final per-(sample, channel) extrema workspace, grown on demand.
// Host submissions are serialized while the event orders asynchronous buffer reuse across
// streams. PerDeviceResource gives each GPU an independent workspace and ordering state.
class AutoContrastWorkspace
{
public:
    AutoContrastWorkspace();
    ~AutoContrastWorkspace();

    AutoContrastWorkspace(const AutoContrastWorkspace &)            = delete;
    AutoContrastWorkspace &operator=(const AutoContrastWorkspace &) = delete;

    [[nodiscard]] std::unique_lock<std::mutex> lock();

    // Returns 2*count floats for paired low/high partial and final extrema storage.
    float *acquire(size_t count, cudaStream_t stream);
    void   release(cudaStream_t stream);
    void   releaseNoThrow(cudaStream_t stream) noexcept;

private:
    std::mutex  m_mutex;
    float      *m_data     = nullptr;
    size_t      m_capacity = 0; // capacity in floats
    cudaEvent_t m_ready    = nullptr;
    bool        m_pending  = false;
};

class AutoContrast final : public IOperator
{
public:
    explicit AutoContrast();

    void operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out) const;

    void operator()(cudaStream_t stream, const nvcv::ImageBatchVarShape &in, const nvcv::ImageBatchVarShape &out) const;

private:
    mutable PerDeviceResource<AutoContrastWorkspace> m_workspace;
};

} // namespace cvcuda::priv

#endif // CVCUDA_PRIV__AUTO_CONTRAST_HPP
