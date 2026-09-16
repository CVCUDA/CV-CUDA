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
 * @file OpGaussianNoise.hpp
 *
 * @brief Defines the private C++ Class for the GaussianNoise operation.
 */

#ifndef CVCUDA_PRIV_GAUSSIAN_NOISE_HPP
#define CVCUDA_PRIV_GAUSSIAN_NOISE_HPP

#include "IOperator.hpp"
#include "PerDeviceResource.hpp"

#include <cuda_runtime.h>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>

#include <cstddef>
#include <utility>

namespace cvcuda::priv {

// Persistent per-device cuRAND state for one submission path.  The generator must continue where
// the previous call left off, so the buffer cannot be a per-call workspace: it is sized once from
// the operator's maximum batch and lives as long as the operator.  It holds one curandState per
// (sample, thread) pair, followed by an equally sized staging half that a segmented launch uses to
// publish the advanced state while every segment still reads the same starting state.  The
// pointers are untyped so <curand_kernel.h> stays inside the implementation translation unit.
struct GaussianNoiseDeviceStates
{
    explicit GaussianNoiseDeviceStates(int maxBatchSize);
    ~GaussianNoiseDeviceStates();

    GaussianNoiseDeviceStates(const GaussianNoiseDeviceStates &)            = delete;
    GaussianNoiseDeviceStates &operator=(const GaussianNoiseDeviceStates &) = delete;

    // A launcher that publishes the advanced states into the staging half can adopt them by
    // flipping the two halves instead of copying them back, which is why `allocation` -- and not
    // `states` -- is the pointer the destructor frees.
    //
    // Precondition: `nextStates` must hold a complete image of the state array, including the
    // entries belonging to threads that had no work.  The generic launch path satisfies this by
    // seeding the staging half with a device-to-device copy before the launch; a launcher that
    // skips that copy has to make its kernel carry the idle threads' states across itself.
    void Swap()
    {
        std::swap(states, nextStates);
    }

    std::byte         *allocation   = nullptr;
    std::byte         *states       = nullptr;
    std::byte         *nextStates   = nullptr;
    unsigned long long seed         = 0;
    int                maxBatchSize = 0;
    bool               setupDone    = false;
};

class GaussianNoise final : public IOperator
{
public:
    explicit GaussianNoise(int maxBatchSize);

    void operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out, const nvcv::Tensor &mu,
                    const nvcv::Tensor &sigma, bool per_channel, unsigned long long seed) const;

    void operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out, float mu, float sigma,
                    bool per_channel, unsigned long long seed, bool reseed, bool clip) const;

    void operator()(cudaStream_t stream, const nvcv::ImageBatchVarShape &in, const nvcv::ImageBatchVarShape &out,
                    const nvcv::Tensor &mu, const nvcv::Tensor &sigma, bool per_channel, unsigned long long seed) const;

private:
    mutable PerDeviceResource<GaussianNoiseDeviceStates> m_tensorStates;
    mutable PerDeviceResource<GaussianNoiseDeviceStates> m_varShapeStates;
};

} // namespace cvcuda::priv

#endif // CVCUDA_PRIV_GAUSSIAN_NOISE_HPP
