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
 * @file OpGaussian.hpp
 *
 * @brief Defines the private C++ class for the Gaussian operation.
 */

#ifndef CVCUDA_PRIV_GAUSSIAN_HPP
#define CVCUDA_PRIV_GAUSSIAN_HPP

#include "IOperator.hpp"
#include "PerDeviceResource.hpp"

#include <cuda_runtime.h>
#include <nvcv/BorderType.h>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Size.hpp>
#include <nvcv/Tensor.hpp>

#include <cstddef>
#include <mutex>
#include <unordered_map>
#include <utility>

namespace cvcuda::priv {

// Owns a cudaEvent_t. `cvcuda::util::CudaEvent` would do, but it lives in `cvcuda_util`, which the
// priv layer does not link; a handful of lines here is cheaper than widening the link graph.
class ScopedCudaEvent
{
public:
    ScopedCudaEvent() = default;

    static ScopedCudaEvent Create(unsigned flags);

    ~ScopedCudaEvent()
    {
        if (m_event != nullptr)
        {
            cudaEventDestroy(m_event);
        }
    }

    ScopedCudaEvent(ScopedCudaEvent &&other) noexcept
        : m_event(std::exchange(other.m_event, nullptr))
    {
    }

    ScopedCudaEvent &operator=(ScopedCudaEvent &&other) noexcept
    {
        std::swap(m_event, other.m_event);
        return *this;
    }

    ScopedCudaEvent(const ScopedCudaEvent &)            = delete;
    ScopedCudaEvent &operator=(const ScopedCudaEvent &) = delete;

    cudaEvent_t get() const
    {
        return m_event;
    }

private:
    cudaEvent_t m_event = nullptr;
};

// Device scratch holding the generated coefficients, plus the identity of the window currently
// resident in it. Sized once from the construction-time maximums so a submit never allocates.
//
// A window depends only on (kernelSize, sigma), and generating one is not cheap: the normalizing
// sum is a K*K loop walked by every thread, so the generator costs ~37 us for the 21x21 window a
// sigma of 2.5 asks for -- against ~517 us for the filter itself. Regenerating it per submit is
// most of that overhead, so a submit that asks for the window already in the buffer reuses it.
//
// Reuse is what makes the buffer shared state, so it carries the ordering with it, in both
// directions:
//
//   * The producing stream records m_ready, and a consumer that skipped the generation waits on it
//     before its own filter kernel can read the buffer. Without it a second stream could read a
//     window the first was still writing.
//   * A regeneration overwrites bytes that earlier submits may still be reading, so it must wait
//     for every stream that has used the buffer, not only the one that produced it. Each submit
//     records a last-use event on its own stream after its filter launch, and an eviction waits on
//     all of them. One shared event cannot express this: with two consuming streams the second
//     record would overwrite the first, leaving the earlier reader unguarded.
//
// The mutex covers only the resident-window bookkeeping and the event map, never a device sync.
class GaussianCoefficientBuffer
{
public:
    explicit GaussianCoefficientBuffer(size_t numFloats);
    ~GaussianCoefficientBuffer();

    GaussianCoefficientBuffer(const GaussianCoefficientBuffer &)            = delete;
    GaussianCoefficientBuffer &operator=(const GaussianCoefficientBuffer &) = delete;

    float *data() const
    {
        return m_data;
    }

    // Calls `generate(data())` on `stream` unless the requested window is already resident, then
    // orders `stream` behind whichever stream produced it. On return the coefficients are visible
    // to work issued on `stream` afterwards. A submit must call RecordUse after its filter launch.
    template<class Generate>
    void EnsureWindow(cudaStream_t stream, int2 kernelSize, double2 sigma, Generate &&generate)
    {
        std::scoped_lock lock(m_mutex);

        if (Resident(kernelSize, sigma))
        {
            WaitReady(stream);
            return;
        }

        // Evicting the resident window overwrites bytes that earlier submits may still be reading.
        WaitForAllUses(stream);

        generate(m_data);
        RecordReady(stream);
        m_kernelSize = kernelSize;
        m_sigma      = sigma;
        m_resident   = true;
    }

    // Marks the point on `stream` after which the buffer is no longer read by this submit. A later
    // eviction orders its regenerate behind every stream that has recorded one.
    void RecordUse(cudaStream_t stream);

private:
    bool Resident(int2 kernelSize, double2 sigma) const
    {
        return m_resident && m_kernelSize.x == kernelSize.x && m_kernelSize.y == kernelSize.y && m_sigma.x == sigma.x
            && m_sigma.y == sigma.y;
    }

    void RecordReady(cudaStream_t stream);
    void WaitReady(cudaStream_t stream);
    void WaitForAllUses(cudaStream_t stream);

    // Streams come and go; drop the ones whose last use has already retired so a process that
    // cycles through short-lived streams does not accumulate events forever.
    void PruneRetiredUses();

    static constexpr size_t kPruneThreshold = 64;

    float          *m_data = nullptr;
    // Declared before m_data is allocated so a failure part-way through construction still
    // destroys whatever was already created.
    ScopedCudaEvent m_ready;

    std::mutex m_mutex;
    bool       m_resident   = false;
    int2       m_kernelSize = {0, 0};
    double2    m_sigma      = {0.0, 0.0};

    std::unordered_map<cudaStream_t, ScopedCudaEvent> m_lastUse;
};

class Gaussian final : public IOperator
{
public:
    explicit Gaussian(nvcv::Size2D maxKernelSize, int maxBatchSize);

    void operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out, nvcv::Size2D kernelSize,
                    double2 sigma, NVCVBorderType borderMode) const;

    void operator()(cudaStream_t stream, const nvcv::ImageBatchVarShape &in, const nvcv::ImageBatchVarShape &out,
                    const nvcv::Tensor &kernelSize, const nvcv::Tensor &sigma, NVCVBorderType borderMode) const;

private:
    // Construction-time caps the operator's public contract makes observable.
    nvcv::Size2D m_maxKernelSize;
    int          m_maxBatchSize;

    mutable PerDeviceResource<GaussianCoefficientBuffer> m_coefs;
    mutable PerDeviceResource<GaussianCoefficientBuffer> m_varShapeCoefs;
};

} // namespace cvcuda::priv

#endif // CVCUDA_PRIV_GAUSSIAN_HPP
