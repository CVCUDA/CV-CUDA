/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * @file OpHistogramEq.hpp
 *
 * @brief Defines the private C++ Class for the HistogramEq operation.
 */

#ifndef CVCUDA_PRIV__HISTOGRAM_EQ_HPP
#define CVCUDA_PRIV__HISTOGRAM_EQ_HPP

#include "IOperator.hpp"
#include "PerDeviceResource.hpp"

#include <cuda_runtime.h>
#include <cvcuda/Types.h>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>

#include <cstddef>
#include <cstdint>

namespace cvcuda::priv {

// Per-image bin counts, then the normalized CDF, live in a device buffer that is allocated once and
// reused by every submit -- sized for the operator's maximum batch and the maximum channel count, so
// no call has to size or allocate anything. Held behind PerDeviceResource, which makes the
// allocation lazy per CUDA device and binds each buffer to the device it was allocated on.
class HistogramEqScratch
{
public:
    explicit HistogramEqScratch(int32_t maxBatchSize);
    ~HistogramEqScratch();

    HistogramEqScratch(const HistogramEqScratch &)            = delete;
    HistogramEqScratch &operator=(const HistogramEqScratch &) = delete;

    int32_t *data() const noexcept
    {
        return m_data;
    }

    size_t sizeBytes() const noexcept
    {
        return m_sizeBytes;
    }

private:
    int32_t *m_data      = nullptr;
    size_t   m_sizeBytes = 0;
};

class HistogramEq final : public IOperator
{
public:
    explicit HistogramEq(uint32_t maxBatchSize);

    void operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out,
                    NVCVHistogramEqMethod method) const;

    void operator()(cudaStream_t stream, const nvcv::ImageBatchVarShape &in, const nvcv::ImageBatchVarShape &out,
                    NVCVHistogramEqMethod method) const;

private:
    int32_t                                       m_maxBatchSize;
    mutable PerDeviceResource<HistogramEqScratch> m_scratch;
    mutable PerDeviceResource<HistogramEqScratch> m_scratchVarShape;
};

} // namespace cvcuda::priv

#endif // CVCUDA_PRIV__HISTOGRAM_EQ_HPP
