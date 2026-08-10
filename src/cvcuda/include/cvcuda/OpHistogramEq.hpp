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
 * @brief Defines the public C++ Class for the HistogramEq operation.
 * @defgroup NVCV_CPP_ALGORITHM__HISTOGRAM_EQ HistogramEq
 * @{
 */

#ifndef CVCUDA__HISTOGRAM_EQ_HPP
#define CVCUDA__HISTOGRAM_EQ_HPP

#include "IOperator.hpp"
#include "OpHistogramEq.h"

#include <cuda_runtime.h>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/ImageFormat.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/alloc/Requirements.hpp>

#include <cassert>

namespace cvcuda {

class HistogramEq final : public IOperator
{
public:
    explicit HistogramEq(uint32_t maxBatchSize);

    void operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out) const;
    void operator()(cudaStream_t stream, const nvcv::ImageBatchVarShape &in, const nvcv::ImageBatchVarShape &out) const;

    NVCVOperatorHandle handle() const noexcept override;

private:
    detail::OperatorHandle m_handle;
};

inline HistogramEq::HistogramEq(uint32_t maxBatchSize)
{
    NVCVOperatorHandle h = nullptr;
    nvcv::detail::CheckThrow(cvcudaHistogramEqCreate(&h, maxBatchSize));
    assert(h);
    m_handle = detail::OperatorHandle{h};
}

inline void HistogramEq::operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out) const
{
    nvcv::detail::CheckThrow(cvcudaHistogramEqSubmit(m_handle.get(), stream, in.handle(), out.handle()));
}

inline void HistogramEq::operator()(cudaStream_t stream, const nvcv::ImageBatchVarShape &in,
                                    const nvcv::ImageBatchVarShape &out) const
{
    nvcv::detail::CheckThrow(cvcudaHistogramEqVarShapeSubmit(m_handle.get(), stream, in.handle(), out.handle()));
}

inline NVCVOperatorHandle HistogramEq::handle() const noexcept
{
    return m_handle.get();
}

} // namespace cvcuda

/** @} */

#endif // CVCUDA__HISTOGRAM_EQ_HPP
