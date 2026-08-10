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
 * @file NonMaximumSuppression.hpp
 *
 * @brief Defines the private C++ Class for the Non-Maximum-Suppression operation.
 */

#ifndef CVCUDA__NON_MAXIMUM_SUPPRESSION_HPP
#define CVCUDA__NON_MAXIMUM_SUPPRESSION_HPP

#include "IOperator.hpp"
#include "OpNonMaximumSuppression.h"

#include <cuda_runtime.h>
#include <nvcv/Tensor.hpp>
#include <nvcv/alloc/Requirements.hpp>

#include <cassert>

namespace cvcuda {

class NonMaximumSuppression final : public IOperator
{
public:
    explicit NonMaximumSuppression();

    void operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out, const nvcv::Tensor &scores,
                    float scoreThreshold, float iouThreshold) const;

    NVCVOperatorHandle handle() const noexcept override;

private:
    detail::OperatorHandle m_handle;
};

inline NonMaximumSuppression::NonMaximumSuppression()
{
    NVCVOperatorHandle h = nullptr;
    nvcv::detail::CheckThrow(cvcudaNonMaximumSuppressionCreate(&h));
    assert(h);
    m_handle = detail::OperatorHandle{h};
}

inline void NonMaximumSuppression::operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out,
                                              const nvcv::Tensor &scores, float scoreThreshold,
                                              float iouThreshold) const
{
    nvcv::detail::CheckThrow(cvcudaNonMaximumSuppressionSubmit(m_handle.get(), stream, in.handle(), out.handle(),
                                                               scores.handle(), scoreThreshold, iouThreshold));
}

inline NVCVOperatorHandle NonMaximumSuppression::handle() const noexcept
{
    return m_handle.get();
}

} // namespace cvcuda

#endif // CVCUDA__NON_MAXIMUM_SUPPRESSION_HPP
