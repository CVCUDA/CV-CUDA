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
 * @file NonMaximumSuppression.hpp
 *
 * @brief Defines the private C++ Class for the Non-Maximum-Suppression operation.
 */

#ifndef CVCUDA__NON_MAXIMUM_SUPPRESSION_HPP
#define CVCUDA__NON_MAXIMUM_SUPPRESSION_HPP

#include "IOperator.hpp"
#include "OpNonMaximumSuppression.h"

#include <cuda_runtime.h>
#include <nvcv/ITensor.hpp>
#include <nvcv/alloc/Requirements.hpp>

namespace cvcuda {

class NonMaximumSuppression final : public IOperator
{
public:
    explicit NonMaximumSuppression();

    ~NonMaximumSuppression();

    void operator()(cudaStream_t stream, nvcv::ITensor &in, nvcv::ITensor &out, nvcv::ITensor &scores,
                    float score_threshold, float iou_threshold);

    virtual NVCVOperatorHandle handle() const noexcept override;

private:
    NVCVOperatorHandle m_handle;
};

inline NonMaximumSuppression::NonMaximumSuppression()
{
    nvcv::detail::CheckThrow(cvcudaNonMaximumSuppressionCreate(&m_handle));
    assert(m_handle);
}

inline NonMaximumSuppression::~NonMaximumSuppression()
{
    nvcvOperatorDestroy(m_handle);
}

inline void NonMaximumSuppression::operator()(cudaStream_t stream, nvcv::ITensor &in, nvcv::ITensor &out,
                                              nvcv::ITensor &scores, float score_threshold, float iou_threshold)
{
    nvcv::detail::CheckThrow(cvcudaNonMaximumSuppressionSubmit(m_handle, stream, in.handle(), out.handle(),
                                                               scores.handle(), score_threshold, iou_threshold));
}

inline NVCVOperatorHandle NonMaximumSuppression::handle() const noexcept
{
    return m_handle;
}

} // namespace cvcuda

#endif // CVCUDA__NON_MAXIMUM_SUPPRESSION_HPP
