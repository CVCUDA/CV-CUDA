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
 * @file OpThreshold.hpp
 *
 * @brief Defines the public C++ Class for the threshold operation.
 * @defgroup NVCV_CPP_ALGORITHM_THRESHOLD Threshold
 * @{
 */

#ifndef CVCUDA_THRESHOLD_HPP
#define CVCUDA_THRESHOLD_HPP

#include "IOperator.hpp"
#include "OpThreshold.h"

#include <cuda_runtime.h>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/ImageFormat.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/alloc/Requirements.hpp>

#include <cassert>

namespace cvcuda {

class Threshold final : public IOperator
{
public:
    explicit Threshold(uint32_t type, int32_t maxBatchSize);

    void operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out, const nvcv::Tensor &thresh,
                    const nvcv::Tensor &maxval) const;

    void operator()(cudaStream_t stream, const nvcv::ImageBatchVarShape &in, const nvcv::ImageBatchVarShape &out,
                    const nvcv::Tensor &thresh, const nvcv::Tensor &maxval) const;

    NVCVOperatorHandle handle() const noexcept override;

private:
    detail::OperatorHandle m_handle;
};

inline Threshold::Threshold(uint32_t type, int32_t maxBatchSize)
{
    NVCVOperatorHandle h = nullptr;
    nvcv::detail::CheckThrow(cvcudaThresholdCreate(&h, type, maxBatchSize));
    assert(h);
    m_handle = detail::OperatorHandle{h};
}

inline void Threshold::operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out,
                                  const nvcv::Tensor &thresh, const nvcv::Tensor &maxval) const
{
    nvcv::detail::CheckThrow(
        cvcudaThresholdSubmit(m_handle.get(), stream, in.handle(), out.handle(), thresh.handle(), maxval.handle()));
}

inline void Threshold::operator()(cudaStream_t stream, const nvcv::ImageBatchVarShape &in,
                                  const nvcv::ImageBatchVarShape &out, const nvcv::Tensor &thresh,
                                  const nvcv::Tensor &maxval) const
{
    nvcv::detail::CheckThrow(cvcudaThresholdVarShapeSubmit(m_handle.get(), stream, in.handle(), out.handle(),
                                                           thresh.handle(), maxval.handle()));
}

inline NVCVOperatorHandle Threshold::handle() const noexcept
{
    return m_handle.get();
}

} // namespace cvcuda

/** @} */

#endif // CVCUDA_THRESHOLD_HPP
