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
 * @brief Defines the public C++ Class for the AdjustContrast operation.
 * @defgroup NVCV_CPP_ALGORITHM_ADJUST_CONTRAST Adjust Contrast
 * @{
 */

#ifndef CVCUDA__ADJUST_CONTRAST_HPP
#define CVCUDA__ADJUST_CONTRAST_HPP

#include "IOperator.hpp"
#include "OpAdjustContrast.h"

#include <cuda_runtime.h>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>

#include <cassert>

namespace cvcuda {

class AdjustContrast final : public IOperator
{
public:
    explicit AdjustContrast();

    void operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out, double contrastFactor) const;

    void operator()(cudaStream_t stream, const nvcv::ImageBatch &in, const nvcv::ImageBatch &out,
                    double contrastFactor) const;

    NVCVOperatorHandle handle() const noexcept override;

private:
    detail::OperatorHandle m_handle;
};

inline AdjustContrast::AdjustContrast()
{
    NVCVOperatorHandle h = nullptr;
    nvcv::detail::CheckThrow(cvcudaAdjustContrastCreate(&h));
    assert(h);
    m_handle = detail::OperatorHandle{h};
}

inline void AdjustContrast::operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out,
                                       double contrastFactor) const
{
    nvcv::detail::CheckThrow(
        cvcudaAdjustContrastSubmit(m_handle.get(), stream, in.handle(), out.handle(), contrastFactor));
}

inline void AdjustContrast::operator()(cudaStream_t stream, const nvcv::ImageBatch &in, const nvcv::ImageBatch &out,
                                       double contrastFactor) const
{
    nvcv::detail::CheckThrow(
        cvcudaAdjustContrastVarShapeSubmit(m_handle.get(), stream, in.handle(), out.handle(), contrastFactor));
}

inline NVCVOperatorHandle AdjustContrast::handle() const noexcept
{
    return m_handle.get();
}

} // namespace cvcuda

/** @} */

#endif // CVCUDA__ADJUST_CONTRAST_HPP
