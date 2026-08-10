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
 * @file OpConverTo.hpp
 *
 * @brief Defines the public C++ Class for the ConvertTo operation.
 * @defgroup NVCV_CPP_ALGORITHM_CONVERT_TO ConvertTo
 * @{
 */

#ifndef CVCUDA_CONVERT_TO_HPP
#define CVCUDA_CONVERT_TO_HPP

#include "IOperator.hpp"
#include "OpConvertTo.h"

#include <cuda_runtime.h>
#include <nvcv/ImageFormat.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/alloc/Requirements.hpp>

#include <cassert>

namespace cvcuda {

class ConvertTo final : public IOperator
{
public:
    explicit ConvertTo();

    void operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out, const double alpha,
                    const double beta, NVCVRoundMode roundMode = NVCV_ROUND_NEAREST) const;

    NVCVOperatorHandle handle() const noexcept override;

private:
    detail::OperatorHandle m_handle;
};

inline ConvertTo::ConvertTo()
{
    NVCVOperatorHandle h = nullptr;
    nvcv::detail::CheckThrow(cvcudaConvertToCreate(&h));
    assert(h);
    m_handle = detail::OperatorHandle{h};
}

inline void ConvertTo::operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out,
                                  const double alpha, const double beta, NVCVRoundMode roundMode) const
{
    nvcv::detail::CheckThrow(
        cvcudaConvertToSubmit(m_handle.get(), stream, in.handle(), out.handle(), alpha, beta, roundMode));
}

inline NVCVOperatorHandle ConvertTo::handle() const noexcept
{
    return m_handle.get();
}

} // namespace cvcuda

/** @} */

#endif // CVCUDA_CONVERT_TO_HPP
