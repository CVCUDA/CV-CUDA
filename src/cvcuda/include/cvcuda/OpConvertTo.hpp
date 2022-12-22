/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <nvcv/ITensor.hpp>
#include <nvcv/ImageFormat.hpp>
#include <nvcv/alloc/Requirements.hpp>

namespace cvcuda {

class ConvertTo final : public IOperator
{
public:
    explicit ConvertTo();

    ~ConvertTo();

    void operator()(cudaStream_t stream, nvcv::ITensor &in, nvcv::ITensor &out, const double alpha, const double beta);

    virtual NVCVOperatorHandle handle() const noexcept override;

private:
    NVCVOperatorHandle m_handle;
};

inline ConvertTo::ConvertTo()
{
    nvcv::detail::CheckThrow(cvcudaConvertToCreate(&m_handle));
    assert(m_handle);
}

inline ConvertTo::~ConvertTo()
{
    nvcvOperatorDestroy(m_handle);
    m_handle = nullptr;
}

inline void ConvertTo::operator()(cudaStream_t stream, nvcv::ITensor &in, nvcv::ITensor &out, const double alpha,
                                  const double beta)
{
    nvcv::detail::CheckThrow(cvcudaConvertToSubmit(m_handle, stream, in.handle(), out.handle(), alpha, beta));
}

inline NVCVOperatorHandle ConvertTo::handle() const noexcept
{
    return m_handle;
}

} // namespace cvcuda

#endif // CVCUDA_CONVERT_TO_HPP
