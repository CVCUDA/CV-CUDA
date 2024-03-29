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
 * @file OpHistogram.hpp
 *
 * @brief Defines the public C++ Class for the Histogram operation.
 * @defgroup NVCV_CPP_ALGORITHM__HISTOGRAM Histogram
 * @{
 */

#ifndef CVCUDA__HISTOGRAM_HPP
#define CVCUDA__HISTOGRAM_HPP

#include "IOperator.hpp"
#include "OpHistogram.h"

#include <cuda_runtime.h>
#include <nvcv/ImageFormat.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/alloc/Requirements.hpp>

namespace cvcuda {

class Histogram final : public IOperator
{
public:
    explicit Histogram();

    ~Histogram();

    void operator()(cudaStream_t stream, const nvcv::Tensor &in, nvcv::OptionalTensorConstRef mask,
                    const nvcv::Tensor &histogram);

    virtual NVCVOperatorHandle handle() const noexcept override;

private:
    NVCVOperatorHandle m_handle;
};

inline Histogram::Histogram()
{
    nvcv::detail::CheckThrow(cvcudaHistogramCreate(&m_handle));
    assert(m_handle);
}

inline Histogram::~Histogram()
{
    nvcvOperatorDestroy(m_handle);
    m_handle = nullptr;
}

inline void Histogram::operator()(cudaStream_t stream, const nvcv::Tensor &in, nvcv::OptionalTensorConstRef mask,
                                  const nvcv::Tensor &histogram)
{
    nvcv::detail::CheckThrow(
        cvcudaHistogramSubmit(m_handle, stream, in.handle(), NVCV_OPTIONAL_TO_HANDLE(mask), histogram.handle()));
}

inline NVCVOperatorHandle Histogram::handle() const noexcept
{
    return m_handle;
}

} // namespace cvcuda

#endif // CVCUDA__HISTOGRAM_HPP
