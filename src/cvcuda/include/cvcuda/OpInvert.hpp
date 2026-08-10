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
 * @file OpInvert.hpp
 *
 * @brief Defines the public C++ Class for the Invert operation.
 * @defgroup NVCV_CPP_ALGORITHM__INVERT Invert
 * @{
 */

#ifndef CVCUDA__INVERT_HPP
#define CVCUDA__INVERT_HPP

#include "IOperator.hpp"
#include "OpInvert.h"

#include <cuda_runtime.h>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/ImageFormat.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/alloc/Requirements.hpp>

namespace cvcuda {

class Invert final : public IOperator
{
public:
    explicit Invert();

    void operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out) const;

    void operator()(cudaStream_t stream, const nvcv::ImageBatch &in, const nvcv::ImageBatch &out) const;

    NVCVOperatorHandle handle() const noexcept override;

private:
    detail::OperatorHandle m_handle;
};

inline Invert::Invert()
{
    NVCVOperatorHandle h = nullptr;
    nvcv::detail::CheckThrow(cvcudaInvertCreate(&h));
    assert(h);
    m_handle = detail::OperatorHandle{h};
}

inline void Invert::operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out) const
{
    nvcv::detail::CheckThrow(cvcudaInvertSubmit(m_handle.get(), stream, in.handle(), out.handle()));
}

inline void Invert::operator()(cudaStream_t stream, const nvcv::ImageBatch &in, const nvcv::ImageBatch &out) const
{
    nvcv::detail::CheckThrow(cvcudaInvertVarShapeSubmit(m_handle.get(), stream, in.handle(), out.handle()));
}

inline NVCVOperatorHandle Invert::handle() const noexcept
{
    return m_handle.get();
}

} // namespace cvcuda

#endif // CVCUDA__INVERT_HPP
