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
 * @file OpReformat.hpp
 *
 * @brief Defines the public C++ Class for the reformat operation.
 * @defgroup NVCV_CPP_ALGORITHM_REFORMAT Reformat
 * @{
 */

#ifndef CVCUDA_REFORMAT_HPP
#define CVCUDA_REFORMAT_HPP

#include "IOperator.hpp"
#include "OpReformat.h"

#include <cuda_runtime.h>
#include <nvcv/ImageFormat.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/alloc/Requirements.hpp>

#include <cassert>

namespace cvcuda {

class Reformat final : public IOperator
{
public:
    explicit Reformat();

    void operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out) const;

    NVCVOperatorHandle handle() const noexcept override;

private:
    detail::OperatorHandle m_handle;
};

inline Reformat::Reformat()
{
    NVCVOperatorHandle h = nullptr;
    nvcv::detail::CheckThrow(cvcudaReformatCreate(&h));
    assert(h);
    m_handle = detail::OperatorHandle{h};
}

inline void Reformat::operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out) const
{
    nvcv::detail::CheckThrow(cvcudaReformatSubmit(m_handle.get(), stream, in.handle(), out.handle()));
}

inline NVCVOperatorHandle Reformat::handle() const noexcept
{
    return m_handle.get();
}

} // namespace cvcuda

/** @} */

#endif // CVCUDA_REFORMAT_HPP
