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
 * @file OpStack.hpp
 *
 * @brief Defines the public C++ Class for the Stack operation.
 * @defgroup NVCV_CPP_ALGORITHM__STACK Stack
 * @{
 */

#ifndef CVCUDA__STACK_HPP
#define CVCUDA__STACK_HPP

#include "IOperator.hpp"
#include "OpStack.h"

#include <cuda_runtime.h>
#include <nvcv/ImageFormat.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorBatch.hpp>
#include <nvcv/alloc/Requirements.hpp>

namespace cvcuda {

class Stack final : public IOperator
{
public:
    explicit Stack();

    ~Stack();

    void operator()(cudaStream_t stream, const nvcv::TensorBatch &in, const nvcv::Tensor &out);

    virtual NVCVOperatorHandle handle() const noexcept override;

private:
    NVCVOperatorHandle m_handle;
};

inline Stack::Stack()
{
    nvcv::detail::CheckThrow(cvcudaStackCreate(&m_handle));
    assert(m_handle);
}

inline Stack::~Stack()
{
    nvcvOperatorDestroy(m_handle);
    m_handle = nullptr;
}

inline void Stack::operator()(cudaStream_t stream, const nvcv::TensorBatch &in, const nvcv::Tensor &out)
{
    nvcv::detail::CheckThrow(cvcudaStackSubmit(m_handle, stream, in.handle(), out.handle()));
}

inline NVCVOperatorHandle Stack::handle() const noexcept
{
    return m_handle;
}

} // namespace cvcuda

#endif // CVCUDA__STACK_HPP
