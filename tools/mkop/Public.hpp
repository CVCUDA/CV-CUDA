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
 * @file Op__OPNAME__.hpp
 *
 * @brief Defines the public C++ Class for the __OPNAME__ operation.
 * @defgroup NVCV_CPP_ALGORITHM___OPNAMECAP__ __OPNAME__
 * @{
 */

#ifndef CVCUDA___OPNAMECAP___HPP
#define CVCUDA___OPNAMECAP___HPP

#include "IOperator.hpp"
#include "Op__OPNAME__.h"

#include <cuda_runtime.h>
#include <nvcv/ImageFormat.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/alloc/Requirements.hpp>

namespace cvcuda {

class __OPNAME__ final : public IOperator
{
public:
    explicit __OPNAME__();

    ~__OPNAME__();

    void operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out);

    virtual NVCVOperatorHandle handle() const noexcept override;

private:
    NVCVOperatorHandle m_handle;
};

inline __OPNAME__::__OPNAME__()
{
    nvcv::detail::CheckThrow(cvcuda__OPNAME__Create(&m_handle));
    assert(m_handle);
}

inline __OPNAME__::~__OPNAME__()
{
    nvcvOperatorDestroy(m_handle);
    m_handle = nullptr;
}

inline void __OPNAME__::operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out)
{
    nvcv::detail::CheckThrow(cvcuda__OPNAME__Submit(m_handle, stream, in.handle(), out.handle()));
}

inline NVCVOperatorHandle __OPNAME__::handle() const noexcept
{
    return m_handle;
}

} // namespace cvcuda

#endif // CVCUDA___OPNAMECAP___HPP
