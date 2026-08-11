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
 * @file IOperator.hpp
 *
 * @brief Defines the public C++ interface to operator interfaces.
 */

#ifndef CVCUDA_IOPERATOR_HPP
#define CVCUDA_IOPERATOR_HPP

#include "Operator.h"

namespace cvcuda {

namespace detail {

// Move-only RAII wrapper around an NVCVOperatorHandle. Owning operator
// wrappers hold one of these so they don't need to write any rule-of-five
// boilerplate of their own — copy is deleted, move transfers the handle,
// destruction calls nvcvOperatorDestroy exactly once.
class OperatorHandle
{
public:
    OperatorHandle() noexcept = default;

    explicit OperatorHandle(NVCVOperatorHandle h) noexcept
        : m_handle{h}
    {
    }

    ~OperatorHandle()
    {
        nvcvOperatorDestroy(m_handle);
    }

    OperatorHandle(const OperatorHandle &)            = delete;
    OperatorHandle &operator=(const OperatorHandle &) = delete;

    OperatorHandle(OperatorHandle &&that) noexcept
        : m_handle{that.m_handle}
    {
        that.m_handle = nullptr;
    }

    OperatorHandle &operator=(OperatorHandle &&that) noexcept
    {
        if (this != &that)
        {
            nvcvOperatorDestroy(m_handle);
            m_handle      = that.m_handle;
            that.m_handle = nullptr;
        }
        return *this;
    }

    NVCVOperatorHandle get() const noexcept
    {
        return m_handle;
    }

private:
    NVCVOperatorHandle m_handle = nullptr;
};

} // namespace detail

class IOperator
{
public:
    IOperator()                             = default;
    virtual ~IOperator()                    = default;
    IOperator(const IOperator &)            = delete;
    IOperator &operator=(const IOperator &) = delete;
    // Defaulted move is correct only because IOperator carries no state; if
    // data members are added here, revisit move semantics so subclasses don't
    // silently inherit a wrong default.
    IOperator(IOperator &&) noexcept            = default;
    IOperator &operator=(IOperator &&) noexcept = default;

    virtual NVCVOperatorHandle handle() const noexcept = 0;
};

} // namespace cvcuda

#endif // CVCUDA_IOPERATOR_HPP
