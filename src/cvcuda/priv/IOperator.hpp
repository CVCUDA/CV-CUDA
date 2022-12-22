/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

/**
 * @file IOperator.hpp
 *
 * @brief Defines the private C++ Class for the operator interface.
 */

#ifndef CVCUDA_PRIV_IOPERATOR_HPP
#define CVCUDA_PRIV_IOPERATOR_HPP

#include "Version.hpp"

#include <cvcuda/Operator.h>
#include <nvcv/Exception.hpp>

namespace cvcuda::priv {

class IOperator
{
public:
    using HandleType    = NVCVOperatorHandle;
    using InterfaceType = IOperator;

    virtual ~IOperator() = default;

    HandleType handle() const
    {
        return reinterpret_cast<HandleType>(const_cast<IOperator *>(static_cast<const IOperator *>(this)));
    }

    Version version()
    {
        return CURRENT_VERSION;
    }
};

IOperator *ToOperatorPtr(void *handle);

template<class T>
inline T *ToDynamicPtr(NVCVOperatorHandle h)
{
    return dynamic_cast<T *>(ToOperatorPtr(h));
}

template<class T>
inline T &ToDynamicRef(NVCVOperatorHandle h)
{
    if (h == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Handle cannot be NULL");
    }

    if (T *child = ToDynamicPtr<T>(h))
    {
        return *child;
    }
    else
    {
        throw nvcv::Exception(nvcv::Status::ERROR_NOT_COMPATIBLE,
                              "Handle doesn't correspond to the requested object or was already destroyed.");
    }
}

} // namespace cvcuda::priv

#endif // CVCUDA_PRIV_IOPERATOR_HPP
