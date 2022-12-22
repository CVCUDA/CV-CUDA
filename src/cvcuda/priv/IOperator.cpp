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

#include "IOperator.hpp"

#include <sstream>

namespace cvcuda::priv {

IOperator *ToOperatorPtr(void *handle)
{
    // First cast to the operator interface, this must always succeed.
    if (IOperator *op = reinterpret_cast<IOperator *>(handle))
    {
        // If major version are the same,
        if (op->version().major() == CURRENT_VERSION.major())
        {
            return op;
        }
        else
        {
            std::ostringstream ss;
            ss << "Object version " << op->version() << " not compatible with NVCV OP version " << CURRENT_VERSION;
            throw nvcv::Exception(nvcv::Status::ERROR_NOT_COMPATIBLE, "%s", ss.str().c_str());
        }
    }
    else
    {
        return nullptr;
    }
}

} // namespace cvcuda::priv
