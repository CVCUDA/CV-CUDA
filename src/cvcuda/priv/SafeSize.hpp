/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef CVCUDA_PRIV_SAFE_SIZE_HPP
#define CVCUDA_PRIV_SAFE_SIZE_HPP

#include <nvcv/Exception.hpp>

#include <cstddef>
#include <initializer_list>
#include <limits>

namespace cvcuda::priv {

inline size_t CheckedMul(size_t a, size_t b, const char *message)
{
    if (b != 0 && a > std::numeric_limits<size_t>::max() / b)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "%s", message);
    }
    return a * b;
}

inline size_t CheckedMulMany(std::initializer_list<size_t> factors, const char *message)
{
    size_t result = 1;
    for (size_t factor : factors)
    {
        result = CheckedMul(result, factor, message);
    }
    return result;
}

inline size_t CheckedNonNegativeToSize(int value, const char *name)
{
    if (value < 0)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "%s must be >= 0", name);
    }
    return static_cast<size_t>(value);
}

inline size_t CheckedPositiveToSize(int value, const char *name)
{
    if (value <= 0)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "%s must be > 0", name);
    }
    return static_cast<size_t>(value);
}

} // namespace cvcuda::priv

#endif // CVCUDA_PRIV_SAFE_SIZE_HPP
