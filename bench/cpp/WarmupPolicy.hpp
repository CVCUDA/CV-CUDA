/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef CVCUDA_WARMUP_POLICY_HPP
#define CVCUDA_WARMUP_POLICY_HPP

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

namespace benchutils {

constexpr const char *WARMUP_CAP_ENV = "CVCUDA_BENCH_WARMUP_CAP";

inline int parse_warmup_cap(std::string_view rawValue)
{
    if (rawValue.empty())
    {
        throw std::invalid_argument(std::string(WARMUP_CAP_ENV) + " must be a nonnegative decimal integer");
    }

    int cap = 0;
    for (char ch : rawValue)
    {
        if (ch < '0' || ch > '9')
        {
            throw std::invalid_argument(std::string(WARMUP_CAP_ENV) + " must be a nonnegative decimal integer");
        }

        int digit = ch - '0';
        if (cap > (std::numeric_limits<int>::max() - digit) / 10)
        {
            throw std::invalid_argument(std::string(WARMUP_CAP_ENV) + " exceeds the supported integer range");
        }
        cap = cap * 10 + digit;
    }
    return cap;
}

inline int resolve_warmup_iterations(int configuredIterations, const char *rawCap)
{
    if (configuredIterations <= 0 || rawCap == nullptr)
    {
        return configuredIterations;
    }
    return std::min(configuredIterations, parse_warmup_cap(rawCap));
}

inline int resolve_warmup_iterations(int configuredIterations)
{
    return resolve_warmup_iterations(configuredIterations, std::getenv(WARMUP_CAP_ENV));
}

} // namespace benchutils

#endif // CVCUDA_WARMUP_POLICY_HPP
