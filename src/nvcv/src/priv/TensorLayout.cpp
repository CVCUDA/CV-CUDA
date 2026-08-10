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

#include "TensorLayout.hpp"

#include "Exception.hpp"

#include <nvcv/util/Assert.h>

#include <algorithm>
#include <array>

namespace nvcv::priv {

NVCVTensorLayout CreateLayout(const char *beg, const char *end)
{
    if (beg == nullptr || end == nullptr)
    {
        throw Exception(NVCV_ERROR_INVALID_ARGUMENT) << "Range pointers must not be NULL";
    }

    if (end - beg < 0)
    {
        throw Exception(NVCV_ERROR_INVALID_ARGUMENT) << "Range must not have negative length";
    }

    if (end - beg > NVCV_TENSOR_MAX_RANK)
    {
        throw Exception(NVCV_ERROR_INVALID_ARGUMENT)
            << "Range length " << end - beg << " too large, must be <= " << NVCV_TENSOR_MAX_RANK;
    }

    NVCVTensorLayout out;
    out.rank = static_cast<int32_t>(end - beg);
    std::copy(beg, end, out.data);

    return out;
}

NVCVTensorLayout CreateLayout(const char *descr)
{
    NVCVTensorLayout out;

    if (descr == nullptr)
    {
        out = {};
    }
    else
    {
        const char *cur = descr;
        for (int i = 0; i < NVCV_TENSOR_MAX_RANK && *cur; ++i, ++cur)
        {
            out.data[i] = *cur;
        }

        if (*cur != '\0')
        {
            // Avoids going through the whole descr buffer, which might pose a
            // security hazard.
            std::array<char, 32> buf;
            int                  reqs = snprintf(buf.data(), buf.size(), "%s", descr);
            if (reqs < 0)
            {
                reqs   = 0;
                buf[0] = '\0';
            }
            throw Exception(NVCV_ERROR_INVALID_ARGUMENT)
                << "Tensor layout description is too big, must have at most 16 labels: " << buf.data()
                << (reqs <= static_cast<int>(buf.size()) - 1 ? "" : "...");
        }

        out.rank = static_cast<int32_t>(cur - descr);
        NVCV_ASSERT(0 <= out.rank && (size_t)out.rank < sizeof(out.data) / sizeof(out.data[0]));
        out.data[out.rank] = '\0'; // add null terminator
    }
    return out;
}

int FindDimIndex(const NVCVTensorLayout &layout, char dimLabel)
{
    if (const void *p = memchr(layout.data, dimLabel, layout.rank))
    {
        return static_cast<int>(
            std::distance(reinterpret_cast<const std::byte *>(layout.data), reinterpret_cast<const std::byte *>(p)));
    }
    else
    {
        return -1;
    }
}

bool IsChannelLast(const NVCVTensorLayout &layout)
{
    return layout.rank == 0 || layout.data[layout.rank - 1] == 'C';
}

NVCVTensorLayout CreateFirst(const NVCVTensorLayout &layout, int n)
{
    if (n >= 0)
    {
        NVCVTensorLayout out;
        out.rank = std::min(n, layout.rank);
        if (out.rank > NVCV_TENSOR_MAX_RANK)
            throw Exception(NVCV_ERROR_INVALID_ARGUMENT) << "Layout rank must be <= " << NVCV_TENSOR_MAX_RANK;
        std::copy(layout.data, layout.data + out.rank, out.data);
        return out;
    }
    else
    {
        return CreateLast(layout, -n);
    }
}

NVCVTensorLayout CreateLast(const NVCVTensorLayout &layout, int n)
{
    if (n >= 0)
    {
        NVCVTensorLayout out;
        out.rank = std::min(n, layout.rank);
        if (out.rank > NVCV_TENSOR_MAX_RANK)
            throw Exception(NVCV_ERROR_INVALID_ARGUMENT) << "Layout rank must be <= " << NVCV_TENSOR_MAX_RANK;
        std::copy(layout.data + layout.rank - out.rank, layout.data + layout.rank, out.data);
        return out;
    }
    else
    {
        return CreateFirst(layout, -n);
    }
}

NVCVTensorLayout CreateSubRange(const NVCVTensorLayout &layout, int beg, int end)
{
    if (beg < 0)
    {
        beg = std::max(0, layout.rank + beg);
    }
    else
    {
        beg = std::min(beg, layout.rank);
    }

    if (end < 0)
    {
        end = std::max(0, layout.rank + end);
    }
    else
    {
        end = std::min(end, layout.rank);
    }

    NVCVTensorLayout out;

    out.rank = end - beg;
    if (out.rank > 0)
    {
        if (out.rank > NVCV_TENSOR_MAX_RANK)
            throw Exception(NVCV_ERROR_INVALID_ARGUMENT) << "Layout rank must be <= " << NVCV_TENSOR_MAX_RANK;
        std::copy(layout.data + beg, layout.data + end, out.data);
    }
    else
    {
        out.rank = 0;
    }

    return out;
}

bool operator==(const NVCVTensorLayout &a, const NVCVTensorLayout &b)
{
    if (a.rank == b.rank)
    {
        return memcmp(a.data, b.data, a.rank * sizeof(a.data[0])) == 0;
    }
    else
    {
        return false;
    }
}

bool operator!=(const NVCVTensorLayout &a, const NVCVTensorLayout &b)
{
    return !operator==(a, b);
}

} // namespace nvcv::priv
