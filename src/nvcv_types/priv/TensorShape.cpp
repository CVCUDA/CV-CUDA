/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "TensorShape.hpp"

#include "Exception.hpp"
#include "TensorLayout.hpp"

namespace nvcv::priv {

void PermuteShape(const NVCVTensorLayout &srcLayout, const int64_t *srcShape, const NVCVTensorLayout &dstLayout,
                  int64_t *dstShape)
{
    if (srcShape == nullptr)
    {
        throw Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to source shape cannot be NULL");
    }

    if (dstShape == nullptr)
    {
        throw Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to destination shape cannot be NULL");
    }

    std::fill_n(dstShape, dstLayout.rank, 1);

    for (int i = 0; i < srcLayout.rank; ++i)
    {
        int dstIdx = FindDimIndex(dstLayout, srcLayout.data[i]);
        if (dstIdx >= 0)
        {
            dstShape[dstIdx] = srcShape[i];
        }
    }
}

} // namespace nvcv::priv
