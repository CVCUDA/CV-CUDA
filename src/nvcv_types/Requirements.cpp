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

#include "priv/Requirements.hpp"

#include "priv/Exception.hpp"
#include "priv/Status.hpp"
#include "priv/SymbolVersioning.hpp"

#include <nvcv/alloc/Requirements.h>

namespace priv = nvcv::priv;

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvRequirementsInit, (NVCVRequirements * reqs))
{
    return priv::ProtectCall(
        [&]
        {
            if (reqs == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to requirements must not be NULL");
            }

            priv::Init(*reqs);
        });
}

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvRequirementsAdd, (NVCVRequirements * reqSum, const NVCVRequirements *req))
{
    return priv::ProtectCall(
        [&]
        {
            if (reqSum == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to summed requirements must not be NULL");
            }

            if (req == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to input requirements must not be NULL");
            }

            priv::Add(*reqSum, *req);
        });
}

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvMemRequirementsCalcTotalSizeBytes,
                (const NVCVMemRequirements *memReq, int64_t *sizeBytes))
{
    return priv::ProtectCall(
        [&]
        {
            if (memReq == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT,
                                      "Pointer to the memory requirements must not be NULL");
            }

            if (sizeBytes == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to the size output must not be NULL");
            }

            *sizeBytes = priv::CalcTotalSizeBytes(*memReq);
        });
}

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvMemRequirementsAddBuffer,
                (NVCVMemRequirements * memReq, int64_t bufSize, int64_t bufAlignment))
{
    return priv::ProtectCall(
        [&]
        {
            if (memReq == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT,
                                      "Pointer to the memory requirements must not be NULL");
            }

            priv::AddBuffer(*memReq, bufSize, bufAlignment);
        });
}
