/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "priv/Types.hpp"

#include "priv/SymbolVersioning.hpp"

#include <cuda_runtime.h>
#include <cvcuda/Types.h>
#include <nvcv/Exception.hpp>

#include <memory>
#include <vector>

namespace priv = cvcuda::priv;

namespace {

bool isPODElementType(NVCVOSDType type)
{
    switch (type)
    {
    case NVCV_OSD_RECT:
    case NVCV_OSD_POINT:
    case NVCV_OSD_LINE:
    case NVCV_OSD_ROTATED_RECT:
    case NVCV_OSD_CIRCLE:
    case NVCV_OSD_ARROW:
        return true;
    default:
        return false;
    }
}

} // namespace

CVCUDA_DEFINE_API(0, 16, NVCVStatus, nvcvBndBoxesIConstruct,
                  (NVCVBndBoxesI * handle, const NVCVBndBoxI *boxes, const int32_t *numBoxesPerBatch,
                   int32_t batchSize))
{
    return nvcv::ProtectCall(
        [&handle, &boxes, &numBoxesPerBatch, &batchSize]
        {
            if (handle == nullptr)
            {
                throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Output handle pointer must not be NULL");
            }
            if (batchSize < 0)
            {
                throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "batchSize must be >= 0");
            }
            if (batchSize > 0 && numBoxesPerBatch == nullptr)
            {
                throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                      "numBoxesPerBatch must not be NULL when batchSize > 0");
            }

            std::vector<std::vector<NVCVBndBoxI>> vec(batchSize);
            const NVCVBndBoxI                    *cursor = boxes;
            for (int32_t b = 0; b < batchSize; ++b)
            {
                int32_t n = numBoxesPerBatch[b];
                if (n < 0)
                {
                    throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Per-batch box counts must be >= 0");
                }
                if (n > 0)
                {
                    if (cursor == nullptr)
                    {
                        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                              "boxes must not be NULL when any batch has boxes");
                    }
                    vec[b].assign(cursor, cursor + n);
                    cursor += n;
                }
            }

            auto impl = std::make_unique<priv::NVCVBndBoxesImpl>(vec);
            *handle   = static_cast<NVCVBndBoxesI>(static_cast<void *>(impl.release()));
        });
}

CVCUDA_DEFINE_API(0, 16, NVCVStatus, nvcvBndBoxesIDestroy, (NVCVBndBoxesI handle))
{
    return nvcv::ProtectCall(
        [&handle]
        {
            std::unique_ptr<priv::NVCVBndBoxesImpl> impl(
                static_cast<priv::NVCVBndBoxesImpl *>(static_cast<void *>(handle)));
            (void)impl;
        });
}

CVCUDA_DEFINE_API(0, 16, NVCVStatus, nvcvBlurBoxesIConstruct,
                  (NVCVBlurBoxesI * handle, const NVCVBlurBoxI *boxes, const int32_t *numBoxesPerBatch,
                   int32_t batchSize))
{
    return nvcv::ProtectCall(
        [&handle, &boxes, &numBoxesPerBatch, &batchSize]
        {
            if (handle == nullptr)
            {
                throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Output handle pointer must not be NULL");
            }
            if (batchSize < 0)
            {
                throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "batchSize must be >= 0");
            }
            if (batchSize > 0 && numBoxesPerBatch == nullptr)
            {
                throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                      "numBoxesPerBatch must not be NULL when batchSize > 0");
            }

            std::vector<std::vector<NVCVBlurBoxI>> vec(batchSize);
            const NVCVBlurBoxI                    *cursor = boxes;
            for (int32_t b = 0; b < batchSize; ++b)
            {
                int32_t n = numBoxesPerBatch[b];
                if (n < 0)
                {
                    throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Per-batch box counts must be >= 0");
                }
                if (n > 0)
                {
                    if (cursor == nullptr)
                    {
                        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                              "boxes must not be NULL when any batch has boxes");
                    }
                    vec[b].assign(cursor, cursor + n);
                    cursor += n;
                }
            }

            auto impl = std::make_unique<priv::NVCVBlurBoxesImpl>(vec);
            *handle   = static_cast<NVCVBlurBoxesI>(static_cast<void *>(impl.release()));
        });
}

CVCUDA_DEFINE_API(0, 16, NVCVStatus, nvcvBlurBoxesIDestroy, (NVCVBlurBoxesI handle))
{
    return nvcv::ProtectCall(
        [&handle]
        {
            std::unique_ptr<priv::NVCVBlurBoxesImpl> impl(
                static_cast<priv::NVCVBlurBoxesImpl *>(static_cast<void *>(handle)));
            (void)impl;
        });
}

CVCUDA_DEFINE_API(0, 16, NVCVStatus, nvcvElementsConstruct,
                  (NVCVElements * handle, const NVCVOSDType *types, const void *const *payloads,
                   const int32_t *numElementsPerBatch, int32_t batchSize))
{
    return nvcv::ProtectCall(
        [&handle, &types, &payloads, &numElementsPerBatch, &batchSize]
        {
            if (handle == nullptr)
            {
                throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Output handle pointer must not be NULL");
            }
            if (batchSize < 0)
            {
                throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "batchSize must be >= 0");
            }
            if (batchSize > 0 && numElementsPerBatch == nullptr)
            {
                throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                      "numElementsPerBatch must not be NULL when batchSize > 0");
            }

            std::vector<std::vector<std::shared_ptr<priv::NVCVElement>>> vec(batchSize);
            int32_t                                                      flatIdx = 0;
            for (int32_t b = 0; b < batchSize; ++b)
            {
                int32_t n = numElementsPerBatch[b];
                if (n < 0)
                {
                    throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                          "Per-batch element counts must be >= 0");
                }
                if (n > 0 && (types == nullptr || payloads == nullptr))
                {
                    throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                          "types and payloads must not be NULL when any batch has elements");
                }
                vec[b].reserve(n);
                for (int32_t i = 0; i < n; ++i, ++flatIdx)
                {
                    NVCVOSDType t = types[flatIdx];
                    if (!isPODElementType(t))
                    {
                        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                              "Only POD OSD element types are supported by this API");
                    }
                    if (payloads[flatIdx] == nullptr)
                    {
                        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Element payload must not be NULL");
                    }
                    const void *payload = payloads[flatIdx];
                    switch (t)
                    {
                    case NVCV_OSD_RECT:
                        vec[b].push_back(
                            std::make_shared<priv::NVCVElement>(t, static_cast<const NVCVBndBoxI *>(payload)));
                        break;
                    case NVCV_OSD_POINT:
                        vec[b].push_back(
                            std::make_shared<priv::NVCVElement>(t, static_cast<const NVCVPoint *>(payload)));
                        break;
                    case NVCV_OSD_LINE:
                        vec[b].push_back(
                            std::make_shared<priv::NVCVElement>(t, static_cast<const NVCVLine *>(payload)));
                        break;
                    case NVCV_OSD_ROTATED_RECT:
                        vec[b].push_back(
                            std::make_shared<priv::NVCVElement>(t, static_cast<const NVCVRotatedBox *>(payload)));
                        break;
                    case NVCV_OSD_CIRCLE:
                        vec[b].push_back(
                            std::make_shared<priv::NVCVElement>(t, static_cast<const NVCVCircle *>(payload)));
                        break;
                    case NVCV_OSD_ARROW:
                        vec[b].push_back(
                            std::make_shared<priv::NVCVElement>(t, static_cast<const NVCVArrow *>(payload)));
                        break;
                    default:
                        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                              "Only POD OSD element types are supported by this API");
                    }
                }
            }

            auto impl = std::make_unique<priv::NVCVElementsImpl>(vec);
            *handle   = static_cast<NVCVElements>(static_cast<void *>(impl.release()));
        });
}

CVCUDA_DEFINE_API(0, 16, NVCVStatus, nvcvElementsDestroy, (NVCVElements handle))
{
    return nvcv::ProtectCall(
        [&handle]
        {
            std::unique_ptr<priv::NVCVElementsImpl> impl(
                static_cast<priv::NVCVElementsImpl *>(static_cast<void *>(handle)));
            (void)impl;
        });
}
