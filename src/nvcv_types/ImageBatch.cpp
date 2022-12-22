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

#include "priv/AllocatorManager.hpp"
#include "priv/Exception.hpp"
#include "priv/IAllocator.hpp"
#include "priv/ImageBatchManager.hpp"
#include "priv/ImageBatchVarShape.hpp"
#include "priv/ImageFormat.hpp"
#include "priv/Status.hpp"
#include "priv/SymbolVersioning.hpp"

#include <nvcv/ImageBatch.h>

namespace priv = nvcv::priv;

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvImageBatchVarShapeCalcRequirements,
                (int32_t capacity, NVCVImageBatchVarShapeRequirements *reqs))
{
    return priv::ProtectCall(
        [&]
        {
            if (reqs == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to output requirements must not be NULL");
            }

            *reqs = priv::ImageBatchVarShape::CalcRequirements(capacity);
        });
}

NVCV_DEFINE_API(0, 2, NVCVStatus, nvcvImageBatchVarShapeConstruct,
                (const NVCVImageBatchVarShapeRequirements *reqs, NVCVAllocatorHandle halloc,
                 NVCVImageBatchHandle *handle))
{
    return priv::ProtectCall(
        [&]
        {
            if (reqs == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT,
                                      "Pointer to varshape image batch requirements must not be NULL");
            }

            if (handle == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to output handle must not be NULL");
            }

            priv::IAllocator &alloc = priv::GetAllocator(halloc);

            *handle = priv::CreateCoreObject<priv::ImageBatchVarShape>(*reqs, alloc);
        });
}

NVCV_DEFINE_API(0, 2, NVCVStatus, nvcvImageBatchDestroy, (NVCVImageBatchHandle handle))
{
    return priv::ProtectCall([&] { priv::DestroyCoreObject(handle); });
}

NVCV_DEFINE_API(0, 2, NVCVStatus, nvcvImageBatchSetUserPointer, (NVCVImageBatchHandle handle, void *userPtr))
{
    return priv::ProtectCall(
        [&]
        {
            auto &img = priv::ToStaticRef<priv::IImageBatch>(handle);
            if (priv::MustProvideHiddenFunctionality(handle))
            {
                img.setCXXObject(userPtr);
            }
            else
            {
                img.setUserPointer(userPtr);
            }
        });
}

NVCV_DEFINE_API(0, 2, NVCVStatus, nvcvImageBatchGetUserPointer, (NVCVImageBatchHandle handle, void **outUserPtr))
{
    return priv::ProtectCall(
        [&]
        {
            if (outUserPtr == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to output user pointer cannot be NULL");
            }

            auto &img = priv::ToStaticRef<const priv::IImageBatch>(handle);
            if (priv::MustProvideHiddenFunctionality(handle))
            {
                img.getCXXObject(outUserPtr);
            }
            else
            {
                *outUserPtr = img.userPointer();
            }
        });
}

NVCV_DEFINE_API(0, 2, NVCVStatus, nvcvImageBatchGetNumImages, (NVCVImageBatchHandle handle, int32_t *size))
{
    return priv::ProtectCall(
        [&]
        {
            if (size == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to output width cannot be NULL");
            }

            auto &batch = priv::ToStaticRef<const priv::IImageBatch>(handle);

            *size = batch.numImages();
        });
}

NVCV_DEFINE_API(0, 2, NVCVStatus, nvcvImageBatchGetCapacity, (NVCVImageBatchHandle handle, int32_t *capacity))
{
    return priv::ProtectCall(
        [&]
        {
            if (capacity == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to output width cannot be NULL");
            }

            auto &batch = priv::ToStaticRef<const priv::IImageBatch>(handle);

            *capacity = batch.capacity();
        });
}

NVCV_DEFINE_API(0, 2, NVCVStatus, nvcvImageBatchVarShapeGetMaxSize,
                (NVCVImageBatchHandle handle, int32_t *maxWidth, int32_t *maxHeight))
{
    return priv::ProtectCall(
        [&]
        {
            if (maxWidth == nullptr && maxHeight == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT,
                                      "Both output width and height pointers cannot be NULL");
            }

            auto &batch = priv::ToStaticRef<const priv::IImageBatchVarShape>(handle);

            priv::Size2D s = batch.maxSize();
            if (maxWidth)
            {
                *maxWidth = s.w;
            }
            if (maxHeight)
            {
                *maxHeight = s.h;
            }
        });
}

NVCV_DEFINE_API(0, 2, NVCVStatus, nvcvImageBatchVarShapeGetUniqueFormat,
                (NVCVImageBatchHandle handle, NVCVImageFormat *fmt))
{
    return priv::ProtectCall(
        [&]
        {
            if (fmt == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to output image format cannot be NULL");
            }

            auto &batch = priv::ToDynamicRef<const priv::IImageBatchVarShape>(handle);

            *fmt = batch.uniqueFormat().value();
        });
}

NVCV_DEFINE_API(0, 2, NVCVStatus, nvcvImageBatchGetAllocator,
                (NVCVImageBatchHandle handle, NVCVAllocatorHandle *halloc))
{
    return priv::ProtectCall(
        [&]
        {
            if (halloc == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to output allocator handle cannot be NULL");
            }

            auto &batch = priv::ToStaticRef<const priv::IImageBatch>(handle);

            *halloc = batch.alloc().handle();
        });
}

NVCV_DEFINE_API(0, 2, NVCVStatus, nvcvImageBatchGetType, (NVCVImageBatchHandle handle, NVCVTypeImageBatch *type))
{
    return priv::ProtectCall(
        [&]
        {
            if (type == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to output image batch type cannot be NULL");
            }

            auto &batch = priv::ToStaticRef<const priv::IImageBatch>(handle);

            *type = batch.type();
        });
}

NVCV_DEFINE_API(0, 2, NVCVStatus, nvcvImageBatchExportData,
                (NVCVImageBatchHandle handle, CUstream stream, NVCVImageBatchData *data))
{
    return priv::ProtectCall(
        [&]
        {
            if (data == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to output image batch data cannot be NULL");
            }

            auto &batch = priv::ToStaticRef<const priv::IImageBatch>(handle);
            batch.exportData(stream, *data);
        });
}

NVCV_DEFINE_API(0, 2, NVCVStatus, nvcvImageBatchVarShapePushImages,
                (NVCVImageBatchHandle handle, const NVCVImageHandle *images, int32_t numImages))
{
    return priv::ProtectCall(
        [&]
        {
            auto &batch = priv::ToDynamicRef<priv::IImageBatchVarShape>(handle);

            batch.pushImages(images, numImages);
        });
}

NVCV_DEFINE_API(0, 2, NVCVStatus, nvcvImageBatchVarShapePushImagesCallback,
                (NVCVImageBatchHandle handle, NVCVPushImageFunc cbPushImage, void *ctxCallback))
{
    return priv::ProtectCall(
        [&]
        {
            auto &batch = priv::ToDynamicRef<priv::IImageBatchVarShape>(handle);

            batch.pushImages(cbPushImage, ctxCallback);
        });
}

NVCV_DEFINE_API(0, 2, NVCVStatus, nvcvImageBatchVarShapePopImages, (NVCVImageBatchHandle handle, int32_t numImages))
{
    return priv::ProtectCall(
        [&]
        {
            auto &batch = priv::ToDynamicRef<priv::IImageBatchVarShape>(handle);

            batch.popImages(numImages);
        });
}

NVCV_DEFINE_API(0, 2, NVCVStatus, nvcvImageBatchVarShapeClear, (NVCVImageBatchHandle handle))
{
    return priv::ProtectCall(
        [&]
        {
            auto &batch = priv::ToDynamicRef<priv::IImageBatchVarShape>(handle);

            batch.clear();
        });
}

NVCV_DEFINE_API(0, 2, NVCVStatus, nvcvImageBatchVarShapeGetImages,
                (NVCVImageBatchHandle handle, int32_t begIndex, NVCVImageHandle *outImages, int32_t numImages))
{
    return priv::ProtectCall(
        [&]
        {
            auto &batch = priv::ToDynamicRef<const priv::IImageBatchVarShape>(handle);

            batch.getImages(begIndex, outImages, numImages);
        });
}
