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

#include "priv/Image.hpp"

#include "priv/AllocatorManager.hpp"
#include "priv/Exception.hpp"
#include "priv/IAllocator.hpp"
#include "priv/ImageManager.hpp"
#include "priv/Status.hpp"
#include "priv/SymbolVersioning.hpp"

#include <nvcv/Image.h>

namespace priv = nvcv::priv;

NVCV_DEFINE_API(0, 2, NVCVStatus, nvcvImageCalcRequirements,
                (int32_t width, int32_t height, NVCVImageFormat format, int32_t baseAlign, int32_t rowAlign,
                 NVCVImageRequirements *reqs))
{
    return priv::ProtectCall(
        [&]
        {
            if (reqs == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to output requirements must not be NULL");
            }

            *reqs = priv::Image::CalcRequirements({width, height}, priv::ImageFormat{format}, baseAlign, rowAlign);
        });
}

NVCV_DEFINE_API(0, 2, NVCVStatus, nvcvImageConstruct,
                (const NVCVImageRequirements *reqs, NVCVAllocatorHandle halloc, NVCVImageHandle *handle))
{
    return priv::ProtectCall(
        [&]
        {
            if (reqs == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to image requirements must not be NULL");
            }

            if (handle == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to output handle must not be NULL");
            }

            priv::IAllocator &alloc = priv::GetAllocator(halloc);

            *handle = priv::CreateCoreObject<priv::Image>(*reqs, alloc);
        });
}

NVCV_DEFINE_API(0, 2, NVCVStatus, nvcvImageWrapDataConstruct,
                (const NVCVImageData *data, NVCVImageDataCleanupFunc cleanup, void *ctxCleanup,
                 NVCVImageHandle *handle))
{
    return priv::ProtectCall(
        [&]
        {
            if (handle == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to output handle must not be NULL");
            }

            if (data == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Image data must not be NULL");
            }

            *handle = priv::CreateCoreObject<priv::ImageWrapData>(*data, cleanup, ctxCleanup);
        });
}

NVCV_DEFINE_API(0, 2, NVCVStatus, nvcvImageDestroy, (NVCVImageHandle handle))
{
    return priv::ProtectCall([&] { priv::DestroyCoreObject(handle); });
}

NVCV_DEFINE_API(0, 2, NVCVStatus, nvcvImageGetSize, (NVCVImageHandle handle, int32_t *width, int32_t *height))
{
    return priv::ProtectCall(
        [&]
        {
            if (width == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to output width cannot be NULL");
            }
            if (height == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to output height cannot be NULL");
            }

            auto &img = priv::ToStaticRef<const priv::IImage>(handle);

            priv::Size2D size = img.size();

            *width  = size.w;
            *height = size.h;
        });
}

NVCV_DEFINE_API(0, 2, NVCVStatus, nvcvImageGetFormat, (NVCVImageHandle handle, NVCVImageFormat *fmt))
{
    return priv::ProtectCall(
        [&]
        {
            if (fmt == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to output image format cannot be NULL");
            }

            auto &img = priv::ToStaticRef<const priv::IImage>(handle);

            *fmt = img.format().value();
        });
}

NVCV_DEFINE_API(0, 2, NVCVStatus, nvcvImageGetAllocator, (NVCVImageHandle handle, NVCVAllocatorHandle *halloc))
{
    return priv::ProtectCall(
        [&]
        {
            if (halloc == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to output allocator handle cannot be NULL");
            }

            auto &img = priv::ToStaticRef<const priv::IImage>(handle);

            *halloc = img.alloc().handle();
        });
}

NVCV_DEFINE_API(0, 2, NVCVStatus, nvcvImageGetType, (NVCVImageHandle handle, NVCVTypeImage *type))
{
    return priv::ProtectCall(
        [&]
        {
            if (type == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to output image type cannot be NULL");
            }

            auto &img = priv::ToStaticRef<const priv::IImage>(handle);

            *type = img.type();
        });
}

NVCV_DEFINE_API(0, 2, NVCVStatus, nvcvImageExportData, (NVCVImageHandle handle, NVCVImageData *data))
{
    return priv::ProtectCall(
        [&]
        {
            if (data == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to output image data cannot be NULL");
            }

            auto &img = priv::ToStaticRef<const priv::IImage>(handle);
            img.exportData(*data);
        });
}

NVCV_DEFINE_API(0, 2, NVCVStatus, nvcvImageSetUserPointer, (NVCVImageHandle handle, void *userPtr))
{
    return priv::ProtectCall(
        [&]
        {
            auto &img = priv::ToStaticRef<priv::IImage>(handle);

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

NVCV_DEFINE_API(0, 2, NVCVStatus, nvcvImageGetUserPointer, (NVCVImageHandle handle, void **outUserPtr))
{
    return priv::ProtectCall(
        [&]
        {
            if (outUserPtr == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to output user pointer cannot be NULL");
            }

            auto &img = priv::ToStaticRef<const priv::IImage>(handle);

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
