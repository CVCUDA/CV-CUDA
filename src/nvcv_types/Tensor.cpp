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

#include "priv/Tensor.hpp"

#include "priv/AllocatorManager.hpp"
#include "priv/DataType.hpp"
#include "priv/Exception.hpp"
#include "priv/IAllocator.hpp"
#include "priv/IImage.hpp"
#include "priv/ImageFormat.hpp"
#include "priv/Status.hpp"
#include "priv/SymbolVersioning.hpp"
#include "priv/TensorData.hpp"
#include "priv/TensorLayout.hpp"
#include "priv/TensorManager.hpp"
#include "priv/TensorWrapDataStrided.hpp"

#include <nvcv/Tensor.h>

#include <algorithm>

namespace priv = nvcv::priv;

NVCV_DEFINE_API(0, 2, NVCVStatus, nvcvTensorCalcRequirementsForImages,
                (int32_t batch, int32_t width, int32_t height, NVCVImageFormat format, int32_t baseAlign,
                 int32_t rowAlign, NVCVTensorRequirements *reqs))
{
    return priv::ProtectCall(
        [&]
        {
            if (reqs == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to output requirements must not be NULL");
            }

            priv::ImageFormat fmt{format};

            *reqs = priv::Tensor::CalcRequirements(batch, {width, height}, fmt, baseAlign, rowAlign);
        });
}

NVCV_DEFINE_API(0, 2, NVCVStatus, nvcvTensorCalcRequirements,
                (int32_t rank, const int64_t *shape, NVCVDataType dtype, NVCVTensorLayout layout, int32_t baseAlign,
                 int32_t rowAlign, NVCVTensorRequirements *reqs))
{
    return priv::ProtectCall(
        [&]
        {
            if (reqs == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to output requirements must not be NULL");
            }

            priv::DataType type{dtype};

            *reqs = priv::Tensor::CalcRequirements(rank, shape, type, layout, baseAlign, rowAlign);
        });
}

NVCV_DEFINE_API(0, 2, NVCVStatus, nvcvTensorConstruct,
                (const NVCVTensorRequirements *reqs, NVCVAllocatorHandle halloc, NVCVTensorHandle *handle))
{
    return priv::ProtectCall(
        [&]
        {
            if (reqs == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT,
                                      "Pointer to tensor image batch requirements must not be NULL");
            }

            if (handle == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to output handle must not be NULL");
            }

            priv::IAllocator &alloc = priv::GetAllocator(halloc);

            *handle = priv::CreateCoreObject<priv::Tensor>(*reqs, alloc);
        });
}

NVCV_DEFINE_API(0, 2, NVCVStatus, nvcvTensorWrapDataConstruct,
                (const NVCVTensorData *data, NVCVTensorDataCleanupFunc cleanup, void *ctxCleanup,
                 NVCVTensorHandle *handle))
{
    return priv::ProtectCall(
        [&]
        {
            if (data == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to image batch data must not be NULL");
            }

            if (handle == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to output handle must not be NULL");
            }

            switch (data->bufferType)
            {
            case NVCV_TENSOR_BUFFER_STRIDED_CUDA:
                *handle = priv::CreateCoreObject<priv::TensorWrapDataStrided>(*data, cleanup, ctxCleanup);
                break;

            default:
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT) << "Image buffer type not supported";
            }
        });
}

NVCV_DEFINE_API(0, 2, NVCVStatus, nvcvTensorWrapImageConstruct, (NVCVImageHandle himg, NVCVTensorHandle *handle))
{
    return priv::ProtectCall(
        [&]
        {
            if (himg == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Image handle must not be NULL");
            }

            if (handle == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to output handle must not be NULL");
            }

            auto &img = priv::ToStaticRef<priv::IImage>(himg);

            NVCVTensorData tensorData;
            FillTensorData(img, tensorData);

            *handle = priv::CreateCoreObject<priv::TensorWrapDataStrided>(tensorData, nullptr, nullptr);
        });
}

NVCV_DEFINE_API(0, 2, NVCVStatus, nvcvTensorDestroy, (NVCVTensorHandle handle))
{
    return priv::ProtectCall([&] { priv::DestroyCoreObject(handle); });
}

NVCV_DEFINE_API(0, 2, NVCVStatus, nvcvTensorGetLayout, (NVCVTensorHandle handle, NVCVTensorLayout *layout))
{
    return priv::ProtectCall(
        [&]
        {
            if (layout == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to tensor layout output cannot be NULL");
            }

            auto &tensor = priv::ToStaticRef<const priv::ITensor>(handle);

            *layout = tensor.layout();
        });
}

NVCV_DEFINE_API(0, 2, NVCVStatus, nvcvTensorGetAllocator, (NVCVTensorHandle handle, NVCVAllocatorHandle *halloc))
{
    return priv::ProtectCall(
        [&]
        {
            if (halloc == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to output allocator handle cannot be NULL");
            }

            auto &tensor = priv::ToStaticRef<const priv::ITensor>(handle);

            *halloc = tensor.alloc().handle();
        });
}

NVCV_DEFINE_API(0, 2, NVCVStatus, nvcvTensorExportData, (NVCVTensorHandle handle, NVCVTensorData *data))
{
    return priv::ProtectCall(
        [&]
        {
            if (data == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to output image batch data cannot be NULL");
            }

            auto &tensor = priv::ToStaticRef<const priv::ITensor>(handle);
            tensor.exportData(*data);
        });
}

NVCV_DEFINE_API(0, 2, NVCVStatus, nvcvTensorGetShape, (NVCVTensorHandle handle, int32_t *rank, int64_t *shape))
{
    return priv::ProtectCall(
        [&]
        {
            auto &tensor = priv::ToStaticRef<const priv::ITensor>(handle);

            if (rank == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Input pointer to rank cannot be NULL");
            }

            if (shape != nullptr)
            {
                // Number of shape elements to copy
                int n = std::min(*rank, tensor.rank());
                if (n > 0)
                {
                    if (shape == nullptr)
                    {
                        throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to shape output cannot be NULL");
                    }

                    NVCV_ASSERT(*rank - n >= 0);
                    std::fill_n(shape, *rank - n, 1);
                    std::copy_n(tensor.shape() + tensor.rank() - n, n, shape + *rank - n);
                }
            }

            *rank = tensor.rank();
        });
}

NVCV_DEFINE_API(0, 2, NVCVStatus, nvcvTensorGetDataType, (NVCVTensorHandle handle, NVCVDataType *dtype))
{
    return priv::ProtectCall(
        [&]
        {
            if (dtype == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to data type output cannot be NULL");
            }

            auto &tensor = priv::ToStaticRef<const priv::ITensor>(handle);
            *dtype       = tensor.dtype().value();
        });
}

NVCV_DEFINE_API(0, 2, NVCVStatus, nvcvTensorSetUserPointer, (NVCVTensorHandle handle, void *userPtr))
{
    return priv::ProtectCall(
        [&]
        {
            auto &tensor = priv::ToStaticRef<priv::ITensor>(handle);
            if (priv::MustProvideHiddenFunctionality(handle))
            {
                tensor.setCXXObject(userPtr);
            }
            else
            {
                tensor.setUserPointer(userPtr);
            }
        });
}

NVCV_DEFINE_API(0, 2, NVCVStatus, nvcvTensorGetUserPointer, (NVCVTensorHandle handle, void **outUserPtr))
{
    return priv::ProtectCall(
        [&]
        {
            if (outUserPtr == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to output user pointer cannot be NULL");
            }

            auto &tensor = priv::ToStaticRef<const priv::ITensor>(handle);

            if (priv::MustProvideHiddenFunctionality(handle))
            {
                tensor.getCXXObject(outUserPtr);
            }
            else
            {
                *outUserPtr = tensor.userPointer();
            }
        });
}
