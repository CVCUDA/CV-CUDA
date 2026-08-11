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

#include "priv/OpInpaint.hpp"

#include "priv/Nvtx.hpp"
#include "priv/SymbolVersioning.hpp"

#include <nvcv/Exception.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/util/Assert.h>

namespace priv = cvcuda::priv;

CVCUDA_DEFINE_API(0, 4, NVCVStatus, cvcudaInpaintCreate,
                  (NVCVOperatorHandle * handle, int32_t maxBatchSize, int32_t maxHeight, int32_t maxWidth))
{
    return nvcv::ProtectCall(
        [&handle, &maxBatchSize, &maxWidth, &maxHeight]
        {
            if (handle == nullptr)
            {
                throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                      "Pointer to NVCVOperator handle must not be NULL");
            }
            if (maxBatchSize <= 0)
            {
                throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "maxBatchSize must be > 0");
            }
            if (maxHeight <= 0 || maxWidth <= 0)
            {
                throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "maxHeight and maxWidth must be > 0");
            }

            *handle = priv::CreateOperatorHandle<priv::Inpaint>(maxBatchSize, nvcv::Size2D{maxWidth, maxHeight});
        });
}

CVCUDA_DEFINE_API(0, 4, NVCVStatus, cvcudaInpaintSubmit,
                  (NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle in, NVCVTensorHandle masks,
                   NVCVTensorHandle out, double inpaintRadius))
{
    CVCUDA_NVTX_RANGE("cvcudaInpaintSubmit");
    return nvcv::ProtectCall(
        [&in, &out, &masks, &handle, &stream, &inpaintRadius]
        {
            nvcv::TensorWrapHandle input(in);
            nvcv::TensorWrapHandle output(out);
            nvcv::TensorWrapHandle maskswrap(masks);
            priv::ToDynamicRef<priv::Inpaint>(handle)(stream, input.resource(), maskswrap.resource(), output.resource(),
                                                      inpaintRadius);
        });
}

CVCUDA_DEFINE_API(0, 4, NVCVStatus, cvcudaInpaintVarShapeSubmit,
                  (NVCVOperatorHandle handle, cudaStream_t stream, NVCVImageBatchHandle in, NVCVImageBatchHandle masks,
                   NVCVImageBatchHandle out, double inpaintRadius))
{
    CVCUDA_NVTX_RANGE("cvcudaInpaintVarShapeSubmit");
    return nvcv::ProtectCall(
        [&in, &out, &masks, &handle, &stream, &inpaintRadius]
        {
            nvcv::ImageBatchVarShapeWrapHandle input(in);
            nvcv::ImageBatchVarShapeWrapHandle output(out);
            nvcv::ImageBatchVarShapeWrapHandle maskswrap(masks);
            priv::ToDynamicRef<priv::Inpaint>(handle)(stream, input.resource(), maskswrap.resource(), output.resource(),
                                                      inpaintRadius);
        });
}
