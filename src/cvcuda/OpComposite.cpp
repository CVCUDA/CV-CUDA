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

#include "priv/OpComposite.hpp"

#include "priv/Nvtx.hpp"
#include "priv/SymbolVersioning.hpp"

#include <nvcv/Exception.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/util/Assert.h>

namespace priv = cvcuda::priv;

CVCUDA_DEFINE_API(0, 2, NVCVStatus, cvcudaCompositeCreate, (NVCVOperatorHandle * handle))
{
    return nvcv::ProtectCall(
        [&handle]
        {
            if (handle == nullptr)
            {
                throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                      "Pointer to NVCVOperator handle must not be NULL");
            }

            *handle = priv::CreateOperatorHandle<priv::Composite>();
        });
}

CVCUDA_DEFINE_API(0, 2, NVCVStatus, cvcudaCompositeSubmit,
                  (NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle fg, NVCVTensorHandle bg,
                   NVCVTensorHandle fgMask, NVCVTensorHandle out))
{
    CVCUDA_NVTX_RANGE("cvcudaCompositeSubmit");
    return nvcv::ProtectCall(
        [&fg, &bg, &fgMask, &out, &handle, &stream]
        {
            nvcv::TensorWrapHandle foreground(fg);
            nvcv::TensorWrapHandle background(bg);
            nvcv::TensorWrapHandle mask(fgMask);
            nvcv::TensorWrapHandle output(out);
            priv::ToDynamicRef<priv::Composite>(handle)(stream, foreground.resource(), background.resource(),
                                                        mask.resource(), output.resource());
        });
}

CVCUDA_DEFINE_API(0, 2, NVCVStatus, cvcudaCompositeVarShapeSubmit,
                  (NVCVOperatorHandle handle, cudaStream_t stream, NVCVImageBatchHandle fg, NVCVImageBatchHandle bg,
                   NVCVImageBatchHandle fgMask, NVCVImageBatchHandle out))
{
    CVCUDA_NVTX_RANGE("cvcudaCompositeVarShapeSubmit");
    return nvcv::ProtectCall(
        [&fg, &bg, &fgMask, &out, &handle, &stream]
        {
            nvcv::ImageBatchVarShapeWrapHandle foreground(fg);
            nvcv::ImageBatchVarShapeWrapHandle background(bg);
            nvcv::ImageBatchVarShapeWrapHandle mask(fgMask);
            nvcv::ImageBatchVarShapeWrapHandle output(out);
            priv::ToDynamicRef<priv::Composite>(handle)(stream, foreground.resource(), background.resource(),
                                                        mask.resource(), output.resource());
        });
}
