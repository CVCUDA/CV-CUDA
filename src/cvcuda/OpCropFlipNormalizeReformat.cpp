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

#include "priv/OpCropFlipNormalizeReformat.hpp"

#include "priv/Nvtx.hpp"
#include "priv/SymbolVersioning.hpp"

#include <nvcv/Exception.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/util/Assert.h>

namespace priv = cvcuda::priv;

CVCUDA_DEFINE_API(0, 3, NVCVStatus, cvcudaCropFlipNormalizeReformatCreate, (NVCVOperatorHandle * handle))
{
    return nvcv::ProtectCall(
        [&handle]
        {
            if (handle == nullptr)
            {
                throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                      "Pointer to NVCVOperator handle must not be NULL");
            }

            *handle = priv::CreateOperatorHandle<priv::CropFlipNormalizeReformat>();
        });
}

CVCUDA_DEFINE_API(0, 3, NVCVStatus, cvcudaCropFlipNormalizeReformatSubmit,
                  (NVCVOperatorHandle handle, cudaStream_t stream, NVCVImageBatchHandle in, NVCVTensorHandle out,
                   NVCVTensorHandle cropRect, NVCVBorderType borderMode, float borderValue, NVCVTensorHandle flipCode,
                   NVCVTensorHandle base, NVCVTensorHandle scale, float global_scale, float shift, float epsilon,
                   uint32_t flags))
{
    CVCUDA_NVTX_RANGE("cvcudaCropFlipNormalizeReformatSubmit");
    return nvcv::ProtectCall(
        [&base, &scale, &flipCode, &cropRect, &out, &in, &handle, &stream, &borderMode, &borderValue, &global_scale,
         &shift, &epsilon, &flags]
        {
            nvcv::TensorWrapHandle             baseWrap(base);
            nvcv::TensorWrapHandle             scaleWrap(scale);
            nvcv::TensorWrapHandle             flipCodeWrap(flipCode);
            nvcv::TensorWrapHandle             cropRectWrap(cropRect);
            nvcv::TensorWrapHandle             outWrap(out);
            nvcv::ImageBatchVarShapeWrapHandle inWrap(in);
            priv::ToDynamicRef<priv::CropFlipNormalizeReformat>(handle)(
                stream, inWrap.resource(), outWrap.resource(), cropRectWrap.resource(), borderMode, borderValue,
                flipCodeWrap.resource(), baseWrap.resource(), scaleWrap.resource(), global_scale, shift, epsilon,
                flags);
        });
}
