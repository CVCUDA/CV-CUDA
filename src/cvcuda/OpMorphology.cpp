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

#include "priv/OpMorphology.hpp"

#include "priv/Nvtx.hpp"
#include "priv/SymbolVersioning.hpp"

#include <nvcv/Exception.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/util/Assert.h>

namespace priv = cvcuda::priv;

CVCUDA_DEFINE_API(0, 4, NVCVStatus, cvcudaMorphologyCreate, (NVCVOperatorHandle * handle))
{
    return nvcv::ProtectCall(
        [&handle]
        {
            if (handle == nullptr)
            {
                throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                      "Pointer to NVCVOperator handle must not be NULL");
            }

            *handle = priv::CreateOperatorHandle<priv::Morphology>();
        });
}

CVCUDA_DEFINE_API(0, 4, NVCVStatus, cvcudaMorphologySubmit,
                  (NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle in, NVCVTensorHandle out,
                   NVCVTensorHandle workspace, NVCVMorphologyType morphType, int32_t maskWidth, int32_t maskHeight,
                   int32_t anchorX, int32_t anchorY, int32_t iteration, const NVCVBorderType borderMode))
{
    CVCUDA_NVTX_RANGE("cvcudaMorphologySubmit");
    return nvcv::ProtectCall(
        [&in, &out, &maskWidth, &maskHeight, &anchorX, &anchorY, &handle, &stream, &workspace, &morphType, &iteration,
         &borderMode]
        {
            nvcv::TensorWrapHandle input(in);
            nvcv::TensorWrapHandle output(out);
            nvcv::Size2D           maskSize = {maskWidth, maskHeight};
            int2                   anchor   = {anchorX, anchorY};
            priv::ToDynamicRef<priv::Morphology>(handle)(stream, input.resource(), output.resource(),
                                                         NVCV_TENSOR_HANDLE_TO_OPTIONAL(workspace), morphType, maskSize,
                                                         anchor, iteration, borderMode);
        });
}

CVCUDA_DEFINE_API(0, 4, NVCVStatus, cvcudaMorphologyVarShapeSubmit,
                  (NVCVOperatorHandle handle, cudaStream_t stream, NVCVImageBatchHandle in, NVCVImageBatchHandle out,
                   NVCVImageBatchHandle workspace, NVCVMorphologyType morphType, NVCVTensorHandle masks,
                   NVCVTensorHandle anchors, int32_t iteration, const NVCVBorderType borderMode))
{
    CVCUDA_NVTX_RANGE("cvcudaMorphologyVarShapeSubmit");
    return nvcv::ProtectCall(
        [&in, &out, &masks, &anchors, &handle, &stream, &workspace, &morphType, &iteration, &borderMode]
        {
            nvcv::ImageBatchVarShapeWrapHandle input(in);
            nvcv::ImageBatchVarShapeWrapHandle output(out);
            nvcv::TensorWrapHandle             masksWrap(masks);
            nvcv::TensorWrapHandle             anchorsWrap(anchors);
            priv::ToDynamicRef<priv::Morphology>(handle)(
                stream, input.resource(), output.resource(), NVCV_IMAGE_BATCH_VAR_SHAPE_HANDLE_TO_OPTIONAL(workspace),
                morphType, masksWrap.resource(), anchorsWrap.resource(), iteration, borderMode);
        });
}
