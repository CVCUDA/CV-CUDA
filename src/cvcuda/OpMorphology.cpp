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

#include "priv/OpMorphology.hpp"

#include "priv/SymbolVersioning.hpp"

#include <nvcv/Exception.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>
#include <util/Assert.h>

namespace priv = cvcuda::priv;

CVCUDA_DEFINE_API(0, 2, NVCVStatus, cvcudaMorphologyCreate,
                  (NVCVOperatorHandle * handle, const int32_t maxVarShapeBatchSize))
{
    return nvcv::ProtectCall(
        [&]
        {
            if (handle == nullptr)
            {
                throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                      "Pointer to NVCVOperator handle must not be NULL");
            }

            *handle = reinterpret_cast<NVCVOperatorHandle>(new priv::Morphology(maxVarShapeBatchSize));
        });
}

CVCUDA_DEFINE_API(0, 0, NVCVStatus, cvcudaMorphologySubmit,
                  (NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle in, NVCVTensorHandle out,
                   NVCVMorphologyType morphType, int32_t maskWidth, int32_t maskHeight, int32_t anchorX,
                   int32_t anchorY, int32_t iteration, const NVCVBorderType borderMode))
{
    return nvcv::ProtectCall(
        [&]
        {
            nvcv::TensorWrapHandle input(in), output(out);
            nvcv::Size2D           maskSize = {maskWidth, maskHeight};
            int2                   anchor   = {anchorX, anchorY};
            priv::ToDynamicRef<priv::Morphology>(handle)(stream, input, output, morphType, maskSize, anchor, iteration,
                                                         borderMode);
        });
}

CVCUDA_DEFINE_API(0, 2, NVCVStatus, cvcudaMorphologyVarShapeSubmit,
                  (NVCVOperatorHandle handle, cudaStream_t stream, NVCVImageBatchHandle in, NVCVImageBatchHandle out,
                   NVCVMorphologyType morphType, NVCVTensorHandle masks, NVCVTensorHandle anchors, int32_t iteration,
                   const NVCVBorderType borderMode))
{
    return nvcv::ProtectCall(
        [&]
        {
            nvcv::ImageBatchVarShapeWrapHandle input(in), output(out);
            nvcv::TensorWrapHandle             masksWrap(masks), anchorsWrap(anchors);
            priv::ToDynamicRef<priv::Morphology>(handle)(stream, input, output, morphType, masksWrap, anchorsWrap,
                                                         iteration, borderMode);
        });
}
