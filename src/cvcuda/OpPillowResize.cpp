/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "cvcuda/OpPillowResize.h"

#include "priv/OpPillowResize.hpp"
#include "priv/SymbolVersioning.hpp"

#include <nvcv/Exception.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>
#include <util/Assert.h>

namespace priv = cvcuda::priv;

CVCUDA_DEFINE_API(0, 3, NVCVStatus, cvcudaPillowResizeCreate, (NVCVOperatorHandle * handle))
{
    return nvcv::ProtectCall(
        [&]
        {
            if (handle == nullptr)
            {
                throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                      "Pointer to NVCVOperator handle must not be NULL");
            }
            *handle = reinterpret_cast<NVCVOperatorHandle>(new priv::PillowResize());
        });
}

CVCUDA_DEFINE_API(0, 3, NVCVStatus, cvcudaPillowResizeGetWorkspaceRequirements,
                  (NVCVOperatorHandle handle, int maxBatchSize, int32_t maxInWidth, int32_t maxInHeight,
                   int32_t maxOutWidth, int32_t maxOutHeight, NVCVImageFormat fmt, NVCVWorkspaceRequirements *reqOut))
{
    if (!reqOut)
        return NVCV_ERROR_INVALID_ARGUMENT;

    return nvcv::ProtectCall(
        [&]
        {
            NVCVSize2D maxInSize  = {maxInWidth, maxInHeight};
            NVCVSize2D maxOutSize = {maxOutWidth, maxOutHeight};
            *reqOut = priv::ToDynamicRef<priv::PillowResize>(handle).getWorkspaceRequirements(maxBatchSize, maxInSize,
                                                                                              maxOutSize, fmt);
        });
}

CVCUDA_DEFINE_API(0, 3, NVCVStatus, cvcudaPillowResizeVarShapeGetWorkspaceRequirements,
                  (NVCVOperatorHandle handle, int batchSize, const NVCVSize2D *inputSizes,
                   const NVCVSize2D *outputSizes, NVCVImageFormat fmt, NVCVWorkspaceRequirements *reqOut))
{
    if (!reqOut)
        return NVCV_ERROR_INVALID_ARGUMENT;

    return nvcv::ProtectCall(
        [&]
        {
            *reqOut = priv::ToDynamicRef<priv::PillowResize>(handle).getWorkspaceRequirements(
                batchSize, static_cast<const nvcv::Size2D *>(inputSizes),
                static_cast<const nvcv::Size2D *>(outputSizes), fmt);
        });
}

CVCUDA_DEFINE_API(0, 3, NVCVStatus, cvcudaPillowResizeSubmit,
                  (NVCVOperatorHandle handle, cudaStream_t stream, const NVCVWorkspace *ws, NVCVTensorHandle in,
                   NVCVTensorHandle out, const NVCVInterpolationType interpolation))
{
    return nvcv::ProtectCall(
        [&]
        {
            nvcv::TensorWrapHandle input(in), output(out);
            priv::ToDynamicRef<priv::PillowResize>(handle)(stream, *ws, input, output, interpolation);
        });
}

CVCUDA_DEFINE_API(0, 3, NVCVStatus, cvcudaPillowResizeVarShapeSubmit,
                  (NVCVOperatorHandle handle, cudaStream_t stream, const NVCVWorkspace *ws, NVCVImageBatchHandle in,
                   NVCVImageBatchHandle out, const NVCVInterpolationType interpolation))
{
    return nvcv::ProtectCall(
        [&]
        {
            nvcv::ImageBatchVarShapeWrapHandle input(in), output(out);
            priv::ToDynamicRef<priv::PillowResize>(handle)(stream, *ws, input, output, interpolation);
        });
}
