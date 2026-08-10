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

#include "cvcuda/OpPillowResize.h"

#include "priv/Nvtx.hpp"
#include "priv/OpPillowResize.hpp"
#include "priv/SymbolVersioning.hpp"

#include <nvcv/Exception.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/util/Assert.h>

namespace priv = cvcuda::priv;

namespace {

bool isPillowResizeInterpolationValid(NVCVInterpolationType interpolation)
{
    switch (interpolation)
    {
    case NVCV_INTERP_LINEAR:
    case NVCV_INTERP_CUBIC:
    case NVCV_INTERP_LANCZOS:
    case NVCV_INTERP_BOX:
    case NVCV_INTERP_HAMMING:
        return true;
    default:
        return false;
    }
}

} // namespace

CVCUDA_DEFINE_API(0, 3, NVCVStatus, cvcudaPillowResizeCreate, (NVCVOperatorHandle * handle))
{
    return nvcv::ProtectCall(
        [&handle]
        {
            if (handle == nullptr)
            {
                throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                      "Pointer to NVCVOperator handle must not be NULL");
            }
            *handle = priv::CreateOperatorHandle<priv::PillowResize>();
        });
}

CVCUDA_DEFINE_API(0, 3, NVCVStatus, cvcudaPillowResizeGetWorkspaceRequirements,
                  (NVCVOperatorHandle handle, int maxBatchSize, int32_t maxInWidth, int32_t maxInHeight,
                   int32_t maxOutWidth, int32_t maxOutHeight, NVCVImageFormat fmt, NVCVWorkspaceRequirements *reqOut))
{
    if (!reqOut)
        return NVCV_ERROR_INVALID_ARGUMENT;

    return nvcv::ProtectCall(
        [&maxInWidth, &maxInHeight, &maxOutWidth, &maxOutHeight, &reqOut, &handle, &maxBatchSize, &fmt]
        {
            NVCVSize2D maxInSize  = {maxInWidth, maxInHeight};
            NVCVSize2D maxOutSize = {maxOutWidth, maxOutHeight};
            *reqOut               = priv::ToDynamicRef<priv::PillowResize>(handle).getWorkspaceRequirements(
                              maxBatchSize, nvcv::Size2D{maxInSize}, nvcv::Size2D{maxOutSize}, fmt);
        });
}

CVCUDA_DEFINE_API(0, 3, NVCVStatus, cvcudaPillowResizeVarShapeGetWorkspaceRequirements,
                  (NVCVOperatorHandle handle, int batchSize, const NVCVSize2D *inputSizes,
                   const NVCVSize2D *outputSizes, NVCVImageFormat fmt, NVCVWorkspaceRequirements *reqOut))
{
    if (!inputSizes || !outputSizes || !reqOut)
        return NVCV_ERROR_INVALID_ARGUMENT;

    return nvcv::ProtectCall(
        [&reqOut, &handle, &batchSize, &inputSizes, &outputSizes, &fmt]
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
    CVCUDA_NVTX_RANGE("cvcudaPillowResizeSubmit");
    if (!isPillowResizeInterpolationValid(interpolation))
        return NVCV_ERROR_INVALID_ARGUMENT;

    if (!ws)
        return NVCV_ERROR_INVALID_ARGUMENT;

    return nvcv::ProtectCall(
        [&in, &out, &handle, &stream, &ws, &interpolation]
        {
            nvcv::TensorWrapHandle input(in);
            nvcv::TensorWrapHandle output(out);
            priv::ToDynamicRef<priv::PillowResize>(handle)(stream, *ws, input.resource(), output.resource(),
                                                           interpolation);
        });
}

CVCUDA_DEFINE_API(0, 3, NVCVStatus, cvcudaPillowResizeVarShapeSubmit,
                  (NVCVOperatorHandle handle, cudaStream_t stream, const NVCVWorkspace *ws, NVCVImageBatchHandle in,
                   NVCVImageBatchHandle out, const NVCVInterpolationType interpolation))
{
    CVCUDA_NVTX_RANGE("cvcudaPillowResizeVarShapeSubmit");
    if (!isPillowResizeInterpolationValid(interpolation))
        return NVCV_ERROR_INVALID_ARGUMENT;

    if (!ws)
        return NVCV_ERROR_INVALID_ARGUMENT;

    return nvcv::ProtectCall(
        [&in, &out, &handle, &stream, &ws, &interpolation]
        {
            nvcv::ImageBatchVarShapeWrapHandle input(in);
            nvcv::ImageBatchVarShapeWrapHandle output(out);
            priv::ToDynamicRef<priv::PillowResize>(handle)(stream, *ws, input.resource(), output.resource(),
                                                           interpolation);
        });
}
