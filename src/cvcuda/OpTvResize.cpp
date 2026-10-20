/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "priv/OpTvResize.hpp"

#include "priv/Nvtx.hpp"
#include "priv/SymbolVersioning.hpp"

#include <cvcuda/OpTvResize.h>
#include <nvcv/Exception.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>

namespace priv = cvcuda::priv;

CVCUDA_DEFINE_API(0, 0, NVCVStatus, cvcudaTvResizeCreate, (NVCVOperatorHandle * handle))
{
    return nvcv::ProtectCall(
        [handle]
        {
            if (!handle)
                throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Null operator output");
            *handle = priv::CreateOperatorHandle<priv::TvResize>();
        });
}

CVCUDA_DEFINE_API(0, 0, NVCVStatus, cvcudaTvResizeTensorGetWorkspaceRequirements,
                  (NVCVOperatorHandle handle, int batchSize, int32_t srcHeight, int32_t srcWidth, int32_t dstHeight,
                   int32_t dstWidth, int32_t numChannels, NVCVInterpolationType interpolation, bool antialias,
                   NVCVWorkspaceRequirements *reqOut))
{
    return nvcv::ProtectCall(
        [handle, batchSize, srcHeight, srcWidth, dstHeight, dstWidth, numChannels, interpolation, antialias, reqOut]
        {
            if (!reqOut)
                throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Null workspace requirements output");
            *reqOut = priv::ToDynamicRef<priv::TvResize>(handle).getWorkspaceRequirements(
                batchSize, srcHeight, srcWidth, dstHeight, dstWidth, numChannels, interpolation, antialias);
        });
}

CVCUDA_DEFINE_API(0, 0, NVCVStatus, cvcudaTvResizeImageBatchGetWorkspaceRequirements,
                  (NVCVOperatorHandle handle, int batchSize, const int32_t *srcHeights, const int32_t *srcWidths,
                   const int32_t *dstHeights, const int32_t *dstWidths, int32_t numChannels,
                   NVCVInterpolationType interpolation, bool antialias, NVCVWorkspaceRequirements *reqOut))
{
    return nvcv::ProtectCall(
        [handle, batchSize, srcHeights, srcWidths, dstHeights, dstWidths, numChannels, interpolation, antialias, reqOut]
        {
            if (!reqOut)
                throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Null workspace requirements output");
            *reqOut = priv::ToDynamicRef<priv::TvResize>(handle).getWorkspaceRequirements(
                batchSize, srcHeights, srcWidths, dstHeights, dstWidths, numChannels, interpolation, antialias);
        });
}

CVCUDA_DEFINE_API(0, 0, NVCVStatus, cvcudaTvResizeSubmit,
                  (NVCVOperatorHandle handle, cudaStream_t stream, const NVCVWorkspace *workspace, NVCVTensorHandle in,
                   NVCVTensorHandle out, NVCVInterpolationType interpolation, bool antialias))
{
    CVCUDA_NVTX_RANGE("cvcudaTvResizeSubmit");
    return nvcv::ProtectCall(
        [handle, stream, workspace, in, out, interpolation, antialias]
        {
            const NVCVWorkspace    empty{};
            nvcv::TensorWrapHandle input(in);
            nvcv::TensorWrapHandle output(out);
            priv::ToDynamicRef<priv::TvResize>(handle)(stream, workspace ? *workspace : empty, input.resource(),
                                                       output.resource(), interpolation, antialias);
        });
}

CVCUDA_DEFINE_API(0, 0, NVCVStatus, cvcudaTvResizeImageBatchSubmit,
                  (NVCVOperatorHandle handle, cudaStream_t stream, const NVCVWorkspace *workspace,
                   NVCVImageBatchHandle in, NVCVImageBatchHandle out, NVCVInterpolationType interpolation,
                   bool antialias))
{
    CVCUDA_NVTX_RANGE("cvcudaTvResizeImageBatchSubmit");
    return nvcv::ProtectCall(
        [handle, stream, workspace, in, out, interpolation, antialias]
        {
            const NVCVWorkspace                empty{};
            nvcv::ImageBatchVarShapeWrapHandle input(in);
            nvcv::ImageBatchVarShapeWrapHandle output(out);
            priv::ToDynamicRef<priv::TvResize>(handle)(stream, workspace ? *workspace : empty, input.resource(),
                                                       output.resource(), interpolation, antialias);
        });
}
