/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "cvcuda/OpHQResize.h"

#include "priv/OpHQResize.hpp"
#include "priv/SymbolVersioning.hpp"

#include <nvcv/Exception.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/util/Assert.h>

namespace priv = cvcuda::priv;

CVCUDA_DEFINE_API(0, 6, NVCVStatus, cvcudaHQResizeCreate, (NVCVOperatorHandle * handle))
{
    return nvcv::ProtectCall(
        [&]
        {
            if (handle == nullptr)
            {
                throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                      "Pointer to NVCVOperator handle must not be NULL");
            }

            *handle = reinterpret_cast<NVCVOperatorHandle>(new priv::HQResize());
        });
}

CVCUDA_DEFINE_API(0, 6, NVCVStatus, cvcudaHQResizeTensorGetWorkspaceRequirements,
                  (NVCVOperatorHandle handle, int batchSize, const HQResizeTensorShapeI inputShape,
                   const HQResizeTensorShapeI outputShape, const NVCVInterpolationType minInterpolation,
                   const NVCVInterpolationType magInterpolation, bool antialias, const HQResizeRoiF *roi,
                   NVCVWorkspaceRequirements *reqOut))
{
    if (!reqOut)
        return NVCV_ERROR_INVALID_ARGUMENT;

    return nvcv::ProtectCall(
        [&]
        {
            *reqOut = priv::ToDynamicRef<priv::HQResize>(handle).getWorkspaceRequirements(
                batchSize, inputShape, outputShape, minInterpolation, magInterpolation, antialias, roi);
        });
}

CVCUDA_DEFINE_API(0, 6, NVCVStatus, cvcudaHQResizeTensorBatchGetWorkspaceRequirements,
                  (NVCVOperatorHandle handle, int batchSize, const HQResizeTensorShapesI inputShapes,
                   const HQResizeTensorShapesI outputShapes, const NVCVInterpolationType minInterpolation,
                   const NVCVInterpolationType magInterpolation, bool antialias, const HQResizeRoisF roi,
                   NVCVWorkspaceRequirements *reqOut))
{
    if (!reqOut)
        return NVCV_ERROR_INVALID_ARGUMENT;

    return nvcv::ProtectCall(
        [&]
        {
            *reqOut = priv::ToDynamicRef<priv::HQResize>(handle).getWorkspaceRequirements(
                batchSize, inputShapes, outputShapes, minInterpolation, magInterpolation, antialias, roi);
        });
}

CVCUDA_DEFINE_API(0, 6, NVCVStatus, cvcudaHQResizeGetMaxWorkspaceRequirements,
                  (NVCVOperatorHandle handle, int maxBatchSize, const HQResizeTensorShapeI maxShape,
                   NVCVWorkspaceRequirements *reqOut))
{
    if (!reqOut)
        return NVCV_ERROR_INVALID_ARGUMENT;

    return nvcv::ProtectCall(
        [&] { *reqOut = priv::ToDynamicRef<priv::HQResize>(handle).getWorkspaceRequirements(maxBatchSize, maxShape); });
}

CVCUDA_DEFINE_API(0, 6, NVCVStatus, cvcudaHQResizeSubmit,
                  (NVCVOperatorHandle handle, cudaStream_t stream, const NVCVWorkspace *ws, NVCVTensorHandle in,
                   NVCVTensorHandle out, const NVCVInterpolationType minInterpolation,
                   const NVCVInterpolationType magInterpolation, bool antialias, const HQResizeRoiF *roi))
{
    if (!ws)
        return NVCV_ERROR_INVALID_ARGUMENT;

    return nvcv::ProtectCall(
        [&]
        {
            nvcv::TensorWrapHandle _in(in), _out(out);
            priv::ToDynamicRef<priv::HQResize>(handle)(stream, *ws, _in, _out, minInterpolation, magInterpolation,
                                                       antialias, roi);
        });
}

CVCUDA_DEFINE_API(0, 6, NVCVStatus, cvcudaHQResizeImageBatchSubmit,
                  (NVCVOperatorHandle handle, cudaStream_t stream, const NVCVWorkspace *ws, NVCVImageBatchHandle in,
                   NVCVImageBatchHandle out, const NVCVInterpolationType minInterpolation,
                   const NVCVInterpolationType magInterpolation, bool antialias, const HQResizeRoisF roi))
{
    if (!ws)
        return NVCV_ERROR_INVALID_ARGUMENT;

    return nvcv::ProtectCall(
        [&]
        {
            nvcv::ImageBatchVarShapeWrapHandle _in(in), _out(out);
            priv::ToDynamicRef<priv::HQResize>(handle)(stream, *ws, _in, _out, minInterpolation, magInterpolation,
                                                       antialias, roi);
        });
}

CVCUDA_DEFINE_API(0, 6, NVCVStatus, cvcudaHQResizeTensorBatchSubmit,
                  (NVCVOperatorHandle handle, cudaStream_t stream, const NVCVWorkspace *ws, NVCVTensorBatchHandle in,
                   NVCVTensorBatchHandle out, const NVCVInterpolationType minInterpolation,
                   const NVCVInterpolationType magInterpolation, bool antialias, const HQResizeRoisF roi))
{
    if (!ws)
        return NVCV_ERROR_INVALID_ARGUMENT;

    return nvcv::ProtectCall(
        [&]
        {
            nvcv::TensorBatchWrapHandle _in(in), _out(out);
            priv::ToDynamicRef<priv::HQResize>(handle)(stream, *ws, _in, _out, minInterpolation, magInterpolation,
                                                       antialias, roi);
        });
}
