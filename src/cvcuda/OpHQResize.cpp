/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "priv/Nvtx.hpp"
#include "priv/OpHQResize.hpp"
#include "priv/SymbolVersioning.hpp"

#include <nvcv/Exception.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/util/Assert.h>

#include <cmath>

namespace priv = cvcuda::priv;

namespace {

bool roiIsFinite(const HQResizeRoiF &roi)
{
    for (int i = 0; i < NVCV_HQ_RESIZE_MAX_RESIZED_NDIM; i++)
    {
        if (!std::isfinite(roi.lo[i]) || !std::isfinite(roi.hi[i]))
            return false;
    }
    return true;
}

bool roisAreFinite(const HQResizeRoisF &rois)
{
    if (rois.roi == nullptr)
        return true;
    for (int i = 0; i < rois.size; i++)
    {
        for (int j = 0; j < rois.ndim; j++)
        {
            if (!std::isfinite(rois.roi[i].lo[j]) || !std::isfinite(rois.roi[i].hi[j]))
                return false;
        }
    }
    return true;
}

} // namespace

CVCUDA_DEFINE_API(0, 6, NVCVStatus, cvcudaHQResizeCreate, (NVCVOperatorHandle * handle))
{
    return nvcv::ProtectCall(
        [&handle]
        {
            if (handle == nullptr)
            {
                throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                      "Pointer to NVCVOperator handle must not be NULL");
            }

            *handle = priv::CreateOperatorHandle<priv::HQResize>();
        });
}

CVCUDA_DEFINE_API(0, 6, NVCVStatus, cvcudaHQResizeTensorGetWorkspaceRequirements,
                  (NVCVOperatorHandle handle, int batchSize, const HQResizeTensorShapeI inputShape,
                   const HQResizeTensorShapeI outputShape, const NVCVInterpolationType minInterpolation,
                   const NVCVInterpolationType magInterpolation, bool antialias, const HQResizeRoiF *roi,
                   NVCVWorkspaceRequirements *reqOut))
{
    return nvcv::ProtectCall(
        [&reqOut, &roi, &handle, &batchSize, &inputShape, &outputShape, &minInterpolation, &magInterpolation,
         &antialias]
        {
            if (reqOut == nullptr)
            {
                throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                      "Pointer to output workspace requirements must not be NULL");
            }
            if (roi != nullptr && !roiIsFinite(*roi))
            {
                throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "ROI coordinates must be finite");
            }

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
    return nvcv::ProtectCall(
        [&reqOut, &roi, &handle, &batchSize, &inputShapes, &outputShapes, &minInterpolation, &magInterpolation,
         &antialias]
        {
            if (reqOut == nullptr)
            {
                throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                      "Pointer to output workspace requirements must not be NULL");
            }
            if (!roisAreFinite(roi))
            {
                throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "ROI coordinates must be finite");
            }

            *reqOut = priv::ToDynamicRef<priv::HQResize>(handle).getWorkspaceRequirements(
                batchSize, inputShapes, outputShapes, minInterpolation, magInterpolation, antialias, roi);
        });
}

CVCUDA_DEFINE_API(0, 6, NVCVStatus, cvcudaHQResizeGetMaxWorkspaceRequirements,
                  (NVCVOperatorHandle handle, int maxBatchSize, const HQResizeTensorShapeI maxShape,
                   NVCVWorkspaceRequirements *reqOut))
{
    return nvcv::ProtectCall(
        [&reqOut, &handle, &maxBatchSize, &maxShape]
        {
            if (reqOut == nullptr)
            {
                throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                      "Pointer to output workspace requirements must not be NULL");
            }

            *reqOut = priv::ToDynamicRef<priv::HQResize>(handle).getWorkspaceRequirements(maxBatchSize, maxShape);
        });
}

CVCUDA_DEFINE_API(0, 6, NVCVStatus, cvcudaHQResizeSubmit,
                  (NVCVOperatorHandle handle, cudaStream_t stream, const NVCVWorkspace *ws, NVCVTensorHandle in,
                   NVCVTensorHandle out, const NVCVInterpolationType minInterpolation,
                   const NVCVInterpolationType magInterpolation, bool antialias, const HQResizeRoiF *roi))
{
    CVCUDA_NVTX_RANGE("cvcudaHQResizeSubmit");
    return nvcv::ProtectCall(
        [&ws, &roi, &in, &out, &handle, &stream, &minInterpolation, &magInterpolation, &antialias]
        {
            if (ws == nullptr)
            {
                throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Pointer to workspace must not be NULL");
            }
            if (roi != nullptr && !roiIsFinite(*roi))
            {
                throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "ROI coordinates must be finite");
            }

            nvcv::TensorWrapHandle _in(in);

            nvcv::TensorWrapHandle _out(out);
            priv::ToDynamicRef<priv::HQResize>(handle)(stream, *ws, _in.resource(), _out.resource(), minInterpolation,
                                                       magInterpolation, antialias, roi);
        });
}

CVCUDA_DEFINE_API(0, 6, NVCVStatus, cvcudaHQResizeImageBatchSubmit,
                  (NVCVOperatorHandle handle, cudaStream_t stream, const NVCVWorkspace *ws, NVCVImageBatchHandle in,
                   NVCVImageBatchHandle out, const NVCVInterpolationType minInterpolation,
                   const NVCVInterpolationType magInterpolation, bool antialias, const HQResizeRoisF roi))
{
    CVCUDA_NVTX_RANGE("cvcudaHQResizeImageBatchSubmit");
    return nvcv::ProtectCall(
        [&ws, &roi, &in, &out, &handle, &stream, &minInterpolation, &magInterpolation, &antialias]
        {
            if (ws == nullptr)
            {
                throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Pointer to workspace must not be NULL");
            }
            if (!roisAreFinite(roi))
            {
                throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "ROI coordinates must be finite");
            }

            nvcv::ImageBatchVarShapeWrapHandle _in(in);

            nvcv::ImageBatchVarShapeWrapHandle _out(out);
            priv::ToDynamicRef<priv::HQResize>(handle)(stream, *ws, _in.resource(), _out.resource(), minInterpolation,
                                                       magInterpolation, antialias, roi);
        });
}

CVCUDA_DEFINE_API(0, 6, NVCVStatus, cvcudaHQResizeTensorBatchSubmit,
                  (NVCVOperatorHandle handle, cudaStream_t stream, const NVCVWorkspace *ws, NVCVTensorBatchHandle in,
                   NVCVTensorBatchHandle out, const NVCVInterpolationType minInterpolation,
                   const NVCVInterpolationType magInterpolation, bool antialias, const HQResizeRoisF roi))
{
    CVCUDA_NVTX_RANGE("cvcudaHQResizeTensorBatchSubmit");
    return nvcv::ProtectCall(
        [&ws, &roi, &in, &out, &handle, &stream, &minInterpolation, &magInterpolation, &antialias]
        {
            if (ws == nullptr)
            {
                throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Pointer to workspace must not be NULL");
            }
            if (!roisAreFinite(roi))
            {
                throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "ROI coordinates must be finite");
            }

            nvcv::TensorBatchWrapHandle _in(in);

            nvcv::TensorBatchWrapHandle _out(out);
            priv::ToDynamicRef<priv::HQResize>(handle)(stream, *ws, _in.resource(), _out.resource(), minInterpolation,
                                                       magInterpolation, antialias, roi);
        });
}
