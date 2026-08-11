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

#include "priv/OpRotate.hpp"

#include "priv/Nvtx.hpp"
#include "priv/SymbolVersioning.hpp"

#include <nvcv/Exception.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/util/Assert.h>

namespace priv = cvcuda::priv;

CVCUDA_DEFINE_API(0, 2, NVCVStatus, cvcudaRotateCreate,
                  (NVCVOperatorHandle * handle, const int32_t maxVarShapeBatchSize))
{
    return nvcv::ProtectCall(
        [&handle, &maxVarShapeBatchSize]
        {
            if (handle == nullptr)
            {
                throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                      "Pointer to NVCVOperator handle must not be NULL");
            }

            *handle = priv::CreateOperatorHandle<priv::Rotate>(maxVarShapeBatchSize);
        });
}

CVCUDA_DEFINE_API(0, 2, NVCVStatus, cvcudaRotateSubmit,
                  (NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle in, NVCVTensorHandle out,
                   const double angleDeg, const double2 shift, const NVCVInterpolationType interpolation))
{
    CVCUDA_NVTX_RANGE("cvcudaRotateSubmit");
    return nvcv::ProtectCall(
        [&in, &out, &handle, &stream, &angleDeg, &shift, &interpolation]
        {
            nvcv::TensorWrapHandle input(in);
            nvcv::TensorWrapHandle output(out);
            priv::ToDynamicRef<priv::Rotate>(handle)(stream, input.resource(), output.resource(), angleDeg, shift,
                                                     interpolation);
        });
}

CVCUDA_DEFINE_API(0, 2, NVCVStatus, cvcudaRotateVarShapeSubmit,
                  (NVCVOperatorHandle handle, cudaStream_t stream, NVCVImageBatchHandle in, NVCVImageBatchHandle out,
                   NVCVTensorHandle angleDeg, NVCVTensorHandle shift, const NVCVInterpolationType interpolation))
{
    CVCUDA_NVTX_RANGE("cvcudaRotateVarShapeSubmit");
    return nvcv::ProtectCall(
        [&in, &out, &angleDeg, &shift, &handle, &stream, &interpolation]
        {
            nvcv::ImageBatchVarShapeWrapHandle input(in);
            nvcv::ImageBatchVarShapeWrapHandle output(out);
            nvcv::TensorWrapHandle             angleDegWrap(angleDeg);
            nvcv::TensorWrapHandle             shiftWrap(shift);
            priv::ToDynamicRef<priv::Rotate>(handle)(stream, input.resource(), output.resource(),
                                                     angleDegWrap.resource(), shiftWrap.resource(), interpolation);
        });
}
