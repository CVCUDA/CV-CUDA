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

#include "priv/OpWarpPerspective.hpp"

#include "priv/Nvtx.hpp"
#include "priv/SymbolVersioning.hpp"

#include <nvcv/Exception.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/util/Assert.h>

namespace priv = cvcuda::priv;

CVCUDA_DEFINE_API(0, 2, NVCVStatus, cvcudaWarpPerspectiveCreate,
                  (NVCVOperatorHandle * handle, const int maxVarShapeBatchSize))
{
    return nvcv::ProtectCall(
        [&handle, &maxVarShapeBatchSize]
        {
            if (handle == nullptr)
            {
                throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                      "Pointer to NVCVOperator handle must not be NULL");
            }

            *handle = priv::CreateOperatorHandle<priv::WarpPerspective>(maxVarShapeBatchSize);
        });
}

CVCUDA_DEFINE_API(0, 2, NVCVStatus, cvcudaWarpPerspectiveSubmit,
                  (NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle in, NVCVTensorHandle out,
                   const NVCVPerspectiveTransform transMatrix, const int flags, const NVCVBorderType borderMode,
                   const float4 borderValue))
{
    CVCUDA_NVTX_RANGE("cvcudaWarpPerspectiveSubmit");
    return nvcv::ProtectCall(
        [&in, &out, &handle, &stream, &transMatrix, &flags, &borderMode, &borderValue]
        {
            nvcv::TensorWrapHandle input(in);
            nvcv::TensorWrapHandle output(out);
            priv::ToDynamicRef<priv::WarpPerspective>(handle)(stream, input.resource(), output.resource(), transMatrix,
                                                              flags, borderMode, borderValue);
        });
}

CVCUDA_DEFINE_API(0, 2, NVCVStatus, cvcudaWarpPerspectiveVarShapeSubmit,
                  (NVCVOperatorHandle handle, cudaStream_t stream, NVCVImageBatchHandle in, NVCVImageBatchHandle out,
                   NVCVTensorHandle transMatrix, const int flags, const NVCVBorderType borderMode,
                   const float4 borderValue))
{
    CVCUDA_NVTX_RANGE("cvcudaWarpPerspectiveVarShapeSubmit");
    return nvcv::ProtectCall(
        [&in, &out, &transMatrix, &handle, &stream, &flags, &borderMode, &borderValue]
        {
            nvcv::ImageBatchVarShapeWrapHandle input(in);
            nvcv::ImageBatchVarShapeWrapHandle output(out);
            nvcv::TensorWrapHandle             transMatrixWrap(transMatrix);
            priv::ToDynamicRef<priv::WarpPerspective>(handle)(stream, input.resource(), output.resource(),
                                                              transMatrixWrap.resource(), flags, borderMode,
                                                              borderValue);
        });
}
