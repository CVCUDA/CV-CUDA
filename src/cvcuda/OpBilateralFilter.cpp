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

#include "priv/OpBilateralFilter.hpp"

#include "priv/Nvtx.hpp"
#include "priv/SymbolVersioning.hpp"

#include <nvcv/Exception.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/util/Assert.h>

namespace priv = cvcuda::priv;

CVCUDA_DEFINE_API(0, 2, NVCVStatus, cvcudaBilateralFilterCreate, (NVCVOperatorHandle * handle))
{
    return nvcv::ProtectCall(
        [&handle]
        {
            if (handle == nullptr)
            {
                throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                      "Pointer to NVCVOperator handle must not be NULL");
            }

            *handle = priv::CreateOperatorHandle<priv::BilateralFilter>();
        });
}

CVCUDA_DEFINE_API(0, 2, NVCVStatus, cvcudaBilateralFilterSubmit,
                  (NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle in, NVCVTensorHandle out,
                   int diameter, float sigmaColor, float sigmaSpace, NVCVBorderType borderMode))
{
    CVCUDA_NVTX_RANGE("cvcudaBilateralFilterSubmit");
    return nvcv::ProtectCall(
        [&in, &out, &handle, &stream, &diameter, &sigmaColor, &sigmaSpace, &borderMode]
        {
            nvcv::TensorWrapHandle input(in);
            nvcv::TensorWrapHandle output(out);
            priv::ToDynamicRef<priv::BilateralFilter>(handle)(stream, input.resource(), output.resource(), diameter,
                                                              sigmaColor, sigmaSpace, borderMode);
        });
}

CVCUDA_DEFINE_API(0, 2, NVCVStatus, cvcudaBilateralFilterVarShapeSubmit,
                  (NVCVOperatorHandle handle, cudaStream_t stream, NVCVImageBatchHandle in, NVCVImageBatchHandle out,
                   NVCVTensorHandle diameter, NVCVTensorHandle sigmaColor, NVCVTensorHandle sigmaSpace,
                   NVCVBorderType borderMode))
{
    CVCUDA_NVTX_RANGE("cvcudaBilateralFilterVarShapeSubmit");
    return nvcv::ProtectCall(
        [&in, &out, &diameter, &sigmaColor, &sigmaSpace, &handle, &stream, &borderMode]
        {
            nvcv::ImageBatchVarShapeWrapHandle input(in);
            nvcv::ImageBatchVarShapeWrapHandle output(out);
            nvcv::TensorWrapHandle             diameterData(diameter);
            nvcv::TensorWrapHandle             sigmaColorData(sigmaColor);
            nvcv::TensorWrapHandle             sigmaSpaceData(sigmaSpace);
            priv::ToDynamicRef<priv::BilateralFilter>(handle)(stream, input.resource(), output.resource(),
                                                              diameterData.resource(), sigmaColorData.resource(),
                                                              sigmaSpaceData.resource(), borderMode);
        });
}
