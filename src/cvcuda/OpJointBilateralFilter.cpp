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

#include "priv/OpJointBilateralFilter.hpp"

#include "priv/SymbolVersioning.hpp"

#include <nvcv/Exception.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>
#include <util/Assert.h>

namespace priv = cvcuda::priv;

CVCUDA_DEFINE_API(0, 2, NVCVStatus, cvcudaJointBilateralFilterCreate, (NVCVOperatorHandle * handle))
{
    return nvcv::ProtectCall(
        [&]
        {
            if (handle == nullptr)
            {
                throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                      "Pointer to NVCVOperator handle must not be NULL");
            }

            *handle = reinterpret_cast<NVCVOperatorHandle>(new priv::JointBilateralFilter());
        });
}

CVCUDA_DEFINE_API(0, 2, NVCVStatus, cvcudaJointBilateralFilterSubmit,
                  (NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle in, NVCVTensorHandle inColor,
                   NVCVTensorHandle out, int diameter, float sigmaColor, float sigmaSpace, NVCVBorderType borderMode))
{
    return nvcv::ProtectCall(
        [&]
        {
            nvcv::TensorWrapHandle input(in), inputColor(inColor), output(out);
            priv::ToDynamicRef<priv::JointBilateralFilter>(handle)(stream, input, inputColor, output, diameter,
                                                                   sigmaColor, sigmaSpace, borderMode);
        });
}

CVCUDA_DEFINE_API(0, 2, NVCVStatus, cvcudaJointBilateralFilterVarShapeSubmit,
                  (NVCVOperatorHandle handle, cudaStream_t stream, NVCVImageBatchHandle in,
                   NVCVImageBatchHandle inColor, NVCVImageBatchHandle out, NVCVTensorHandle diameter,
                   NVCVTensorHandle sigmaColor, NVCVTensorHandle sigmaSpace, NVCVBorderType borderMode))
{
    return nvcv::ProtectCall(
        [&]
        {
            nvcv::ImageBatchVarShapeWrapHandle input(in), inputColor(inColor), output(out);
            nvcv::TensorWrapHandle diameterData(diameter), sigmaColorData(sigmaColor), sigmaSpaceData(sigmaSpace);
            priv::ToDynamicRef<priv::JointBilateralFilter>(handle)(stream, input, inputColor, output, diameterData,
                                                                   sigmaColorData, sigmaSpaceData, borderMode);
        });
}
