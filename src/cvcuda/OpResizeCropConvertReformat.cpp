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

#include "priv/OpResizeCropConvertReformat.hpp"

#include "priv/SymbolVersioning.hpp"

#include <nvcv/Exception.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>
#include <util/Assert.h>

namespace priv = cvcuda::priv;

CVCUDA_DEFINE_API(0, 8, NVCVStatus, cvcudaResizeCropConvertReformatCreate, (NVCVOperatorHandle * handle))
{
    return nvcv::ProtectCall(
        [&]
        {
            if (handle == nullptr)
            {
                throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                      "Pointer to NVCVOperator handle must not be NULL");
            }

            *handle = reinterpret_cast<NVCVOperatorHandle>(new priv::ResizeCropConvertReformat());
        });
}

CVCUDA_DEFINE_API(0, 8, NVCVStatus, cvcudaResizeCropConvertReformatSubmit,
                  (NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle in, NVCVTensorHandle out,
                   const NVCVSize2D resizeDim, const NVCVInterpolationType interpolation, const int2 cropPos,
                   const NVCVChannelManip manip))
{
    return nvcv::ProtectCall(
        [&]
        {
            nvcv::TensorWrapHandle input(in), output(out);
            priv::ToDynamicRef<priv::ResizeCropConvertReformat>(handle)(stream, input, output, resizeDim, interpolation,
                                                                        cropPos, manip);
        });
}

CVCUDA_DEFINE_API(0, 8, NVCVStatus, cvcudaResizeCropConvertReformatVarShapeSubmit,
                  (NVCVOperatorHandle handle, cudaStream_t stream, NVCVImageBatchHandle in, NVCVTensorHandle out,
                   const NVCVSize2D resizeDim, const NVCVInterpolationType interpolation, const int2 cropPos,
                   const NVCVChannelManip manip))
{
    return nvcv::ProtectCall(
        [&]
        {
            nvcv::ImageBatchVarShapeWrapHandle input(in);
            nvcv::TensorWrapHandle             output(out);
            priv::ToDynamicRef<priv::ResizeCropConvertReformat>(handle)(stream, input, output, resizeDim, interpolation,
                                                                        cropPos, manip);
        });
}
