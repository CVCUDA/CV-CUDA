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

#include "priv/OpResizeCropConvertReformat.hpp"

#include "priv/Nvtx.hpp"
#include "priv/SymbolVersioning.hpp"

#include <nvcv/Exception.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/util/Assert.h>

namespace priv = cvcuda::priv;

CVCUDA_DEFINE_API(0, 8, NVCVStatus, cvcudaResizeCropConvertReformatCreate, (NVCVOperatorHandle * handle))
{
    return nvcv::ProtectCall(
        [&handle]
        {
            if (handle == nullptr)
            {
                throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                      "Pointer to NVCVOperator handle must not be NULL");
            }

            *handle = priv::CreateOperatorHandle<priv::ResizeCropConvertReformat>();
        });
}

CVCUDA_DEFINE_API(0, 10, NVCVStatus, cvcudaResizeCropConvertReformatSubmit,
                  (NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle in, NVCVTensorHandle out,
                   const NVCVSize2D resizeDim, const NVCVInterpolationType interpolation, const int2 cropPos,
                   const NVCVChannelManip manip, const float scale, const float offset, const bool srcCast))
{
    CVCUDA_NVTX_RANGE("cvcudaResizeCropConvertReformatSubmit");
    return nvcv::ProtectCall(
        [&in, &out, &handle, &stream, &resizeDim, &interpolation, &cropPos, &manip, &scale, &offset, &srcCast]
        {
            nvcv::TensorWrapHandle input(in);
            nvcv::TensorWrapHandle output(out);
            priv::ToDynamicRef<priv::ResizeCropConvertReformat>(handle)(stream, input.resource(), output.resource(),
                                                                        resizeDim, interpolation, cropPos, manip, scale,
                                                                        offset, srcCast);
        });
}

CVCUDA_DEFINE_API(0, 10, NVCVStatus, cvcudaResizeCropConvertReformatVarShapeSubmit,
                  (NVCVOperatorHandle handle, cudaStream_t stream, NVCVImageBatchHandle in, NVCVTensorHandle out,
                   const NVCVSize2D resizeDim, const NVCVInterpolationType interpolation, const int2 cropPos,
                   const NVCVChannelManip manip, const float scale, const float offset, const bool srcCast))
{
    CVCUDA_NVTX_RANGE("cvcudaResizeCropConvertReformatVarShapeSubmit");
    return nvcv::ProtectCall(
        [&in, &out, &handle, &stream, &resizeDim, &interpolation, &cropPos, &manip, &scale, &offset, &srcCast]
        {
            nvcv::ImageBatchVarShapeWrapHandle input(in);
            nvcv::TensorWrapHandle             output(out);
            priv::ToDynamicRef<priv::ResizeCropConvertReformat>(handle)(stream, input.resource(), output.resource(),
                                                                        resizeDim, interpolation, cropPos, manip, scale,
                                                                        offset, srcCast);
        });
}
