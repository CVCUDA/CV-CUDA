/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "priv/OpNormalize.hpp"

#include "priv/SymbolVersioning.hpp"

#include <nvcv/Exception.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>
#include <util/Assert.h>

namespace priv = cvcuda::priv;

CVCUDA_DEFINE_API(0, 0, NVCVStatus, cvcudaNormalizeCreate, (NVCVOperatorHandle * handle))
{
    return nvcv::ProtectCall(
        [&]
        {
            if (handle == nullptr)
            {
                throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                      "Pointer to NVCVOperator handle must not be NULL");
            }

            *handle = reinterpret_cast<NVCVOperatorHandle>(new priv::Normalize());
        });
}

CVCUDA_DEFINE_API(0, 2, NVCVStatus, cvcudaNormalizeSubmit,
                  (NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle in, NVCVTensorHandle base,
                   NVCVTensorHandle scale, NVCVTensorHandle out, float global_scale, float shift, float epsilon,
                   uint32_t flags))
{
    return nvcv::ProtectCall(
        [&]
        {
            nvcv::TensorWrapHandle inWrap(in), baseWrap(base), scaleWrap(scale), outWrap(out);
            priv::ToDynamicRef<priv::Normalize>(handle)(stream, inWrap, baseWrap, scaleWrap, outWrap, global_scale,
                                                        shift, epsilon, flags);
        });
}

CVCUDA_DEFINE_API(0, 2, NVCVStatus, cvcudaNormalizeVarShapeSubmit,
                  (NVCVOperatorHandle handle, cudaStream_t stream, NVCVImageBatchHandle in, NVCVTensorHandle base,
                   NVCVTensorHandle scale, NVCVImageBatchHandle out, float global_scale, float shift, float epsilon,
                   uint32_t flags))
{
    return nvcv::ProtectCall(
        [&]
        {
            nvcv::TensorWrapHandle             baseWrap(base), scaleWrap(scale);
            nvcv::ImageBatchVarShapeWrapHandle inWrap(in), outWrap(out);
            priv::ToDynamicRef<priv::Normalize>(handle)(stream, inWrap, baseWrap, scaleWrap, outWrap, global_scale,
                                                        shift, epsilon, flags);
        });
}
