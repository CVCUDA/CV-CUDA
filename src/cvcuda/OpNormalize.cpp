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

#include "priv/OpNormalize.hpp"

#include "priv/Nvtx.hpp"
#include "priv/SymbolVersioning.hpp"

#include <nvcv/Exception.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/util/Assert.h>

namespace priv = cvcuda::priv;

CVCUDA_DEFINE_API(0, 0, NVCVStatus, cvcudaNormalizeCreate, (NVCVOperatorHandle * handle))
{
    return nvcv::ProtectCall(
        [&handle]
        {
            if (handle == nullptr)
            {
                throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                      "Pointer to NVCVOperator handle must not be NULL");
            }

            *handle = priv::CreateOperatorHandle<priv::Normalize>();
        });
}

CVCUDA_DEFINE_API(0, 2, NVCVStatus, cvcudaNormalizeSubmit,
                  (NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle in, NVCVTensorHandle base,
                   NVCVTensorHandle scale, NVCVTensorHandle out, float global_scale, float shift, float epsilon,
                   uint32_t flags))
{
    CVCUDA_NVTX_RANGE("cvcudaNormalizeSubmit");
    return nvcv::ProtectCall(
        [&in, &base, &scale, &out, &handle, &stream, &global_scale, &shift, &epsilon, &flags]
        {
            nvcv::TensorWrapHandle inWrap(in);
            nvcv::TensorWrapHandle baseWrap(base);
            nvcv::TensorWrapHandle scaleWrap(scale);
            nvcv::TensorWrapHandle outWrap(out);
            priv::ToDynamicRef<priv::Normalize>(handle)(stream, inWrap.resource(), baseWrap.resource(),
                                                        scaleWrap.resource(), outWrap.resource(), global_scale, shift,
                                                        epsilon, flags);
        });
}

CVCUDA_DEFINE_API(0, 17, NVCVStatus, cvcudaNormalizeScalarSubmit,
                  (NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle in, float4 base, float4 scale,
                   int32_t baseChannels, int32_t scaleChannels, NVCVTensorHandle out, float global_scale, float shift,
                   float epsilon, uint32_t flags))
{
    CVCUDA_NVTX_RANGE("cvcudaNormalizeScalarSubmit");
    return nvcv::ProtectCall(
        [&in, &base, &scale, &baseChannels, &scaleChannels, &out, &handle, &stream, &global_scale, &shift, &epsilon,
         &flags]
        {
            nvcv::TensorWrapHandle inWrap(in);
            nvcv::TensorWrapHandle outWrap(out);
            priv::ToDynamicRef<priv::Normalize>(handle)(stream, inWrap.resource(), base, scale, baseChannels,
                                                        scaleChannels, outWrap.resource(), global_scale, shift, epsilon,
                                                        flags);
        });
}

CVCUDA_DEFINE_API(0, 2, NVCVStatus, cvcudaNormalizeVarShapeSubmit,
                  (NVCVOperatorHandle handle, cudaStream_t stream, NVCVImageBatchHandle in, NVCVTensorHandle base,
                   NVCVTensorHandle scale, NVCVImageBatchHandle out, float global_scale, float shift, float epsilon,
                   uint32_t flags))
{
    CVCUDA_NVTX_RANGE("cvcudaNormalizeVarShapeSubmit");
    return nvcv::ProtectCall(
        [&base, &scale, &in, &out, &handle, &stream, &global_scale, &shift, &epsilon, &flags]
        {
            nvcv::TensorWrapHandle             baseWrap(base);
            nvcv::TensorWrapHandle             scaleWrap(scale);
            nvcv::ImageBatchVarShapeWrapHandle inWrap(in);
            nvcv::ImageBatchVarShapeWrapHandle outWrap(out);
            priv::ToDynamicRef<priv::Normalize>(handle)(stream, inWrap.resource(), baseWrap.resource(),
                                                        scaleWrap.resource(), outWrap.resource(), global_scale, shift,
                                                        epsilon, flags);
        });
}
