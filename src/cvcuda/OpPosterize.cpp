/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "priv/OpPosterize.hpp"

#include "priv/Nvtx.hpp"
#include "priv/SymbolVersioning.hpp"

#include <nvcv/Exception.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>

namespace priv = cvcuda::priv;

CVCUDA_DEFINE_API(0, 0, NVCVStatus, cvcudaPosterizeCreate, (NVCVOperatorHandle * handle))
{
    return nvcv::ProtectCall(
        [handle]
        {
            if (handle == nullptr)
            {
                throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                      "Pointer to NVCVOperator handle must not be NULL");
            }

            // Ownership of the operator is transferred to the C handle and released by
            // cvcudaOperatorDestroy; this matches every other operator's Create entry point.
            *handle = reinterpret_cast<NVCVOperatorHandle>(new priv::Posterize()); // NOSONAR
        });
}

CVCUDA_DEFINE_API(0, 0, NVCVStatus, cvcudaPosterizeSubmit,
                  (NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle in, NVCVTensorHandle out,
                   int32_t bits))
{
    CVCUDA_NVTX_RANGE("cvcudaPosterizeSubmit");
    return nvcv::ProtectCall(
        [handle, stream, in, out, bits]
        {
            nvcv::TensorWrapHandle input(in);
            nvcv::TensorWrapHandle output(out);
            priv::ToDynamicRef<priv::Posterize>(handle)(stream, input.resource(), output.resource(), bits);
        });
}

CVCUDA_DEFINE_API(0, 0, NVCVStatus, cvcudaPosterizeVarShapeSubmit,
                  (NVCVOperatorHandle handle, cudaStream_t stream, NVCVImageBatchHandle in, NVCVImageBatchHandle out,
                   int32_t bits))
{
    CVCUDA_NVTX_RANGE("cvcudaPosterizeVarShapeSubmit");
    return nvcv::ProtectCall(
        [handle, stream, in, out, bits]
        {
            nvcv::ImageBatchVarShapeWrapHandle input(in);
            nvcv::ImageBatchVarShapeWrapHandle output(out);
            priv::ToDynamicRef<priv::Posterize>(handle)(stream, input.resource(), output.resource(), bits);
        });
}
