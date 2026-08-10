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

#include "priv/OpInvert.hpp"

#include "priv/Nvtx.hpp"
#include "priv/SymbolVersioning.hpp"

#include <nvcv/Exception.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>

namespace priv = cvcuda::priv;

CVCUDA_DEFINE_API(0, 0, NVCVStatus, cvcudaInvertCreate, (NVCVOperatorHandle * handle))
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
            *handle = reinterpret_cast<NVCVOperatorHandle>(new priv::Invert()); // NOSONAR
        });
}

CVCUDA_DEFINE_API(0, 0, NVCVStatus, cvcudaInvertSubmit,
                  (NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle in, NVCVTensorHandle out))
{
    CVCUDA_NVTX_RANGE("cvcudaInvertSubmit");
    return nvcv::ProtectCall(
        [handle, stream, in, out]
        {
            nvcv::TensorWrapHandle input(in);
            nvcv::TensorWrapHandle output(out);
            priv::ToDynamicRef<priv::Invert>(handle)(stream, input.resource(), output.resource());
        });
}

CVCUDA_DEFINE_API(0, 0, NVCVStatus, cvcudaInvertVarShapeSubmit,
                  (NVCVOperatorHandle handle, cudaStream_t stream, NVCVImageBatchHandle in, NVCVImageBatchHandle out))
{
    CVCUDA_NVTX_RANGE("cvcudaInvertVarShapeSubmit");
    return nvcv::ProtectCall(
        [handle, stream, in, out]
        {
            nvcv::ImageBatchVarShapeWrapHandle input(in);
            nvcv::ImageBatchVarShapeWrapHandle output(out);
            priv::ToDynamicRef<priv::Invert>(handle)(stream, input.resource(), output.resource());
        });
}
