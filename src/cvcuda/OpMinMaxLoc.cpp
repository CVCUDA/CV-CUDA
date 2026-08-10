/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "priv/OpMinMaxLoc.hpp"

#include "priv/Nvtx.hpp"
#include "priv/SymbolVersioning.hpp"

#include <nvcv/Exception.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/util/Assert.h>

namespace priv = cvcuda::priv;

CVCUDA_DEFINE_API(0, 4, NVCVStatus, cvcudaMinMaxLocCreate, (NVCVOperatorHandle * handle))
{
    return nvcv::ProtectCall(
        [&handle]
        {
            if (handle == nullptr)
            {
                throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                      "Pointer to NVCVOperator handle must not be NULL");
            }

            *handle = priv::CreateOperatorHandle<priv::MinMaxLoc>();
        });
}

CVCUDA_DEFINE_API(0, 4, NVCVStatus, cvcudaMinMaxLocSubmit,
                  (NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle in, NVCVTensorHandle minVal,
                   NVCVTensorHandle minLoc, NVCVTensorHandle numMin, NVCVTensorHandle maxVal, NVCVTensorHandle maxLoc,
                   NVCVTensorHandle numMax))
{
    CVCUDA_NVTX_RANGE("cvcudaMinMaxLocSubmit");
    return nvcv::ProtectCall(
        [&in, &handle, &stream, &minVal, &minLoc, &numMin, &maxVal, &maxLoc, &numMax]
        {
            nvcv::TensorWrapHandle input(in);

            priv::ToDynamicRef<priv::MinMaxLoc>(handle)(
                stream, input.resource(), nvcv::TensorWrapHandle{minVal}.resource(),
                nvcv::TensorWrapHandle{minLoc}.resource(), nvcv::TensorWrapHandle{numMin}.resource(),
                nvcv::TensorWrapHandle{maxVal}.resource(), nvcv::TensorWrapHandle{maxLoc}.resource(),
                nvcv::TensorWrapHandle{numMax}.resource());
        });
}

CVCUDA_DEFINE_API(0, 4, NVCVStatus, cvcudaMinMaxLocVarShapeSubmit,
                  (NVCVOperatorHandle handle, cudaStream_t stream, NVCVImageBatchHandle in, NVCVTensorHandle minVal,
                   NVCVTensorHandle minLoc, NVCVTensorHandle numMin, NVCVTensorHandle maxVal, NVCVTensorHandle maxLoc,
                   NVCVTensorHandle numMax))
{
    CVCUDA_NVTX_RANGE("cvcudaMinMaxLocVarShapeSubmit");
    return nvcv::ProtectCall(
        [&in, &handle, &stream, &minVal, &minLoc, &numMin, &maxVal, &maxLoc, &numMax]
        {
            nvcv::ImageBatchVarShapeWrapHandle input(in);

            priv::ToDynamicRef<priv::MinMaxLoc>(handle)(
                stream, input.resource(), nvcv::TensorWrapHandle{minVal}.resource(),
                nvcv::TensorWrapHandle{minLoc}.resource(), nvcv::TensorWrapHandle{numMin}.resource(),
                nvcv::TensorWrapHandle{maxVal}.resource(), nvcv::TensorWrapHandle{maxLoc}.resource(),
                nvcv::TensorWrapHandle{numMax}.resource());
        });
}
