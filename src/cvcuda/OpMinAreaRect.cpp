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

#include "priv/OpMinAreaRect.hpp"

#include "priv/Nvtx.hpp"
#include "priv/SymbolVersioning.hpp"

#include <nvcv/Exception.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/util/Assert.h>

namespace priv = cvcuda::priv;

CVCUDA_DEFINE_API(0, 4, NVCVStatus, cvcudaMinAreaRectCreate, (NVCVOperatorHandle * handle, int maxContourNum))
{
    return nvcv::ProtectCall(
        [&handle, &maxContourNum]
        {
            if (handle == nullptr)
            {
                throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                      "Pointer to NVCVOperator handle must not be NULL");
            }

            *handle = priv::CreateOperatorHandle<priv::MinAreaRect>(maxContourNum);
        });
}

CVCUDA_DEFINE_API(0, 4, NVCVStatus, cvcudaMinAreaRectSubmit,
                  (NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle in, NVCVTensorHandle out,
                   NVCVTensorHandle numPointsInContour, const int totalContours))
{
    CVCUDA_NVTX_RANGE("cvcudaMinAreaRectSubmit");
    return nvcv::ProtectCall(
        [&in, &out, &numPointsInContour, &handle, &stream, &totalContours]
        {
            nvcv::TensorWrapHandle input(in);
            nvcv::TensorWrapHandle output(out);
            nvcv::TensorWrapHandle _numPointsInContour(numPointsInContour);
            priv::ToDynamicRef<priv::MinAreaRect>(handle)(stream, input.resource(), output.resource(),
                                                          _numPointsInContour.resource(), totalContours);
        });
}
