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

#include "priv/OpLabel.hpp"

#include "priv/Nvtx.hpp"
#include "priv/SymbolVersioning.hpp"

#include <nvcv/Exception.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/util/Assert.h>

CVCUDA_DEFINE_API(0, 5, NVCVStatus, cvcudaLabelCreate, (NVCVOperatorHandle * handle))
{
    return nvcv::ProtectCall(
        [&handle]
        {
            if (handle == nullptr)
            {
                throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                      "Pointer to NVCVOperator handle must not be NULL");
            }

            *handle = cvcuda::priv::CreateOperatorHandle<cvcuda::priv::Label>();
        });
}

CVCUDA_DEFINE_API(0, 7, NVCVStatus, cvcudaLabelSubmit,
                  (NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle in, NVCVTensorHandle out,
                   NVCVTensorHandle bgLabel, NVCVTensorHandle minThresh, NVCVTensorHandle maxThresh,
                   NVCVTensorHandle minSize, NVCVTensorHandle count, NVCVTensorHandle stats, NVCVTensorHandle mask,
                   NVCVConnectivityType connectivity, NVCVLabelType assignLabels, NVCVLabelMaskType maskType))
{
    CVCUDA_NVTX_RANGE("cvcudaLabelSubmit");
    return nvcv::ProtectCall(
        [&handle, &stream, &in, &out, &bgLabel, &minThresh, &maxThresh, &minSize, &count, &stats, &mask, &connectivity,
         &assignLabels, &maskType]
        {
            cvcuda::priv::ToDynamicRef<cvcuda::priv::Label>(handle)(
                stream, nvcv::TensorWrapHandle{in}.resource(), nvcv::TensorWrapHandle{out}.resource(),
                nvcv::TensorWrapHandle{bgLabel}.resource(), nvcv::TensorWrapHandle{minThresh}.resource(),
                nvcv::TensorWrapHandle{maxThresh}.resource(), nvcv::TensorWrapHandle{minSize}.resource(),
                nvcv::TensorWrapHandle{count}.resource(), nvcv::TensorWrapHandle{stats}.resource(),
                nvcv::TensorWrapHandle{mask}.resource(), connectivity, assignLabels, maskType);
        });
}
