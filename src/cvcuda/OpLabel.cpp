/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "priv/SymbolVersioning.hpp"

#include <nvcv/Exception.hpp>
#include <nvcv/Tensor.hpp>
#include <util/Assert.h>

CVCUDA_DEFINE_API(0, 5, NVCVStatus, cvcudaLabelCreate, (NVCVOperatorHandle * handle))
{
    return nvcv::ProtectCall(
        [&]
        {
            if (handle == nullptr)
            {
                throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                      "Pointer to NVCVOperator handle must not be NULL");
            }

            *handle = reinterpret_cast<NVCVOperatorHandle>(new cvcuda::priv::Label());
        });
}

CVCUDA_DEFINE_API(0, 7, NVCVStatus, cvcudaLabelSubmit,
                  (NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle in, NVCVTensorHandle out,
                   NVCVTensorHandle bgLabel, NVCVTensorHandle minThresh, NVCVTensorHandle maxThresh,
                   NVCVTensorHandle minSize, NVCVTensorHandle count, NVCVTensorHandle stats, NVCVTensorHandle mask,
                   NVCVConnectivityType connectivity, NVCVLabelType assignLabels, NVCVLabelMaskType maskType))
{
    return nvcv::ProtectCall(
        [&]
        {
            cvcuda::priv::ToDynamicRef<cvcuda::priv::Label>(handle)(
                stream, nvcv::TensorWrapHandle{in}, nvcv::TensorWrapHandle{out}, nvcv::TensorWrapHandle{bgLabel},
                nvcv::TensorWrapHandle{minThresh}, nvcv::TensorWrapHandle{maxThresh}, nvcv::TensorWrapHandle{minSize},
                nvcv::TensorWrapHandle{count}, nvcv::TensorWrapHandle{stats}, nvcv::TensorWrapHandle{mask},
                connectivity, assignLabels, maskType);
        });
}
