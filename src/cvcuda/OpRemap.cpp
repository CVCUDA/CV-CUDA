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

#include "priv/OpRemap.hpp"

#include "priv/SymbolVersioning.hpp"

#include <nvcv/Exception.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>
#include <util/Assert.h>

namespace priv = cvcuda::priv;

CVCUDA_DEFINE_API(0, 3, NVCVStatus, cvcudaRemapCreate, (NVCVOperatorHandle * handle))
{
    return nvcv::ProtectCall(
        [&]
        {
            if (handle == nullptr)
            {
                throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                      "Pointer to NVCVOperator handle must not be NULL");
            }

            *handle = reinterpret_cast<NVCVOperatorHandle>(new priv::Remap());
        });
}

CVCUDA_DEFINE_API(0, 3, NVCVStatus, cvcudaRemapSubmit,
                  (NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle in, NVCVTensorHandle out,
                   NVCVTensorHandle map, NVCVInterpolationType inInterp, NVCVInterpolationType mapInterp,
                   NVCVRemapMapValueType mapValueType, int8_t alignCorners, NVCVBorderType border, float4 borderValue))
{
    return nvcv::ProtectCall(
        [&]
        {
            nvcv::TensorWrapHandle _in(in), _out(out), _map(map);
            priv::ToDynamicRef<priv::Remap>(handle)(stream, _in, _out, _map, inInterp, mapInterp, mapValueType,
                                                    static_cast<bool>(alignCorners), border, borderValue);
        });
}

CVCUDA_DEFINE_API(0, 3, NVCVStatus, cvcudaRemapVarShapeSubmit,
                  (NVCVOperatorHandle handle, cudaStream_t stream, NVCVImageBatchHandle in, NVCVImageBatchHandle out,
                   NVCVTensorHandle map, NVCVInterpolationType inInterp, NVCVInterpolationType mapInterp,
                   NVCVRemapMapValueType mapValueType, int8_t alignCorners, NVCVBorderType border, float4 borderValue))
{
    return nvcv::ProtectCall(
        [&]
        {
            nvcv::ImageBatchVarShapeWrapHandle _in(in), _out(out);
            nvcv::TensorWrapHandle             _map(map);
            priv::ToDynamicRef<priv::Remap>(handle)(stream, _in, _out, _map, inInterp, mapInterp, mapValueType,
                                                    static_cast<bool>(alignCorners), border, borderValue);
        });
}
