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

#include "priv/OpErase.hpp"

#include "priv/SymbolVersioning.hpp"

#include <nvcv/Exception.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>
#include <util/Assert.h>

namespace priv = cvcuda::priv;

CVCUDA_DEFINE_API(0, 2, NVCVStatus, cvcudaEraseCreate, (NVCVOperatorHandle * handle, int32_t max_num_erasing_area))
{
    return nvcv::ProtectCall(
        [&]
        {
            if (handle == nullptr)
            {
                throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                      "Pointer to NVCVOperator handle must not be NULL");
            }

            *handle = reinterpret_cast<NVCVOperatorHandle>(new priv::Erase(max_num_erasing_area));
        });
}

CVCUDA_DEFINE_API(0, 2, NVCVStatus, cvcudaEraseSubmit,
                  (NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle in, NVCVTensorHandle out,
                   NVCVTensorHandle anchor, NVCVTensorHandle erasing, NVCVTensorHandle values, NVCVTensorHandle imgIdx,
                   int8_t random, uint32_t seed))
{
    return nvcv::ProtectCall(
        [&]
        {
            nvcv::TensorWrapHandle input(in), output(out), anchorwrap(anchor), erasingwrap(erasing), valueswrap(values),
                imgIdxwrap(imgIdx);
            priv::ToDynamicRef<priv::Erase>(handle)(stream, input, output, anchorwrap, erasingwrap, valueswrap,
                                                    imgIdxwrap, random, seed);
        });
}

CVCUDA_DEFINE_API(0, 2, NVCVStatus, cvcudaEraseVarShapeSubmit,
                  (NVCVOperatorHandle handle, cudaStream_t stream, NVCVImageBatchHandle in, NVCVImageBatchHandle out,
                   NVCVTensorHandle anchor, NVCVTensorHandle erasing, NVCVTensorHandle values, NVCVTensorHandle imgIdx,
                   int8_t random, uint32_t seed))
{
    return nvcv::ProtectCall(
        [&]
        {
            nvcv::ImageBatchVarShapeWrapHandle input(in), output(out);
            nvcv::TensorWrapHandle anchorwrap(anchor), erasingwrap(erasing), valueswrap(values), imgIdxwrap(imgIdx);
            priv::ToDynamicRef<priv::Erase>(handle)(stream, input, output, anchorwrap, erasingwrap, valueswrap,
                                                    imgIdxwrap, random, seed);
        });
}
