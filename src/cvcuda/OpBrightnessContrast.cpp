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

#include "priv/OpBrightnessContrast.hpp"

#include "priv/SymbolVersioning.hpp"

#include <nvcv/Exception.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>
#include <util/Assert.h>

namespace priv = cvcuda::priv;

CVCUDA_DEFINE_API(0, 3, NVCVStatus, cvcudaBrightnessContrastCreate, (NVCVOperatorHandle * handle))
{
    return nvcv::ProtectCall(
        [&]
        {
            if (handle == nullptr)
            {
                throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                      "Pointer to NVCVOperator handle must not be NULL");
            }

            *handle = reinterpret_cast<NVCVOperatorHandle>(new priv::BrightnessContrast());
        });
}

CVCUDA_DEFINE_API(0, 3, NVCVStatus, cvcudaBrightnessContrastSubmit,
                  (NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle in, NVCVTensorHandle out,
                   NVCVTensorHandle brightness, NVCVTensorHandle contrast, NVCVTensorHandle brightnessShift,
                   NVCVTensorHandle contrastCenter))
{
    return nvcv::ProtectCall(
        [&]
        {
            nvcv::TensorWrapHandle _in(in), _out(out);
            nvcv::TensorWrapHandle _brightness(brightness), _contrast(contrast);
            nvcv::TensorWrapHandle _brightnessShift(brightnessShift), _contrastCenter(contrastCenter);
            priv::ToDynamicRef<priv::BrightnessContrast>(handle)(stream, _in, _out, _brightness, _contrast,
                                                                 _brightnessShift, _contrastCenter);
        });
}

CVCUDA_DEFINE_API(0, 3, NVCVStatus, cvcudaBrightnessContrastVarShapeSubmit,
                  (NVCVOperatorHandle handle, cudaStream_t stream, NVCVImageBatchHandle in, NVCVImageBatchHandle out,
                   NVCVTensorHandle brightness, NVCVTensorHandle contrast, NVCVTensorHandle brightnessShift,
                   NVCVTensorHandle contrastCenter))
{
    return nvcv::ProtectCall(
        [&]
        {
            nvcv::ImageBatchVarShapeWrapHandle _in(in), _out(out);
            nvcv::TensorWrapHandle             _brightness(brightness), _contrast(contrast);
            nvcv::TensorWrapHandle             _brightnessShift(brightnessShift), _contrastCenter(contrastCenter);
            priv::ToDynamicRef<priv::BrightnessContrast>(handle)(stream, _in, _out, _brightness, _contrast,
                                                                 _brightnessShift, _contrastCenter);
        });
}
