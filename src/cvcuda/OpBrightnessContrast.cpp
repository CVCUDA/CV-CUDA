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

#include "priv/OpBrightnessContrast.hpp"

#include "priv/Nvtx.hpp"
#include "priv/SymbolVersioning.hpp"

#include <nvcv/Exception.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/util/Assert.h>

namespace priv = cvcuda::priv;

template<typename InWrap, typename OutWrap, typename InHandle, typename OutHandle>
void SubmitBrightnessContrast(NVCVOperatorHandle handle, cudaStream_t stream, InHandle in, OutHandle out,
                              NVCVTensorHandle brightness, NVCVTensorHandle contrast, NVCVTensorHandle brightnessShift,
                              NVCVTensorHandle contrastCenter)
{
    InWrap                 _in(in);
    OutWrap                _out(out);
    nvcv::TensorWrapHandle _brightness(brightness);
    nvcv::TensorWrapHandle _contrast(contrast);
    nvcv::TensorWrapHandle _brightnessShift(brightnessShift);
    nvcv::TensorWrapHandle _contrastCenter(contrastCenter);
    priv::ToDynamicRef<priv::BrightnessContrast>(handle)(stream, _in.resource(), _out.resource(),
                                                         _brightness.resource(), _contrast.resource(),
                                                         _brightnessShift.resource(), _contrastCenter.resource());
}

template<typename InWrap, typename OutWrap, typename InHandle, typename OutHandle>
void SubmitBrightnessContrastScalar(NVCVOperatorHandle handle, cudaStream_t stream, InHandle in, OutHandle out,
                                    double brightness, double contrast, double brightnessShift, double contrastCenter,
                                    bool clamp)
{
    InWrap  _in(in);
    OutWrap _out(out);
    priv::ToDynamicRef<priv::BrightnessContrast>(handle)(stream, _in.resource(), _out.resource(), brightness, contrast,
                                                         brightnessShift, contrastCenter, clamp);
}

CVCUDA_DEFINE_API(0, 3, NVCVStatus, cvcudaBrightnessContrastCreate, (NVCVOperatorHandle * handle))
{
    return nvcv::ProtectCall(
        [&handle]
        {
            if (handle == nullptr)
            {
                throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                      "Pointer to NVCVOperator handle must not be NULL");
            }

            *handle = priv::CreateOperatorHandle<priv::BrightnessContrast>();
        });
}

CVCUDA_DEFINE_API(0, 3, NVCVStatus, cvcudaBrightnessContrastSubmit,
                  (NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle in, NVCVTensorHandle out,
                   NVCVTensorHandle brightness, NVCVTensorHandle contrast, NVCVTensorHandle brightnessShift,
                   NVCVTensorHandle contrastCenter))
{
    CVCUDA_NVTX_RANGE("cvcudaBrightnessContrastSubmit");
    return nvcv::ProtectCall(
        [&in, &out, &brightness, &contrast, &brightnessShift, &contrastCenter, &handle, &stream]
        {
            SubmitBrightnessContrast<nvcv::TensorWrapHandle, nvcv::TensorWrapHandle>(
                handle, stream, in, out, brightness, contrast, brightnessShift, contrastCenter);
        });
}

CVCUDA_DEFINE_API(0, 3, NVCVStatus, cvcudaBrightnessContrastVarShapeSubmit,
                  (NVCVOperatorHandle handle, cudaStream_t stream, NVCVImageBatchHandle in, NVCVImageBatchHandle out,
                   NVCVTensorHandle brightness, NVCVTensorHandle contrast, NVCVTensorHandle brightnessShift,
                   NVCVTensorHandle contrastCenter))
{
    CVCUDA_NVTX_RANGE("cvcudaBrightnessContrastVarShapeSubmit");
    return nvcv::ProtectCall(
        [&in, &out, &brightness, &contrast, &brightnessShift, &contrastCenter, &handle, &stream]
        {
            SubmitBrightnessContrast<nvcv::ImageBatchVarShapeWrapHandle, nvcv::ImageBatchVarShapeWrapHandle>(
                handle, stream, in, out, brightness, contrast, brightnessShift, contrastCenter);
        });
}

CVCUDA_DEFINE_API(0, 17, NVCVStatus, cvcudaBrightnessContrastScalarSubmit,
                  (NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle in, NVCVTensorHandle out,
                   double brightness, double contrast, double brightnessShift, double contrastCenter, bool clamp))
{
    CVCUDA_NVTX_RANGE("cvcudaBrightnessContrastScalarSubmit");
    return nvcv::ProtectCall(
        [&handle, &stream, &in, &out, &brightness, &contrast, &brightnessShift, &contrastCenter, &clamp]
        {
            SubmitBrightnessContrastScalar<nvcv::TensorWrapHandle, nvcv::TensorWrapHandle>(
                handle, stream, in, out, brightness, contrast, brightnessShift, contrastCenter, clamp);
        });
}

CVCUDA_DEFINE_API(0, 17, NVCVStatus, cvcudaBrightnessContrastVarShapeScalarSubmit,
                  (NVCVOperatorHandle handle, cudaStream_t stream, NVCVImageBatchHandle in, NVCVImageBatchHandle out,
                   double brightness, double contrast, double brightnessShift, double contrastCenter, bool clamp))
{
    CVCUDA_NVTX_RANGE("cvcudaBrightnessContrastVarShapeScalarSubmit");
    return nvcv::ProtectCall(
        [&handle, &stream, &in, &out, &brightness, &contrast, &brightnessShift, &contrastCenter, &clamp]
        {
            SubmitBrightnessContrastScalar<nvcv::ImageBatchVarShapeWrapHandle, nvcv::ImageBatchVarShapeWrapHandle>(
                handle, stream, in, out, brightness, contrast, brightnessShift, contrastCenter, clamp);
        });
}
