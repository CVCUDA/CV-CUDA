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

#include "priv/OpLaplacian.hpp"

#include "priv/SymbolVersioning.hpp"

#include <nvcv/Exception.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>
#include <util/Assert.h>

namespace priv = cvcuda::priv;

CVCUDA_DEFINE_API(0, 2, NVCVStatus, cvcudaLaplacianCreate, (NVCVOperatorHandle * handle))
{
    return nvcv::ProtectCall(
        [&]
        {
            if (handle == nullptr)
            {
                throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                      "Pointer to NVCVOperator handle must not be NULL");
            }

            *handle = reinterpret_cast<NVCVOperatorHandle>(new priv::Laplacian());
        });
}

CVCUDA_DEFINE_API(0, 2, NVCVStatus, cvcudaLaplacianSubmit,
                  (NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle in, NVCVTensorHandle out,
                   int32_t ksize, float scale, NVCVBorderType borderMode))
{
    return nvcv::ProtectCall(
        [&]
        {
            nvcv::TensorWrapHandle output(out), input(in);
            priv::ToDynamicRef<priv::Laplacian>(handle)(stream, input, output, ksize, scale, borderMode);
        });
}

CVCUDA_DEFINE_API(0, 2, NVCVStatus, cvcudaLaplacianVarShapeSubmit,
                  (NVCVOperatorHandle handle, cudaStream_t stream, NVCVImageBatchHandle in, NVCVImageBatchHandle out,
                   NVCVTensorHandle ksize, NVCVTensorHandle scale, NVCVBorderType borderMode))
{
    return nvcv::ProtectCall(
        [&]
        {
            nvcv::ImageBatchVarShapeWrapHandle inWrap(in), outWrap(out);
            nvcv::TensorWrapHandle             ksizeWrap(ksize), scaleWrap(scale);
            priv::ToDynamicRef<priv::Laplacian>(handle)(stream, inWrap, outWrap, ksizeWrap, scaleWrap, borderMode);
        });
}
