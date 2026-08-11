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

#include "priv/OpGaussian.hpp"

#include "priv/Nvtx.hpp"
#include "priv/SymbolVersioning.hpp"

#include <nvcv/Exception.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/util/Assert.h>

namespace priv = cvcuda::priv;

CVCUDA_DEFINE_API(0, 2, NVCVStatus, cvcudaGaussianCreate,
                  (NVCVOperatorHandle * handle, int32_t maxKernelWidth, int32_t maxKernelHeight,
                   int32_t maxVarShapeBatchSize))
{
    return nvcv::ProtectCall(
        [&handle, &maxKernelWidth, &maxKernelHeight, &maxVarShapeBatchSize]
        {
            if (handle == nullptr)
            {
                throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                      "Pointer to NVCVOperator handle must not be NULL");
            }

            *handle = priv::CreateOperatorHandle<priv::Gaussian>(nvcv::Size2D{maxKernelWidth, maxKernelHeight},
                                                                 maxVarShapeBatchSize);
        });
}

CVCUDA_DEFINE_API(0, 2, NVCVStatus, cvcudaGaussianSubmit,
                  (NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle in, NVCVTensorHandle out,
                   int32_t kernelWidth, int32_t kernelHeight, double sigmaX, double sigmaY, NVCVBorderType borderMode))
{
    CVCUDA_NVTX_RANGE("cvcudaGaussianSubmit");
    return nvcv::ProtectCall(
        [&out, &in, &handle, &stream, &kernelWidth, &kernelHeight, &sigmaX, &sigmaY, &borderMode]
        {
            nvcv::TensorWrapHandle output(out);
            nvcv::TensorWrapHandle input(in);
            priv::ToDynamicRef<priv::Gaussian>(handle)(stream, input.resource(), output.resource(),
                                                       nvcv::Size2D{kernelWidth, kernelHeight}, double2{sigmaX, sigmaY},
                                                       borderMode);
        });
}

CVCUDA_DEFINE_API(0, 2, NVCVStatus, cvcudaGaussianVarShapeSubmit,
                  (NVCVOperatorHandle handle, cudaStream_t stream, NVCVImageBatchHandle in, NVCVImageBatchHandle out,
                   NVCVTensorHandle kernelSize, NVCVTensorHandle sigma, NVCVBorderType borderMode))
{
    CVCUDA_NVTX_RANGE("cvcudaGaussianVarShapeSubmit");
    return nvcv::ProtectCall(
        [&in, &out, &kernelSize, &sigma, &handle, &stream, &borderMode]
        {
            nvcv::ImageBatchVarShapeWrapHandle inWrap(in);
            nvcv::ImageBatchVarShapeWrapHandle outWrap(out);
            nvcv::TensorWrapHandle             kernelSizeWrap(kernelSize);
            nvcv::TensorWrapHandle             sigmaWrap(sigma);
            priv::ToDynamicRef<priv::Gaussian>(handle)(stream, inWrap.resource(), outWrap.resource(),
                                                       kernelSizeWrap.resource(), sigmaWrap.resource(), borderMode);
        });
}
