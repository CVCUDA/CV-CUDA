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

#include "priv/OpGaussianNoise.hpp"

#include "priv/Nvtx.hpp"
#include "priv/SymbolVersioning.hpp"

#include <nvcv/Exception.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/util/Assert.h>

namespace priv = cvcuda::priv;

CVCUDA_DEFINE_API(0, 4, NVCVStatus, cvcudaGaussianNoiseCreate, (NVCVOperatorHandle * handle, int maxBatchSize))
{
    return nvcv::ProtectCall(
        [&handle, &maxBatchSize]
        {
            if (handle == nullptr)
            {
                throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                      "Pointer to NVCVOperator handle must not be NULL");
            }

            *handle = priv::CreateOperatorHandle<priv::GaussianNoise>(maxBatchSize);
        });
}

CVCUDA_DEFINE_API(0, 4, NVCVStatus, cvcudaGaussianNoiseSubmit,
                  (NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle in, NVCVTensorHandle out,
                   NVCVTensorHandle mu, NVCVTensorHandle sigma, int8_t per_channel, unsigned long long seed))
{
    CVCUDA_NVTX_RANGE("cvcudaGaussianNoiseSubmit");
    return nvcv::ProtectCall(
        [&in, &out, &mu, &sigma, &handle, &stream, &per_channel, &seed]
        {
            nvcv::TensorWrapHandle input(in);
            nvcv::TensorWrapHandle output(out);
            nvcv::TensorWrapHandle muwrap(mu);
            nvcv::TensorWrapHandle sigmawrap(sigma);
            priv::ToDynamicRef<priv::GaussianNoise>(handle)(stream, input.resource(), output.resource(),
                                                            muwrap.resource(), sigmawrap.resource(),
                                                            static_cast<bool>(per_channel), seed);
        });
}

CVCUDA_DEFINE_API(0, 17, NVCVStatus, cvcudaGaussianNoiseScalarSubmit,
                  (NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle in, NVCVTensorHandle out, float mu,
                   float sigma, int8_t per_channel, unsigned long long seed, int8_t reseed, int8_t clip))
{
    CVCUDA_NVTX_RANGE("cvcudaGaussianNoiseScalarSubmit");
    return nvcv::ProtectCall(
        [&in, &out, &mu, &sigma, &handle, &stream, &per_channel, &seed, &reseed, &clip]
        {
            nvcv::TensorWrapHandle input(in);
            nvcv::TensorWrapHandle output(out);
            priv::ToDynamicRef<priv::GaussianNoise>(handle)(stream, input.resource(), output.resource(), mu, sigma,
                                                            static_cast<bool>(per_channel), seed,
                                                            static_cast<bool>(reseed), static_cast<bool>(clip));
        });
}

CVCUDA_DEFINE_API(0, 4, NVCVStatus, cvcudaGaussianNoiseVarShapeSubmit,
                  (NVCVOperatorHandle handle, cudaStream_t stream, NVCVImageBatchHandle in, NVCVImageBatchHandle out,
                   NVCVTensorHandle mu, NVCVTensorHandle sigma, int8_t per_channel, unsigned long long seed))
{
    CVCUDA_NVTX_RANGE("cvcudaGaussianNoiseVarShapeSubmit");
    return nvcv::ProtectCall(
        [&in, &out, &mu, &sigma, &handle, &stream, &per_channel, &seed]
        {
            nvcv::ImageBatchVarShapeWrapHandle input(in);
            nvcv::ImageBatchVarShapeWrapHandle output(out);
            nvcv::TensorWrapHandle             muwrap(mu);
            nvcv::TensorWrapHandle             sigmawrap(sigma);
            priv::ToDynamicRef<priv::GaussianNoise>(handle)(stream, input.resource(), output.resource(),
                                                            muwrap.resource(), sigmawrap.resource(),
                                                            static_cast<bool>(per_channel), seed);
        });
}
