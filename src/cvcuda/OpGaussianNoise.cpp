/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "priv/SymbolVersioning.hpp"

#include <nvcv/Exception.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>
#include <util/Assert.h>

namespace priv = cvcuda::priv;

CVCUDA_DEFINE_API(0, 4, NVCVStatus, cvcudaGaussianNoiseCreate, (NVCVOperatorHandle * handle, int maxBatchSize))
{
    return nvcv::ProtectCall(
        [&]
        {
            if (handle == nullptr)
            {
                throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                      "Pointer to NVCVOperator handle must not be NULL");
            }

            *handle = reinterpret_cast<NVCVOperatorHandle>(new priv::GaussianNoise(maxBatchSize));
        });
}

CVCUDA_DEFINE_API(0, 4, NVCVStatus, cvcudaGaussianNoiseSubmit,
                  (NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle in, NVCVTensorHandle out,
                   NVCVTensorHandle mu, NVCVTensorHandle sigma, int8_t per_channel, unsigned long long seed))
{
    return nvcv::ProtectCall(
        [&]
        {
            nvcv::TensorWrapHandle input(in), output(out), muwrap(mu), sigmawrap(sigma);
            priv::ToDynamicRef<priv::GaussianNoise>(handle)(stream, input, output, muwrap, sigmawrap,
                                                            static_cast<bool>(per_channel), seed);
        });
}

CVCUDA_DEFINE_API(0, 4, NVCVStatus, cvcudaGaussianNoiseVarShapeSubmit,
                  (NVCVOperatorHandle handle, cudaStream_t stream, NVCVImageBatchHandle in, NVCVImageBatchHandle out,
                   NVCVTensorHandle mu, NVCVTensorHandle sigma, int8_t per_channel, unsigned long long seed))
{
    return nvcv::ProtectCall(
        [&]
        {
            nvcv::ImageBatchVarShapeWrapHandle input(in), output(out);
            nvcv::TensorWrapHandle             muwrap(mu), sigmawrap(sigma);
            priv::ToDynamicRef<priv::GaussianNoise>(handle)(stream, input, output, muwrap, sigmawrap,
                                                            static_cast<bool>(per_channel), seed);
        });
}
