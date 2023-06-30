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

/**
 * @file OpGaussianNoise.hpp
 *
 * @brief Defines the public C++ Class for the GaussianNoise operation.
 * @defgroup NVCV_CPP_ALGORITHM_GAUSSIAN_NOISE GaussianNoise
 * @{
 */

#ifndef CVCUDA_GAUSSIAN_NOISE_HPP
#define CVCUDA_GAUSSIAN_NOISE_HPP

#include "IOperator.hpp"
#include "OpGaussianNoise.h"

#include <cuda_runtime.h>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/ImageFormat.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/alloc/Requirements.hpp>

namespace cvcuda {

class GaussianNoise final : public IOperator
{
public:
    explicit GaussianNoise(int maxBatchSize);

    ~GaussianNoise();

    void operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out, const nvcv::Tensor &mu,
                    const nvcv::Tensor &sigma, bool per_channel, unsigned long long seed);

    void operator()(cudaStream_t stream, const nvcv::ImageBatchVarShape &in, const nvcv::ImageBatchVarShape &out,
                    const nvcv::Tensor &mu, const nvcv::Tensor &sigma, bool per_channel, unsigned long long seed);

    virtual NVCVOperatorHandle handle() const noexcept override;

private:
    NVCVOperatorHandle m_handle;
};

inline GaussianNoise::GaussianNoise(int maxBatchSize)
{
    nvcv::detail::CheckThrow(cvcudaGaussianNoiseCreate(&m_handle, maxBatchSize));
    assert(m_handle);
}

inline GaussianNoise::~GaussianNoise()
{
    nvcvOperatorDestroy(m_handle);
    m_handle = nullptr;
}

inline void GaussianNoise::operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out,
                                      const nvcv::Tensor &mu, const nvcv::Tensor &sigma, bool per_channel,
                                      unsigned long long seed)
{
    nvcv::detail::CheckThrow(cvcudaGaussianNoiseSubmit(m_handle, stream, in.handle(), out.handle(), mu.handle(),
                                                       sigma.handle(), static_cast<int8_t>(per_channel), seed));
}

inline void GaussianNoise::operator()(cudaStream_t stream, const nvcv::ImageBatchVarShape &in,
                                      const nvcv::ImageBatchVarShape &out, const nvcv::Tensor &mu,
                                      const nvcv::Tensor &sigma, bool per_channel, unsigned long long seed)
{
    nvcv::detail::CheckThrow(cvcudaGaussianNoiseVarShapeSubmit(m_handle, stream, in.handle(), out.handle(), mu.handle(),
                                                               sigma.handle(), static_cast<int8_t>(per_channel), seed));
}

inline NVCVOperatorHandle GaussianNoise::handle() const noexcept
{
    return m_handle;
}

} // namespace cvcuda

#endif // CVCUDA_GAUSSIAN_NOISE_HPP
