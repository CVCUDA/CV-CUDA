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

/**
 * @file OpNormalize.hpp
 *
 * @brief Defines the public C++ Class for the normalize operation.
 * @defgroup NVCV_CPP_ALGORITHM_NORMALIZE Normalize
 * @{
 */

#ifndef CVCUDA_NORMALIZE_HPP
#define CVCUDA_NORMALIZE_HPP

#include "IOperator.hpp"
#include "OpNormalize.h"

#include <cuda_runtime.h>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/ImageFormat.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/alloc/Requirements.hpp>

#include <cassert>

namespace cvcuda {

class Normalize final : public IOperator
{
public:
    explicit Normalize();

    void operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &base, const nvcv::Tensor &scale,
                    const nvcv::Tensor &out, float global_scale, float shift, float epsilon, uint32_t flags = 0) const;

    void operator()(cudaStream_t stream, const nvcv::Tensor &in, const float4 base, const float4 scale,
                    int32_t baseChannels, int32_t scaleChannels, const nvcv::Tensor &out, float global_scale,
                    float shift, float epsilon, uint32_t flags = 0) const;

    void operator()(cudaStream_t stream, const nvcv::ImageBatch &in, const nvcv::Tensor &base,
                    const nvcv::Tensor &scale, const nvcv::ImageBatch &out, float global_scale, float shift,
                    float epsilon, uint32_t flags = 0) const;

    NVCVOperatorHandle handle() const noexcept override;

private:
    detail::OperatorHandle m_handle;
};

inline Normalize::Normalize()
{
    NVCVOperatorHandle h = nullptr;
    nvcv::detail::CheckThrow(cvcudaNormalizeCreate(&h));
    assert(h);
    m_handle = detail::OperatorHandle{h};
}

inline void Normalize::operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &base,
                                  const nvcv::Tensor &scale, const nvcv::Tensor &out, float global_scale, float shift,
                                  float epsilon, uint32_t flags) const
{
    nvcv::detail::CheckThrow(cvcudaNormalizeSubmit(m_handle.get(), stream, in.handle(), base.handle(), scale.handle(),
                                                   out.handle(), global_scale, shift, epsilon, flags));
}

inline void Normalize::operator()(cudaStream_t stream, const nvcv::Tensor &in, const float4 base, const float4 scale,
                                  int32_t baseChannels, int32_t scaleChannels, const nvcv::Tensor &out,
                                  float global_scale, float shift, float epsilon, uint32_t flags) const
{
    nvcv::detail::CheckThrow(cvcudaNormalizeScalarSubmit(m_handle.get(), stream, in.handle(), base, scale, baseChannels,
                                                         scaleChannels, out.handle(), global_scale, shift, epsilon,
                                                         flags));
}

inline void Normalize::operator()(cudaStream_t stream, const nvcv::ImageBatch &in, const nvcv::Tensor &base,
                                  const nvcv::Tensor &scale, const nvcv::ImageBatch &out, float global_scale,
                                  float shift, float epsilon, uint32_t flags) const
{
    nvcv::detail::CheckThrow(cvcudaNormalizeVarShapeSubmit(m_handle.get(), stream, in.handle(), base.handle(),
                                                           scale.handle(), out.handle(), global_scale, shift, epsilon,
                                                           flags));
}

inline NVCVOperatorHandle Normalize::handle() const noexcept
{
    return m_handle.get();
}

} // namespace cvcuda

/** @} */

#endif // CVCUDA_NORMALIZE_HPP
