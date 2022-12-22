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
#include <nvcv/IImageBatch.hpp>
#include <nvcv/ITensor.hpp>
#include <nvcv/ImageFormat.hpp>
#include <nvcv/alloc/Requirements.hpp>

namespace cvcuda {

class Normalize final : public IOperator
{
public:
    explicit Normalize();

    ~Normalize();

    void operator()(cudaStream_t stream, nvcv::ITensor &in, nvcv::ITensor &base, nvcv::ITensor &scale,
                    nvcv::ITensor &out, float global_scale, float shift, float epsilon, uint32_t flags = 0);

    void operator()(cudaStream_t stream, nvcv::IImageBatch &in, nvcv::ITensor &base, nvcv::ITensor &scale,
                    nvcv::IImageBatch &out, float global_scale, float shift, float epsilon, uint32_t flags = 0);

    virtual NVCVOperatorHandle handle() const noexcept override;

private:
    NVCVOperatorHandle m_handle;
};

inline Normalize::Normalize()
{
    nvcv::detail::CheckThrow(cvcudaNormalizeCreate(&m_handle));
    assert(m_handle);
}

inline Normalize::~Normalize()
{
    nvcvOperatorDestroy(m_handle);
    m_handle = nullptr;
}

inline void Normalize::operator()(cudaStream_t stream, nvcv::ITensor &in, nvcv::ITensor &base, nvcv::ITensor &scale,
                                  nvcv::ITensor &out, float global_scale, float shift, float epsilon, uint32_t flags)
{
    nvcv::detail::CheckThrow(cvcudaNormalizeSubmit(m_handle, stream, in.handle(), base.handle(), scale.handle(),
                                                   out.handle(), global_scale, shift, epsilon, flags));
}

inline void Normalize::operator()(cudaStream_t stream, nvcv::IImageBatch &in, nvcv::ITensor &base, nvcv::ITensor &scale,
                                  nvcv::IImageBatch &out, float global_scale, float shift, float epsilon,
                                  uint32_t flags)
{
    nvcv::detail::CheckThrow(cvcudaNormalizeVarShapeSubmit(m_handle, stream, in.handle(), base.handle(), scale.handle(),
                                                           out.handle(), global_scale, shift, epsilon, flags));
}

inline NVCVOperatorHandle Normalize::handle() const noexcept
{
    return m_handle;
}

} // namespace cvcuda

#endif // CVCUDA_NORMALIZE_HPP
