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
 * @file OpFlip.hpp
 *
 * @brief Defines the public C++ class for the Flip operation.
 * @defgroup NVCV_CPP_ALGORITHM_FLIP Flip
 * @{
 */

#ifndef CVCUDA_FLIP_HPP
#define CVCUDA_FLIP_HPP

#include "IOperator.hpp"
#include "OpFlip.h"

#include <cuda_runtime.h>
#include <nvcv/IImageBatch.hpp>
#include <nvcv/ITensor.hpp>
#include <nvcv/ImageFormat.hpp>
#include <nvcv/Size.hpp>
#include <nvcv/alloc/Requirements.hpp>

namespace cvcuda {

class Flip final : public IOperator
{
public:
    explicit Flip(int32_t maxVarShapeBatchSize = 0);
    ~Flip();

    void operator()(cudaStream_t stream, nvcv::ITensor &in, nvcv::ITensor &out, int32_t flipCode);
    void operator()(cudaStream_t stream, nvcv::IImageBatch &in, nvcv::IImageBatch &out, nvcv::ITensor &flipCode);

    virtual NVCVOperatorHandle handle() const noexcept override;

private:
    NVCVOperatorHandle m_handle;
};

inline Flip::Flip(int32_t maxVarShapeBatchSize)
{
    nvcv::detail::CheckThrow(cvcudaFlipCreate(&m_handle, maxVarShapeBatchSize));
    assert(m_handle);
}

inline Flip::~Flip()
{
    nvcvOperatorDestroy(m_handle);
    m_handle = nullptr;
}

inline void Flip::operator()(cudaStream_t stream, nvcv::ITensor &in, nvcv::ITensor &out, int32_t flipCode)
{
    nvcv::detail::CheckThrow(cvcudaFlipSubmit(m_handle, stream, in.handle(), out.handle(), flipCode));
}

inline void Flip::operator()(cudaStream_t stream, nvcv::IImageBatch &in, nvcv::IImageBatch &out,
                             nvcv::ITensor &flipCode)
{
    nvcv::detail::CheckThrow(cvcudaFlipVarShapeSubmit(m_handle, stream, in.handle(), out.handle(), flipCode.handle()));
}

inline NVCVOperatorHandle Flip::handle() const noexcept
{
    return m_handle;
}

} // namespace cvcuda

#endif // CVCUDA_FLIP_HPP
