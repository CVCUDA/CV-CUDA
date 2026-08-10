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
#include <nvcv/ImageBatch.hpp>
#include <nvcv/ImageFormat.hpp>
#include <nvcv/Size.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/alloc/Requirements.hpp>

#include <cassert>

namespace cvcuda {

class Flip final : public IOperator
{
public:
    explicit Flip(int32_t maxVarShapeBatchSize = 0);

    void operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out, int32_t flipCode) const;
    void operator()(cudaStream_t stream, const nvcv::ImageBatch &in, const nvcv::ImageBatch &out,
                    const nvcv::Tensor &flipCode) const;

    NVCVOperatorHandle handle() const noexcept override;

private:
    detail::OperatorHandle m_handle;
};

inline Flip::Flip(int32_t maxVarShapeBatchSize)
{
    NVCVOperatorHandle h = nullptr;
    nvcv::detail::CheckThrow(cvcudaFlipCreate(&h, maxVarShapeBatchSize));
    assert(h);
    m_handle = detail::OperatorHandle{h};
}

inline void Flip::operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out,
                             int32_t flipCode) const
{
    nvcv::detail::CheckThrow(cvcudaFlipSubmit(m_handle.get(), stream, in.handle(), out.handle(), flipCode));
}

inline void Flip::operator()(cudaStream_t stream, const nvcv::ImageBatch &in, const nvcv::ImageBatch &out,
                             const nvcv::Tensor &flipCode) const
{
    nvcv::detail::CheckThrow(
        cvcudaFlipVarShapeSubmit(m_handle.get(), stream, in.handle(), out.handle(), flipCode.handle()));
}

inline NVCVOperatorHandle Flip::handle() const noexcept
{
    return m_handle.get();
}

} // namespace cvcuda

/** @} */

#endif // CVCUDA_FLIP_HPP
