/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * @file OpPosterize.hpp
 *
 * @brief Defines the public C++ Class for the Posterize operation.
 * @defgroup NVCV_CPP_ALGORITHM__POSTERIZE Posterize
 * @{
 */

#ifndef CVCUDA__POSTERIZE_HPP
#define CVCUDA__POSTERIZE_HPP

#include "IOperator.hpp"
#include "OpPosterize.h"

#include <cuda_runtime.h>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/ImageFormat.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/alloc/Requirements.hpp>

namespace cvcuda {

class Posterize final : public IOperator
{
public:
    explicit Posterize();

    void operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out, int32_t bits) const;

    void operator()(cudaStream_t stream, const nvcv::ImageBatchVarShape &in, const nvcv::ImageBatchVarShape &out,
                    int32_t bits) const;

    NVCVOperatorHandle handle() const noexcept override;

private:
    detail::OperatorHandle m_handle;
};

inline Posterize::Posterize()
{
    NVCVOperatorHandle h = nullptr;
    nvcv::detail::CheckThrow(cvcudaPosterizeCreate(&h));
    assert(h);
    m_handle = detail::OperatorHandle{h};
}

inline void Posterize::operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out,
                                  int32_t bits) const
{
    nvcv::detail::CheckThrow(cvcudaPosterizeSubmit(m_handle.get(), stream, in.handle(), out.handle(), bits));
}

inline void Posterize::operator()(cudaStream_t stream, const nvcv::ImageBatchVarShape &in,
                                  const nvcv::ImageBatchVarShape &out, int32_t bits) const
{
    nvcv::detail::CheckThrow(cvcudaPosterizeVarShapeSubmit(m_handle.get(), stream, in.handle(), out.handle(), bits));
}

inline NVCVOperatorHandle Posterize::handle() const noexcept
{
    return m_handle.get();
}

} // namespace cvcuda

#endif // CVCUDA__POSTERIZE_HPP
