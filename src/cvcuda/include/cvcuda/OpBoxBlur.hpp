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
 * @file OpBoxBlur.hpp
 *
 * @brief Defines the public C++ Class for the BoxBlur operation.
 * @defgroup NVCV_CPP_ALGORITHM__BOX_BLUR BoxBlur
 * @{
 */

#ifndef CVCUDA__BOX_BLUR_HPP
#define CVCUDA__BOX_BLUR_HPP

#include "IOperator.hpp"
#include "OpBoxBlur.h"

#include <cuda_runtime.h>
#include <nvcv/ImageFormat.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/alloc/Requirements.hpp>

#include <cassert>

namespace cvcuda {

class BoxBlur final : public IOperator
{
public:
    explicit BoxBlur();

    void operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out,
                    const NVCVBlurBoxesI bboxes) const;

    NVCVOperatorHandle handle() const noexcept override;

private:
    detail::OperatorHandle m_handle;
};

inline BoxBlur::BoxBlur()
{
    NVCVOperatorHandle h = nullptr;
    nvcv::detail::CheckThrow(cvcudaBoxBlurCreate(&h));
    assert(h);
    m_handle = detail::OperatorHandle{h};
}

inline void BoxBlur::operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out,
                                const NVCVBlurBoxesI bboxes) const
{
    nvcv::detail::CheckThrow(cvcudaBoxBlurSubmit(m_handle.get(), stream, in.handle(), out.handle(), bboxes));
}

inline NVCVOperatorHandle BoxBlur::handle() const noexcept
{
    return m_handle.get();
}

} // namespace cvcuda

/** @} */

#endif // CVCUDA__BOX_BLUR_HPP
