/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <nvcv/ITensor.hpp>
#include <nvcv/ImageFormat.hpp>
#include <nvcv/alloc/Requirements.hpp>

namespace cvcuda {

class BoxBlur final : public IOperator
{
public:
    explicit BoxBlur();

    ~BoxBlur();

    void operator()(cudaStream_t stream, nvcv::ITensor &in, nvcv::ITensor &out, const NVCVBlurBoxesI bboxes);

    virtual NVCVOperatorHandle handle() const noexcept override;

private:
    NVCVOperatorHandle m_handle;
};

inline BoxBlur::BoxBlur()
{
    nvcv::detail::CheckThrow(cvcudaBoxBlurCreate(&m_handle));
    assert(m_handle);
}

inline BoxBlur::~BoxBlur()
{
    nvcvOperatorDestroy(m_handle);
    m_handle = nullptr;
}

inline void BoxBlur::operator()(cudaStream_t stream, nvcv::ITensor &in, nvcv::ITensor &out, const NVCVBlurBoxesI bboxes)
{
    nvcv::detail::CheckThrow(cvcudaBoxBlurSubmit(m_handle, stream, in.handle(), out.handle(), bboxes));
}

inline NVCVOperatorHandle BoxBlur::handle() const noexcept
{
    return m_handle;
}

} // namespace cvcuda

#endif // CVCUDA__BOX_BLUR_HPP
