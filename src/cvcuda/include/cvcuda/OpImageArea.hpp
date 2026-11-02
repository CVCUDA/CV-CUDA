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
 * @file OpImageArea.hpp
 *
 * @brief Defines the public C++ Class for the ImageArea operation.
 * @defgroup NVCV_CPP_ALGORITHM__IMAGEAREA ImageArea
 * @{
 */

#ifndef CVCUDA__IMAGEAREA_HPP
#define CVCUDA__IMAGEAREA_HPP

#include "IOperator.hpp"
#include "OpImageArea.h"

#include <cuda_runtime.h>
#include <nvcv/Tensor.hpp>
#include <nvcv/alloc/Requirements.hpp>

namespace cvcuda {

class ImageArea final : public IOperator
{
public:
    explicit ImageArea();

    void operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out) const;

    NVCVOperatorHandle handle() const noexcept override;

private:
    detail::OperatorHandle m_handle;
};

inline ImageArea::ImageArea()
{
    NVCVOperatorHandle h = nullptr;
    nvcv::detail::CheckThrow(cvcudaImageAreaCreate(&h));
    assert(h);
    m_handle = detail::OperatorHandle{h};
}

inline void ImageArea::operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out) const
{
    nvcv::detail::CheckThrow(cvcudaImageAreaSubmit(m_handle.get(), stream, in.handle(), out.handle()));
}

inline NVCVOperatorHandle ImageArea::handle() const noexcept
{
    return m_handle.get();
}

} // namespace cvcuda

#endif // CVCUDA__IMAGEAREA_HPP
