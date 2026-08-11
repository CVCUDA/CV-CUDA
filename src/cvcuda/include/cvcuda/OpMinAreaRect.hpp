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
 * @file OpMinAreaRect.hpp
 *
 * @brief Defines the public C++ Class for the MinAreaRect operation.
 * @defgroup NVCV_CPP_ALGORITHM__MIN_AREA_RECT MinAreaRect
 * @{
 */

#ifndef CVCUDA__MIN_AREA_RECT_HPP
#define CVCUDA__MIN_AREA_RECT_HPP

#include "IOperator.hpp"
#include "OpMinAreaRect.h"

#include <cuda_runtime.h>
#include <nvcv/ImageFormat.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/alloc/Requirements.hpp>

#include <cassert>

namespace cvcuda {

class MinAreaRect final : public IOperator
{
public:
    explicit MinAreaRect(int maxContourNum);

    void operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out,
                    const nvcv::Tensor &numPointsInContour, int totalContours) const;

    NVCVOperatorHandle handle() const noexcept override;

private:
    detail::OperatorHandle m_handle;
};

inline MinAreaRect::MinAreaRect(int maxContourNum)
{
    NVCVOperatorHandle h = nullptr;
    nvcv::detail::CheckThrow(cvcudaMinAreaRectCreate(&h, maxContourNum));
    assert(h);
    m_handle = detail::OperatorHandle{h};
}

inline void MinAreaRect::operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out,
                                    const nvcv::Tensor &numPointsInContour, const int totalContours) const
{
    nvcv::detail::CheckThrow(cvcudaMinAreaRectSubmit(m_handle.get(), stream, in.handle(), out.handle(),
                                                     numPointsInContour.handle(), totalContours));
}

inline NVCVOperatorHandle MinAreaRect::handle() const noexcept
{
    return m_handle.get();
}

} // namespace cvcuda

/** @} */

#endif // CVCUDA__MIN_AREA_RECT_HPP
