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

namespace cvcuda {

class MinAreaRect final : public IOperator
{
public:
    explicit MinAreaRect(int maxContourNum);

    ~MinAreaRect();

    void operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out,
                    const nvcv::Tensor &numPointsInContour, int totalContours);

    virtual NVCVOperatorHandle handle() const noexcept override;

private:
    NVCVOperatorHandle m_handle;
};

inline MinAreaRect::MinAreaRect(int maxContourNum)
{
    nvcv::detail::CheckThrow(cvcudaMinAreaRectCreate(&m_handle, maxContourNum));
    assert(m_handle);
}

inline MinAreaRect::~MinAreaRect()
{
    nvcvOperatorDestroy(m_handle);
    m_handle = nullptr;
}

inline void MinAreaRect::operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out,
                                    const nvcv::Tensor &numPointsInContour, const int totalContours)
{
    nvcv::detail::CheckThrow(cvcudaMinAreaRectSubmit(m_handle, stream, in.handle(), out.handle(),
                                                     numPointsInContour.handle(), totalContours));
}

inline NVCVOperatorHandle MinAreaRect::handle() const noexcept
{
    return m_handle;
}

} // namespace cvcuda

#endif // CVCUDA__MIN_AREA_RECT_HPP
