/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * @file OpFindContours.hpp
 *
 * @brief Defines the public C++ Class for the resize operation.
 * @defgroup NVCV_CPP_ALGORITHM_FIND_CONTOURS Find Contours
 * @{
 */

#ifndef CVCUDA_FIND_CONTOURS_HPP
#define CVCUDA_FIND_CONTOURS_HPP

#include "IOperator.hpp"
#include "OpFindContours.h"

#include <cuda_runtime.h>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/ImageFormat.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/alloc/Requirements.hpp>

namespace cvcuda {

class FindContours final : public IOperator
{
public:
    static constexpr int32_t MAX_NUM_CONTOURS   = 256;
    static constexpr int32_t MAX_CONTOUR_POINTS = 4 * 1024;
    static constexpr int32_t MAX_TOTAL_POINTS   = MAX_NUM_CONTOURS * MAX_CONTOUR_POINTS;

    explicit FindContours() = delete;
    explicit FindContours(nvcv::Size2D maxSize, int32_t maxBatchSize);

    ~FindContours();

    void operator()(cudaStream_t stream, nvcv::Tensor &in, nvcv::Tensor &points, nvcv::Tensor &numPoints);

    virtual NVCVOperatorHandle handle() const noexcept override;

private:
    NVCVOperatorHandle m_handle;
};

inline FindContours::FindContours(nvcv::Size2D maxSize, int32_t maxBatchSize)
{
    nvcv::detail::CheckThrow(cvcudaFindContoursCreate(&m_handle, maxSize.w, maxSize.h, maxBatchSize));
    assert(m_handle);
}

inline FindContours::~FindContours()
{
    nvcvOperatorDestroy(m_handle);
    m_handle = nullptr;
}

inline void FindContours::operator()(cudaStream_t stream, nvcv::Tensor &in, nvcv::Tensor &points,
                                     nvcv::Tensor &numPoints)
{
    nvcv::detail::CheckThrow(
        cvcudaFindContoursSubmit(m_handle, stream, in.handle(), points.handle(), numPoints.handle()));
}

inline NVCVOperatorHandle FindContours::handle() const noexcept
{
    return m_handle;
}

} // namespace cvcuda

#endif // CVCUDA_FIND_CONTOURS_HPP
