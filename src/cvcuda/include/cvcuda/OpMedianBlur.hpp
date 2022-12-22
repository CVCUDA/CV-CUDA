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
 * @file OpMedianBlur.hpp
 *
 * @brief Defines the public C++ Class for the median blur operation.
 * @defgroup NVCV_CPP_ALGORITHM_MEDIAN_BLUR MedianBlur
 * @{
 */

#ifndef CVCUDA_MEDIAN_BLUR_HPP
#define CVCUDA_MEDIAN_BLUR_HPP

#include "IOperator.hpp"
#include "OpMedianBlur.h"

#include <cuda_runtime.h>
#include <nvcv/IImageBatch.hpp>
#include <nvcv/ITensor.hpp>
#include <nvcv/ImageFormat.hpp>
#include <nvcv/alloc/Requirements.hpp>

namespace cvcuda {

class MedianBlur final : public IOperator
{
public:
    explicit MedianBlur(const int maxVarShapeBatchSize);

    ~MedianBlur();

    void operator()(cudaStream_t stream, nvcv::ITensor &in, nvcv::ITensor &out, const nvcv::Size2D ksize);

    void operator()(cudaStream_t stream, nvcv::IImageBatchVarShape &in, nvcv::IImageBatchVarShape &out,
                    nvcv::ITensor &ksize);

    virtual NVCVOperatorHandle handle() const noexcept override;

private:
    NVCVOperatorHandle m_handle;
};

inline MedianBlur::MedianBlur(const int maxVarShapeBatchSize)
{
    nvcv::detail::CheckThrow(cvcudaMedianBlurCreate(&m_handle, maxVarShapeBatchSize));
    assert(m_handle);
}

inline MedianBlur::~MedianBlur()
{
    nvcvOperatorDestroy(m_handle);
    m_handle = nullptr;
}

inline void MedianBlur::operator()(cudaStream_t stream, nvcv::ITensor &in, nvcv::ITensor &out, const nvcv::Size2D ksize)
{
    nvcv::detail::CheckThrow(cvcudaMedianBlurSubmit(m_handle, stream, in.handle(), out.handle(), ksize.w, ksize.h));
}

inline void MedianBlur::operator()(cudaStream_t stream, nvcv::IImageBatchVarShape &in, nvcv::IImageBatchVarShape &out,
                                   nvcv::ITensor &ksize)
{
    nvcv::detail::CheckThrow(
        cvcudaMedianBlurVarShapeSubmit(m_handle, stream, in.handle(), out.handle(), ksize.handle()));
}

inline NVCVOperatorHandle MedianBlur::handle() const noexcept
{
    return m_handle;
}

} // namespace cvcuda

#endif // CVCUDA_MEDIAN_BLUR_HPP
