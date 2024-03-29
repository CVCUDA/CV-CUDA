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
 * @file OpInpaint.hpp
 *
 * @brief Defines the public C++ Class for the inpaint operation.
 * @defgroup NVCV_CPP_ALGORITHM_INPAINT Inpaint
 * @{
 */

#ifndef CVCUDA_INPAINT_HPP
#define CVCUDA_INPAINT_HPP

#include "IOperator.hpp"
#include "OpInpaint.h"

#include <cuda_runtime.h>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/ImageFormat.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/alloc/Requirements.hpp>

namespace cvcuda {

class Inpaint final : public IOperator
{
public:
    explicit Inpaint(int32_t maxBatchSize, nvcv::Size2D maxShape);

    ~Inpaint();

    void operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &masks, const nvcv::Tensor &out,
                    double inpaintRadius);

    void operator()(cudaStream_t stream, const nvcv::ImageBatchVarShape &in, const nvcv::ImageBatchVarShape &masks,
                    const nvcv::ImageBatchVarShape &out, double inpaintRadius);

    virtual NVCVOperatorHandle handle() const noexcept override;

private:
    NVCVOperatorHandle m_handle;
};

inline Inpaint::Inpaint(int32_t maxBatchSize, nvcv::Size2D maxShape)
{
    nvcv::detail::CheckThrow(cvcudaInpaintCreate(&m_handle, maxBatchSize, maxShape.h, maxShape.w));
    assert(m_handle);
}

inline Inpaint::~Inpaint()
{
    nvcvOperatorDestroy(m_handle);
    m_handle = nullptr;
}

inline void Inpaint::operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &masks,
                                const nvcv::Tensor &out, double inpaintRadius)
{
    nvcv::detail::CheckThrow(
        cvcudaInpaintSubmit(m_handle, stream, in.handle(), masks.handle(), out.handle(), inpaintRadius));
}

inline void Inpaint::operator()(cudaStream_t stream, const nvcv::ImageBatchVarShape &in,
                                const nvcv::ImageBatchVarShape &masks, const nvcv::ImageBatchVarShape &out,
                                double inpaintRadius)
{
    nvcv::detail::CheckThrow(
        cvcudaInpaintVarShapeSubmit(m_handle, stream, in.handle(), masks.handle(), out.handle(), inpaintRadius));
}

inline NVCVOperatorHandle Inpaint::handle() const noexcept
{
    return m_handle;
}

} // namespace cvcuda

#endif // CVCUDA_INPAINT_HPP
