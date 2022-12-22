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
 * @file OpConv2D.hpp
 *
 * @brief Defines the public C++ Class for the 2D convolution operation.
 * @defgroup NVCV_CPP_ALGORITHM_CONV2D 2D Convolution
 * @{
 */

#ifndef CVCUDA_CONV2D_HPP
#define CVCUDA_CONV2D_HPP

#include "IOperator.hpp"
#include "OpConv2D.h"

#include <cuda_runtime.h>
#include <nvcv/IImageBatch.hpp>
#include <nvcv/ITensor.hpp>
#include <nvcv/alloc/Requirements.hpp>

namespace cvcuda {

class Conv2D final : public IOperator
{
public:
    explicit Conv2D();

    ~Conv2D();

    void operator()(cudaStream_t stream, nvcv::IImageBatch &in, nvcv::IImageBatch &out, nvcv::IImageBatch &kernel,
                    nvcv::ITensor &kernelAnchor, NVCVBorderType borderMode);

    virtual NVCVOperatorHandle handle() const noexcept override;

private:
    NVCVOperatorHandle m_handle;
};

inline Conv2D::Conv2D()
{
    nvcv::detail::CheckThrow(cvcudaConv2DCreate(&m_handle));
    assert(m_handle);
}

inline Conv2D::~Conv2D()
{
    nvcvOperatorDestroy(m_handle);
    m_handle = nullptr;
}

inline void Conv2D::operator()(cudaStream_t stream, nvcv::IImageBatch &in, nvcv::IImageBatch &out,
                               nvcv::IImageBatch &kernel, nvcv::ITensor &kernelAnchor, NVCVBorderType borderMode)
{
    nvcv::detail::CheckThrow(cvcudaConv2DVarShapeSubmit(m_handle, stream, in.handle(), out.handle(), kernel.handle(),
                                                        kernelAnchor.handle(), borderMode));
}

inline NVCVOperatorHandle Conv2D::handle() const noexcept
{
    return m_handle;
}

} // namespace cvcuda

#endif // CVCUDA_CONV2D_HPP
