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
 * @file OpLaplacian.hpp
 *
 * @brief Defines the public C++ class for the Laplacian operation.
 * @defgroup NVCV_CPP_ALGORITHM_LAPLACIAN Laplacian
 * @{
 */

#ifndef CVCUDA_LAPLACIAN_HPP
#define CVCUDA_LAPLACIAN_HPP

#include "IOperator.hpp"
#include "OpLaplacian.h"

#include <cuda_runtime.h>
#include <nvcv/IImageBatch.hpp>
#include <nvcv/ITensor.hpp>
#include <nvcv/ImageFormat.hpp>
#include <nvcv/alloc/Requirements.hpp>

namespace cvcuda {

class Laplacian final : public IOperator
{
public:
    explicit Laplacian();

    ~Laplacian();

    void operator()(cudaStream_t stream, nvcv::ITensor &in, nvcv::ITensor &out, int32_t ksize, float scale,
                    NVCVBorderType borderMode);

    void operator()(cudaStream_t stream, nvcv::IImageBatch &in, nvcv::IImageBatch &out, nvcv::ITensor &ksize,
                    nvcv::ITensor &scale, NVCVBorderType borderMode);

    virtual NVCVOperatorHandle handle() const noexcept override;

private:
    NVCVOperatorHandle m_handle;
};

inline Laplacian::Laplacian()
{
    nvcv::detail::CheckThrow(cvcudaLaplacianCreate(&m_handle));
    assert(m_handle);
}

inline Laplacian::~Laplacian()
{
    nvcvOperatorDestroy(m_handle);
    m_handle = nullptr;
}

inline void Laplacian::operator()(cudaStream_t stream, nvcv::ITensor &in, nvcv::ITensor &out, int32_t ksize,
                                  float scale, NVCVBorderType borderMode)
{
    nvcv::detail::CheckThrow(
        cvcudaLaplacianSubmit(m_handle, stream, in.handle(), out.handle(), ksize, scale, borderMode));
}

inline void Laplacian::operator()(cudaStream_t stream, nvcv::IImageBatch &in, nvcv::IImageBatch &out,
                                  nvcv::ITensor &ksize, nvcv::ITensor &scale, NVCVBorderType borderMode)
{
    nvcv::detail::CheckThrow(cvcudaLaplacianVarShapeSubmit(m_handle, stream, in.handle(), out.handle(), ksize.handle(),
                                                           scale.handle(), borderMode));
}

inline NVCVOperatorHandle Laplacian::handle() const noexcept
{
    return m_handle;
}

} // namespace cvcuda

#endif // CVCUDA_LAPLACIAN_HPP
