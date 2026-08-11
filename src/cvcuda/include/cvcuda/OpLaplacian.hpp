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
#include <nvcv/ImageBatch.hpp>
#include <nvcv/ImageFormat.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/alloc/Requirements.hpp>

#include <cassert>

namespace cvcuda {

class Laplacian final : public IOperator
{
public:
    explicit Laplacian();

    void operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out, int32_t ksize, float scale,
                    NVCVBorderType borderMode) const;

    void operator()(cudaStream_t stream, const nvcv::ImageBatch &in, const nvcv::ImageBatch &out,
                    const nvcv::Tensor &ksize, const nvcv::Tensor &scale, NVCVBorderType borderMode) const;

    NVCVOperatorHandle handle() const noexcept override;

private:
    detail::OperatorHandle m_handle;
};

inline Laplacian::Laplacian()
{
    NVCVOperatorHandle h = nullptr;
    nvcv::detail::CheckThrow(cvcudaLaplacianCreate(&h));
    assert(h);
    m_handle = detail::OperatorHandle{h};
}

inline void Laplacian::operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out, int32_t ksize,
                                  float scale, NVCVBorderType borderMode) const
{
    nvcv::detail::CheckThrow(
        cvcudaLaplacianSubmit(m_handle.get(), stream, in.handle(), out.handle(), ksize, scale, borderMode));
}

inline void Laplacian::operator()(cudaStream_t stream, const nvcv::ImageBatch &in, const nvcv::ImageBatch &out,
                                  const nvcv::Tensor &ksize, const nvcv::Tensor &scale, NVCVBorderType borderMode) const
{
    nvcv::detail::CheckThrow(cvcudaLaplacianVarShapeSubmit(m_handle.get(), stream, in.handle(), out.handle(),
                                                           ksize.handle(), scale.handle(), borderMode));
}

inline NVCVOperatorHandle Laplacian::handle() const noexcept
{
    return m_handle.get();
}

} // namespace cvcuda

/** @} */

#endif // CVCUDA_LAPLACIAN_HPP
