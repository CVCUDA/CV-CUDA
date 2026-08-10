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
 * @file OpGaussian.hpp
 *
 * @brief Defines the public C++ class for the Gaussian operation.
 * @defgroup NVCV_CPP_ALGORITHM_GAUSSIAN Gaussian
 * @{
 */

#ifndef CVCUDA_GAUSSIAN_HPP
#define CVCUDA_GAUSSIAN_HPP

#include "IOperator.hpp"
#include "OpGaussian.h"

#include <cuda_runtime.h>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/ImageFormat.hpp>
#include <nvcv/Size.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/alloc/Requirements.hpp>

#include <cassert>

namespace cvcuda {

class Gaussian final : public IOperator
{
public:
    explicit Gaussian(nvcv::Size2D maxKernelSize, int32_t maxVarShapeBatchSize);

    void operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out, nvcv::Size2D kernelSize,
                    double2 sigma, NVCVBorderType borderMode) const;

    void operator()(cudaStream_t stream, const nvcv::ImageBatch &in, const nvcv::ImageBatch &out,
                    const nvcv::Tensor &kernelSize, const nvcv::Tensor &sigma, NVCVBorderType borderMode) const;

    NVCVOperatorHandle handle() const noexcept override;

private:
    detail::OperatorHandle m_handle;
};

inline Gaussian::Gaussian(nvcv::Size2D maxKernelSize, int32_t maxVarShapeBatchSize)
{
    NVCVOperatorHandle h = nullptr;
    nvcv::detail::CheckThrow(cvcudaGaussianCreate(&h, maxKernelSize.w, maxKernelSize.h, maxVarShapeBatchSize));
    assert(h);
    m_handle = detail::OperatorHandle{h};
}

inline void Gaussian::operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out,
                                 nvcv::Size2D kernelSize, double2 sigma, NVCVBorderType borderMode) const
{
    nvcv::detail::CheckThrow(cvcudaGaussianSubmit(m_handle.get(), stream, in.handle(), out.handle(), kernelSize.w,
                                                  kernelSize.h, sigma.x, sigma.y, borderMode));
}

inline void Gaussian::operator()(cudaStream_t stream, const nvcv::ImageBatch &in, const nvcv::ImageBatch &out,
                                 const nvcv::Tensor &kernelSize, const nvcv::Tensor &sigma,
                                 NVCVBorderType borderMode) const
{
    nvcv::detail::CheckThrow(cvcudaGaussianVarShapeSubmit(m_handle.get(), stream, in.handle(), out.handle(),
                                                          kernelSize.handle(), sigma.handle(), borderMode));
}

inline NVCVOperatorHandle Gaussian::handle() const noexcept
{
    return m_handle.get();
}

} // namespace cvcuda

/** @} */

#endif // CVCUDA_GAUSSIAN_HPP
