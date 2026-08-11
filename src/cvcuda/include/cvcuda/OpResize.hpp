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
 * @file OpResize.hpp
 *
 * @brief Defines the public C++ Class for the resize operation.
 * @defgroup NVCV_CPP_ALGORITHM_RESIZE Resize
 * @{
 */

#ifndef CVCUDA_RESIZE_HPP
#define CVCUDA_RESIZE_HPP

#include "IOperator.hpp"
#include "OpResize.h"

#include <cuda_runtime.h>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/ImageFormat.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/alloc/Requirements.hpp>

#include <cassert>

namespace cvcuda {

class Resize final : public IOperator
{
public:
    explicit Resize();

    void operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out,
                    const NVCVInterpolationType interpolation) const;
    void operator()(cudaStream_t stream, const nvcv::ImageBatchVarShape &in, const nvcv::ImageBatchVarShape &out,
                    const NVCVInterpolationType interpolation) const;

    NVCVOperatorHandle handle() const noexcept override;

private:
    detail::OperatorHandle m_handle;
};

inline Resize::Resize()
{
    NVCVOperatorHandle h = nullptr;
    nvcv::detail::CheckThrow(cvcudaResizeCreate(&h));
    assert(h);
    m_handle = detail::OperatorHandle{h};
}

inline void Resize::operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out,
                               const NVCVInterpolationType interpolation) const
{
    nvcv::detail::CheckThrow(cvcudaResizeSubmit(m_handle.get(), stream, in.handle(), out.handle(), interpolation));
}

inline void Resize::operator()(cudaStream_t stream, const nvcv::ImageBatchVarShape &in,
                               const nvcv::ImageBatchVarShape &out, const NVCVInterpolationType interpolation) const
{
    nvcv::detail::CheckThrow(
        cvcudaResizeVarShapeSubmit(m_handle.get(), stream, in.handle(), out.handle(), interpolation));
}

inline NVCVOperatorHandle Resize::handle() const noexcept
{
    return m_handle.get();
}

} // namespace cvcuda

/** @} */

#endif // CVCUDA_RESIZE_HPP
