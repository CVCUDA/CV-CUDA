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
 * @file OpPadAndStack.hpp
 *
 * @brief Defines the public C++ class for the pad and stack operation.
 * @defgroup NVCV_CPP_ALGORITHM_PADANDSTACK Pad and stack
 * @{
 */

#ifndef CVCUDA_PADANDSTACK_HPP
#define CVCUDA_PADANDSTACK_HPP

#include "IOperator.hpp"
#include "OpPadAndStack.h"

#include <cuda_runtime.h>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/ImageFormat.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/alloc/Requirements.hpp>

#include <cassert>

namespace cvcuda {

class PadAndStack final : public IOperator
{
public:
    explicit PadAndStack();

    void operator()(cudaStream_t stream, const nvcv::ImageBatchVarShape &in, const nvcv::Tensor &out,
                    const nvcv::Tensor &top, const nvcv::Tensor &left, NVCVBorderType borderMode,
                    float borderValue) const;

    NVCVOperatorHandle handle() const noexcept override;

private:
    detail::OperatorHandle m_handle;
};

inline PadAndStack::PadAndStack()
{
    NVCVOperatorHandle h = nullptr;
    nvcv::detail::CheckThrow(cvcudaPadAndStackCreate(&h));
    assert(h);
    m_handle = detail::OperatorHandle{h};
}

inline void PadAndStack::operator()(cudaStream_t stream, const nvcv::ImageBatchVarShape &in, const nvcv::Tensor &out,
                                    const nvcv::Tensor &top, const nvcv::Tensor &left, NVCVBorderType borderMode,
                                    float borderValue) const
{
    nvcv::detail::CheckThrow(cvcudaPadAndStackSubmit(m_handle.get(), stream, in.handle(), out.handle(), top.handle(),
                                                     left.handle(), borderMode, borderValue));
}

inline NVCVOperatorHandle PadAndStack::handle() const noexcept
{
    return m_handle.get();
}

} // namespace cvcuda

/** @} */

#endif // CVCUDA_PADANDSTACK_HPP
