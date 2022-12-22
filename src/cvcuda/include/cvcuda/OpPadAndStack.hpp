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
#include <nvcv/IImageBatch.hpp>
#include <nvcv/ITensor.hpp>
#include <nvcv/ImageFormat.hpp>
#include <nvcv/alloc/Requirements.hpp>

namespace cvcuda {

class PadAndStack final : public IOperator
{
public:
    explicit PadAndStack();

    ~PadAndStack();

    void operator()(cudaStream_t stream, nvcv::IImageBatchVarShape &in, nvcv::ITensor &out, nvcv::ITensor &top,
                    nvcv::ITensor &left, NVCVBorderType borderMode, float borderValue);

    virtual NVCVOperatorHandle handle() const noexcept override;

private:
    NVCVOperatorHandle m_handle;
};

inline PadAndStack::PadAndStack()
{
    nvcv::detail::CheckThrow(cvcudaPadAndStackCreate(&m_handle));
    assert(m_handle);
}

inline PadAndStack::~PadAndStack()
{
    nvcvOperatorDestroy(m_handle);
    m_handle = nullptr;
}

inline void PadAndStack::operator()(cudaStream_t stream, nvcv::IImageBatchVarShape &in, nvcv::ITensor &out,
                                    nvcv::ITensor &top, nvcv::ITensor &left, NVCVBorderType borderMode,
                                    float borderValue)
{
    nvcv::detail::CheckThrow(cvcudaPadAndStackSubmit(m_handle, stream, in.handle(), out.handle(), top.handle(),
                                                     left.handle(), borderMode, borderValue));
}

inline NVCVOperatorHandle PadAndStack::handle() const noexcept
{
    return m_handle;
}

} // namespace cvcuda

#endif // CVCUDA_PADANDSTACK_HPP
