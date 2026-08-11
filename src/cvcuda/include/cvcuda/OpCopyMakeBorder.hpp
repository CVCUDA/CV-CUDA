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
 * @file OpCopyMakeBorder.hpp
 *
 * @brief Defines the public C++ class for the copy make border operation.
 * @defgroup NVCV_CPP_ALGORITHM_COPYMAKEBORDER Copy make border
 * @{
 */

#ifndef CVCUDA_COPYMAKEBORDER_HPP
#define CVCUDA_COPYMAKEBORDER_HPP

#include "IOperator.hpp"
#include "OpCopyMakeBorder.h"

#include <cuda_runtime.h>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/ImageFormat.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/alloc/Requirements.hpp>

#include <cassert>

namespace cvcuda {

class CopyMakeBorder final : public IOperator
{
public:
    explicit CopyMakeBorder();

    void operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out, int32_t top, int32_t left,
                    NVCVBorderType borderMode, const float4 borderValue) const;
    void operator()(cudaStream_t stream, const nvcv::ImageBatchVarShape &in, const nvcv::ImageBatchVarShape &out,
                    const nvcv::Tensor &top, const nvcv::Tensor &left, NVCVBorderType borderMode,
                    const float4 borderValue) const;
    void operator()(cudaStream_t stream, const nvcv::ImageBatchVarShape &in, const nvcv::Tensor &out,
                    const nvcv::Tensor &top, const nvcv::Tensor &left, NVCVBorderType borderMode,
                    const float4 borderValue) const;

    NVCVOperatorHandle handle() const noexcept override;

private:
    detail::OperatorHandle m_handle;
};

inline CopyMakeBorder::CopyMakeBorder()
{
    NVCVOperatorHandle h = nullptr;
    nvcv::detail::CheckThrow(cvcudaCopyMakeBorderCreate(&h));
    assert(h);
    m_handle = detail::OperatorHandle{h};
}

inline void CopyMakeBorder::operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out,
                                       int32_t top, int32_t left, NVCVBorderType borderMode,
                                       const float4 borderValue) const
{
    nvcv::detail::CheckThrow(cvcudaCopyMakeBorderSubmit(m_handle.get(), stream, in.handle(), out.handle(), top, left,
                                                        borderMode, borderValue));
}

inline void CopyMakeBorder::operator()(cudaStream_t stream, const nvcv::ImageBatchVarShape &in,
                                       const nvcv::ImageBatchVarShape &out, const nvcv::Tensor &top,
                                       const nvcv::Tensor &left, NVCVBorderType borderMode,
                                       const float4 borderValue) const
{
    nvcv::detail::CheckThrow(cvcudaCopyMakeBorderVarShapeSubmit(m_handle.get(), stream, in.handle(), out.handle(),
                                                                top.handle(), left.handle(), borderMode, borderValue));
}

inline void CopyMakeBorder::operator()(cudaStream_t stream, const nvcv::ImageBatchVarShape &in, const nvcv::Tensor &out,
                                       const nvcv::Tensor &top, const nvcv::Tensor &left, NVCVBorderType borderMode,
                                       const float4 borderValue) const
{
    nvcv::detail::CheckThrow(cvcudaCopyMakeBorderVarShapeStackSubmit(
        m_handle.get(), stream, in.handle(), out.handle(), top.handle(), left.handle(), borderMode, borderValue));
}

inline NVCVOperatorHandle CopyMakeBorder::handle() const noexcept
{
    return m_handle.get();
}

} // namespace cvcuda

/** @} */

#endif // CVCUDA_COPYMAKEBORDER_HPP
