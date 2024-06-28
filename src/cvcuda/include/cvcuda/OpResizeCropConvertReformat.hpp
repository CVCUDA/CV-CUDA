/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * @file OpResizeCropConvertReformat.hpp
 *
 * @brief Defines the public C++ class for that fuses resize, crop, data type conversion, channel manipulation, and layout reformat operations to optimize pipelines.
 * @defgroup NVCV_CPP_ALGORITHM__RESIZE_CROP ResizeCropConvertReformat
 * @{
 */

#ifndef CVCUDA__RESIZE_CROP_HPP
#define CVCUDA__RESIZE_CROP_HPP

#include "IOperator.hpp"
#include "OpResizeCropConvertReformat.h"

#include <cuda_runtime.h>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/ImageFormat.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/alloc/Requirements.hpp>

namespace cvcuda {

class ResizeCropConvertReformat final : public IOperator
{
public:
    explicit ResizeCropConvertReformat();

    ~ResizeCropConvertReformat();

    void operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out, const NVCVSize2D resizeDim,
                    const NVCVInterpolationType interpolation, const int2 cropPos,
                    const NVCVChannelManip manip = NVCV_CHANNEL_NO_OP, const float scale = 1, const float offset = 0,
                    const bool srcCast = true);

    void operator()(cudaStream_t stream, const nvcv::ImageBatchVarShape &in, const nvcv::Tensor &out,
                    const NVCVSize2D resizeDim, const NVCVInterpolationType interpolation, const int2 cropPos,
                    const NVCVChannelManip manip = NVCV_CHANNEL_NO_OP, const float scale = 1, const float offset = 0,
                    const bool srcCast = true);

    virtual NVCVOperatorHandle handle() const noexcept override;

private:
    NVCVOperatorHandle m_handle;
};

inline ResizeCropConvertReformat::ResizeCropConvertReformat()
{
    nvcv::detail::CheckThrow(cvcudaResizeCropConvertReformatCreate(&m_handle));
    assert(m_handle);
}

inline ResizeCropConvertReformat::~ResizeCropConvertReformat()
{
    nvcvOperatorDestroy(m_handle);
    m_handle = nullptr;
}

inline void ResizeCropConvertReformat::operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out,
                                                  const NVCVSize2D resizeDim, const NVCVInterpolationType interpolation,
                                                  const int2 cropPos, const NVCVChannelManip manip, const float scale,
                                                  const float offset, const bool srcCast)
{
    nvcv::detail::CheckThrow(cvcudaResizeCropConvertReformatSubmit(
        m_handle, stream, in.handle(), out.handle(), resizeDim, interpolation, cropPos, manip, scale, offset, srcCast));
}

inline void ResizeCropConvertReformat::operator()(cudaStream_t stream, const nvcv::ImageBatchVarShape &in,
                                                  const nvcv::Tensor &out, const NVCVSize2D resizeDim,
                                                  const NVCVInterpolationType interpolation, const int2 cropPos,
                                                  const NVCVChannelManip manip, const float scale, const float offset,
                                                  const bool srcCast)
{
    nvcv::detail::CheckThrow(cvcudaResizeCropConvertReformatVarShapeSubmit(
        m_handle, stream, in.handle(), out.handle(), resizeDim, interpolation, cropPos, manip, scale, offset, srcCast));
}

inline NVCVOperatorHandle ResizeCropConvertReformat::handle() const noexcept
{
    return m_handle;
}

} // namespace cvcuda

#endif // CVCUDA__RESIZE_CROP_HPP
