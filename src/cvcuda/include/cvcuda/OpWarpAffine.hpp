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
 * @file OpWarpAffine.hpp
 *
 * @brief Defines the public C++ Class for the WarpAffine operation.
 * @defgroup NVCV_CPP_ALGORITHM_WARP_AFFINE WarpAffine
 * @{
 */

#ifndef CVCUDA_WARP_AFFINE_HPP
#define CVCUDA_WARP_AFFINE_HPP

#include "IOperator.hpp"
#include "OpWarpAffine.h"
#include "Types.h"

#include <cuda_runtime.h>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/ImageFormat.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/alloc/Requirements.hpp>

#include <cassert>

namespace cvcuda {

class WarpAffine final : public IOperator
{
public:
    explicit WarpAffine(const int32_t maxVarShapeBatchSize);

    void operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out,
                    const NVCVAffineTransform xform, const int32_t flags, const NVCVBorderType borderMode,
                    const float4 borderValue) const;

    void operator()(cudaStream_t stream, const nvcv::ImageBatchVarShape &in, const nvcv::ImageBatchVarShape &out,
                    const nvcv::Tensor &transMatrix, const int32_t flags, const NVCVBorderType borderMode,
                    const float4 borderValue) const;

    NVCVOperatorHandle handle() const noexcept override;

private:
    detail::OperatorHandle m_handle;
};

inline WarpAffine::WarpAffine(const int32_t maxVarShapeBatchSize)
{
    NVCVOperatorHandle h = nullptr;
    nvcv::detail::CheckThrow(cvcudaWarpAffineCreate(&h, maxVarShapeBatchSize));
    assert(h);
    m_handle = detail::OperatorHandle{h};
}

inline void WarpAffine::operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out,
                                   const NVCVAffineTransform xform, const int32_t flags,
                                   const NVCVBorderType borderMode, const float4 borderValue) const
{
    nvcv::detail::CheckThrow(cvcudaWarpAffineSubmit(m_handle.get(), stream, in.handle(), out.handle(), xform, flags,
                                                    borderMode, borderValue));
}

inline void WarpAffine::operator()(cudaStream_t stream, const nvcv::ImageBatchVarShape &in,
                                   const nvcv::ImageBatchVarShape &out, const nvcv::Tensor &transMatrix,
                                   const int32_t flags, const NVCVBorderType borderMode, const float4 borderValue) const
{
    nvcv::detail::CheckThrow(cvcudaWarpAffineVarShapeSubmit(m_handle.get(), stream, in.handle(), out.handle(),
                                                            transMatrix.handle(), flags, borderMode, borderValue));
}

inline NVCVOperatorHandle WarpAffine::handle() const noexcept
{
    return m_handle.get();
}

} // namespace cvcuda

/** @} */

#endif // CVCUDA_WARP_AFFINE_HPP
