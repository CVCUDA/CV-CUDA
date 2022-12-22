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
 * @file OpWarpPerspective.hpp
 *
 * @brief Defines the public C++ Class for the WarpPerspective operation.
 * @defgroup NVCV_CPP_ALGORITHM_WARP_PERSPECTIVE WarpPerspective
 * @{
 */

#ifndef CVCUDA_WARP_PERSPECTIVE_HPP
#define CVCUDA_WARP_PERSPECTIVE_HPP

#include "IOperator.hpp"
#include "OpWarpPerspective.h"
#include "Types.h"

#include <cuda_runtime.h>
#include <nvcv/IImageBatch.hpp>
#include <nvcv/ITensor.hpp>
#include <nvcv/ImageFormat.hpp>
#include <nvcv/alloc/Requirements.hpp>
#include <nvcv/cuda/math/LinAlg.hpp>

namespace cvmath = nvcv::cuda::math;

namespace cvcuda {

class WarpPerspective final : public IOperator
{
public:
    explicit WarpPerspective(const int32_t maxVarShapeBatchSize);

    ~WarpPerspective();

    void operator()(cudaStream_t stream, nvcv::ITensor &in, nvcv::ITensor &out,
                    const NVCVPerspectiveTransform transMatrix, const int32_t flags, const NVCVBorderType borderMode,
                    const float4 borderValue);

    void operator()(cudaStream_t stream, nvcv::IImageBatchVarShape &in, nvcv::IImageBatchVarShape &out,
                    nvcv::ITensor &transMatrix, const int32_t flags, const NVCVBorderType borderMode,
                    const float4 borderValue);

    virtual NVCVOperatorHandle handle() const noexcept override;

private:
    NVCVOperatorHandle m_handle;
};

inline WarpPerspective::WarpPerspective(const int32_t maxVarShapeBatchSize)
{
    nvcv::detail::CheckThrow(cvcudaWarpPerspectiveCreate(&m_handle, maxVarShapeBatchSize));
    assert(m_handle);
}

inline WarpPerspective::~WarpPerspective()
{
    nvcvOperatorDestroy(m_handle);
    m_handle = nullptr;
}

inline void WarpPerspective::operator()(cudaStream_t stream, nvcv::ITensor &in, nvcv::ITensor &out,
                                        const NVCVPerspectiveTransform transMatrix, const int32_t flags,
                                        const NVCVBorderType borderMode, const float4 borderValue)
{
    nvcv::detail::CheckThrow(cvcudaWarpPerspectiveSubmit(m_handle, stream, in.handle(), out.handle(), transMatrix,
                                                         flags, borderMode, borderValue));
}

inline void WarpPerspective::operator()(cudaStream_t stream, nvcv::IImageBatchVarShape &in,
                                        nvcv::IImageBatchVarShape &out, nvcv::ITensor &transMatrix, const int32_t flags,
                                        const NVCVBorderType borderMode, const float4 borderValue)
{
    nvcv::detail::CheckThrow(cvcudaWarpPerspectiveVarShapeSubmit(m_handle, stream, in.handle(), out.handle(),
                                                                 transMatrix.handle(), flags, borderMode, borderValue));
}

inline NVCVOperatorHandle WarpPerspective::handle() const noexcept
{
    return m_handle;
}

} // namespace cvcuda

#endif // CVCUDA_WARP_PERSPECTIVE_HPP
