/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * @file OpAdjustSharpness.hpp
 *
 * @brief Defines the public C++ Class for the AdjustSharpness operation.
 * @defgroup NVCV_CPP_ALGORITHM__ADJUST_SHARPNESS AdjustSharpness
 * @{
 */

#ifndef CVCUDA__ADJUST_SHARPNESS_HPP
#define CVCUDA__ADJUST_SHARPNESS_HPP

#include "IOperator.hpp"
#include "OpAdjustSharpness.h"

#include <cuda_runtime.h>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/ImageFormat.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/alloc/Requirements.hpp>

namespace cvcuda {

class AdjustSharpness final : public IOperator
{
public:
    explicit AdjustSharpness();

    void operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out, float sharpnessFactor) const;

    void operator()(cudaStream_t stream, const nvcv::ImageBatch &in, const nvcv::ImageBatch &out,
                    float sharpnessFactor) const;

    NVCVOperatorHandle handle() const noexcept override;

private:
    detail::OperatorHandle m_handle;
};

inline AdjustSharpness::AdjustSharpness()
{
    NVCVOperatorHandle h = nullptr;
    nvcv::detail::CheckThrow(cvcudaAdjustSharpnessCreate(&h));
    assert(h);
    m_handle = detail::OperatorHandle{h};
}

inline void AdjustSharpness::operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out,
                                        float sharpnessFactor) const
{
    nvcv::detail::CheckThrow(
        cvcudaAdjustSharpnessSubmit(m_handle.get(), stream, in.handle(), out.handle(), sharpnessFactor));
}

inline void AdjustSharpness::operator()(cudaStream_t stream, const nvcv::ImageBatch &in, const nvcv::ImageBatch &out,
                                        float sharpnessFactor) const
{
    nvcv::detail::CheckThrow(
        cvcudaAdjustSharpnessVarShapeSubmit(m_handle.get(), stream, in.handle(), out.handle(), sharpnessFactor));
}

inline NVCVOperatorHandle AdjustSharpness::handle() const noexcept
{
    return m_handle.get();
}

} // namespace cvcuda

#endif // CVCUDA__ADJUST_SHARPNESS_HPP
