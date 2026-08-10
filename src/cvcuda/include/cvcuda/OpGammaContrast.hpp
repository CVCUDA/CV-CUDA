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
 * @file OpGammaContrast.hpp
 *
 * @brief Defines the public C++ Class for the Gamma Contrast operation.
 * @defgroup NVCV_CPP_ALGORITHM_GAMMA_CONTRAST
 * @{
 */

#ifndef CVCUDA_GAMMA_CONTRAST_HPP
#define CVCUDA_GAMMA_CONTRAST_HPP

#include "IOperator.hpp"
#include "OpGammaContrast.h"

#include <cuda_runtime.h>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/alloc/Requirements.hpp>

#include <cassert>

namespace cvcuda {

class GammaContrast final : public IOperator
{
public:
    explicit GammaContrast(const int32_t maxVarShapeBatchSize, const int32_t maxVarShapeChannelCount);

    void operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out,
                    const nvcv::Tensor &gamma) const;

    void operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out, float gamma, float gain,
                    NVCVRoundMode roundMode = NVCV_ROUND_NEAREST) const;

    void operator()(cudaStream_t stream, const nvcv::ImageBatch &in, const nvcv::ImageBatch &out,
                    const nvcv::Tensor &gamma) const;

    NVCVOperatorHandle handle() const noexcept override;

private:
    detail::OperatorHandle m_handle;
};

inline GammaContrast::GammaContrast(const int32_t maxVarShapeBatchSize, const int32_t maxVarShapeChannelCount)
{
    NVCVOperatorHandle h = nullptr;
    nvcv::detail::CheckThrow(cvcudaGammaContrastCreate(&h, maxVarShapeBatchSize, maxVarShapeChannelCount));
    assert(h);
    m_handle = detail::OperatorHandle{h};
}

inline void GammaContrast::operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out,
                                      const nvcv::Tensor &gamma) const
{
    nvcv::detail::CheckThrow(
        cvcudaGammaContrastSubmit(m_handle.get(), stream, in.handle(), out.handle(), gamma.handle()));
}

inline void GammaContrast::operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out, float gamma,
                                      float gain, NVCVRoundMode roundMode) const
{
    nvcv::detail::CheckThrow(
        cvcudaGammaContrastScalarSubmit(m_handle.get(), stream, in.handle(), out.handle(), gamma, gain, roundMode));
}

inline void GammaContrast::operator()(cudaStream_t stream, const nvcv::ImageBatch &in, const nvcv::ImageBatch &out,
                                      const nvcv::Tensor &gamma) const
{
    nvcv::detail::CheckThrow(
        cvcudaGammaContrastVarShapeSubmit(m_handle.get(), stream, in.handle(), out.handle(), gamma.handle()));
}

inline NVCVOperatorHandle GammaContrast::handle() const noexcept
{
    return m_handle.get();
}

} // namespace cvcuda

/** @} */

#endif // CVCUDA_GAMMA_CONTRAST_HPP
