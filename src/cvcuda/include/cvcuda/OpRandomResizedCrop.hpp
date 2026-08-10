/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * @file OpRandomResizedCrop.hpp
 *
 * @brief Defines the public C++ class for the random resized crop operation.
 * @defgroup NVCV_CPP_ALGORITHM_RANDOMRESIZEDCROP Random Resized Crop
 * @{
 */

#ifndef CVCUDA_RANDOMRESIZEDCROP_HPP
#define CVCUDA_RANDOMRESIZEDCROP_HPP

#include "IOperator.hpp"
#include "OpRandomResizedCrop.h"

#include <cuda_runtime.h>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/ImageFormat.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/alloc/Requirements.hpp>

#include <cassert>

namespace cvcuda {

class RandomResizedCrop final : public IOperator
{
public:
    explicit RandomResizedCrop(double minScale, double maxScale, double minRatio, double maxRatio, int32_t maxBatchSize,
                               uint32_t seed);

    void operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out,
                    const NVCVInterpolationType interpolation) const;
    void operator()(cudaStream_t stream, const nvcv::ImageBatchVarShape &in, const nvcv::ImageBatchVarShape &out,
                    const NVCVInterpolationType interpolation) const;

    NVCVOperatorHandle handle() const noexcept override;

private:
    detail::OperatorHandle m_handle;
};

inline RandomResizedCrop::RandomResizedCrop(double minScale, double maxScale, double minRatio, double maxRatio,
                                            int32_t maxBatchSize, uint32_t seed)
{
    NVCVOperatorHandle h = nullptr;
    nvcv::detail::CheckThrow(
        cvcudaRandomResizedCropCreate(&h, minScale, maxScale, minRatio, maxRatio, maxBatchSize, seed));
    assert(h);
    m_handle = detail::OperatorHandle{h};
}

inline void RandomResizedCrop::operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out,
                                          const NVCVInterpolationType interpolation) const
{
    nvcv::detail::CheckThrow(
        cvcudaRandomResizedCropSubmit(m_handle.get(), stream, in.handle(), out.handle(), interpolation));
}

inline void RandomResizedCrop::operator()(cudaStream_t stream, const nvcv::ImageBatchVarShape &in,
                                          const nvcv::ImageBatchVarShape &out,
                                          const NVCVInterpolationType     interpolation) const
{
    nvcv::detail::CheckThrow(
        cvcudaRandomResizedCropVarShapeSubmit(m_handle.get(), stream, in.handle(), out.handle(), interpolation));
}

inline NVCVOperatorHandle RandomResizedCrop::handle() const noexcept
{
    return m_handle.get();
}

} // namespace cvcuda

/** @} */

#endif // CVCUDA_RANDOMRESIZEDCROP_HPP
