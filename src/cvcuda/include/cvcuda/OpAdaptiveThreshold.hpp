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
 * @file OpAdaptiveThreshold.hpp
 *
 * @brief Defines the public C++ class for the adaptive threshold operation.
 * @defgroup NVCV_CPP_ALGORITHM_ADAPTIVETHRESHOLD Adaptive threshold
 * @{
 */

#ifndef CVCUDA_ADAPTIVETHRESHOLD_HPP
#define CVCUDA_ADAPTIVETHRESHOLD_HPP

#include "IOperator.hpp"
#include "OpAdaptiveThreshold.h"

#include <cuda_runtime.h>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/ImageFormat.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/alloc/Requirements.hpp>

#include <cassert>

namespace cvcuda {

class AdaptiveThreshold final : public IOperator
{
public:
    explicit AdaptiveThreshold(int32_t maxBlockSize, int32_t maxVarShapeBatchSize);

    void operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out, double maxValue,
                    NVCVAdaptiveThresholdType adaptiveMethod, NVCVThresholdType thresholdType, int32_t blockSize,
                    double c) const;
    void operator()(cudaStream_t stream, const nvcv::ImageBatchVarShape &in, const nvcv::ImageBatchVarShape &out,
                    const nvcv::Tensor &maxValue, NVCVAdaptiveThresholdType adaptiveMethod,
                    NVCVThresholdType thresholdType, const nvcv::Tensor &blockSize, const nvcv::Tensor &c) const;

    NVCVOperatorHandle handle() const noexcept override;

private:
    detail::OperatorHandle m_handle;
};

inline AdaptiveThreshold::AdaptiveThreshold(int32_t maxBlockSize, int32_t maxVarShapeBatchSize)
{
    NVCVOperatorHandle h = nullptr;
    nvcv::detail::CheckThrow(cvcudaAdaptiveThresholdCreate(&h, maxBlockSize, maxVarShapeBatchSize));
    assert(h);
    m_handle = detail::OperatorHandle{h};
}

inline void AdaptiveThreshold::operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out,
                                          double maxValue, NVCVAdaptiveThresholdType adaptiveMethod,
                                          NVCVThresholdType thresholdType, int32_t blockSize, double c) const
{
    nvcv::detail::CheckThrow(cvcudaAdaptiveThresholdSubmit(m_handle.get(), stream, in.handle(), out.handle(), maxValue,
                                                           adaptiveMethod, thresholdType, blockSize, c));
}

inline void AdaptiveThreshold::operator()(cudaStream_t stream, const nvcv::ImageBatchVarShape &in,
                                          const nvcv::ImageBatchVarShape &out, const nvcv::Tensor &maxValue,
                                          NVCVAdaptiveThresholdType adaptiveMethod, NVCVThresholdType thresholdType,
                                          const nvcv::Tensor &blockSize, const nvcv::Tensor &c) const
{
    nvcv::detail::CheckThrow(cvcudaAdaptiveThresholdVarShapeSubmit(m_handle.get(), stream, in.handle(), out.handle(),
                                                                   maxValue.handle(), adaptiveMethod, thresholdType,
                                                                   blockSize.handle(), c.handle()));
}

inline NVCVOperatorHandle AdaptiveThreshold::handle() const noexcept
{
    return m_handle.get();
}

} // namespace cvcuda

/** @} */

#endif // CVCUDA_ADAPTIVETHRESHOLD_HPP
