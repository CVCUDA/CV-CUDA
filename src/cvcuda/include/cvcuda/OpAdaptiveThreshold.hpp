/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <nvcv/IImageBatch.hpp>
#include <nvcv/ITensor.hpp>
#include <nvcv/ImageFormat.hpp>
#include <nvcv/alloc/Requirements.hpp>

namespace cvcuda {

class AdaptiveThreshold final : public IOperator
{
public:
    explicit AdaptiveThreshold(int32_t maxBlockSize, int32_t maxVarShapeBatchSize);

    ~AdaptiveThreshold();

    void operator()(cudaStream_t stream, nvcv::ITensor &in, nvcv::ITensor &out, double maxValue,
                    NVCVAdaptiveThresholdType adaptiveMethod, NVCVThresholdType thresholdType, int32_t blockSize,
                    double c);
    void operator()(cudaStream_t stream, nvcv::IImageBatchVarShape &in, nvcv::IImageBatchVarShape &out,
                    nvcv::ITensor &maxValue, NVCVAdaptiveThresholdType adaptiveMethod, NVCVThresholdType thresholdType,
                    nvcv::ITensor &blockSize, nvcv::ITensor &c);

    virtual NVCVOperatorHandle handle() const noexcept override;

private:
    NVCVOperatorHandle m_handle;
};

inline AdaptiveThreshold::AdaptiveThreshold(int32_t maxBlockSize, int32_t maxVarShapeBatchSize)
{
    nvcv::detail::CheckThrow(cvcudaAdaptiveThresholdCreate(&m_handle, maxBlockSize, maxVarShapeBatchSize));
    assert(m_handle);
}

inline AdaptiveThreshold::~AdaptiveThreshold()
{
    nvcvOperatorDestroy(m_handle);
    m_handle = nullptr;
}

inline void AdaptiveThreshold::operator()(cudaStream_t stream, nvcv::ITensor &in, nvcv::ITensor &out, double maxValue,
                                          NVCVAdaptiveThresholdType adaptiveMethod, NVCVThresholdType thresholdType,
                                          int32_t blockSize, double c)
{
    nvcv::detail::CheckThrow(cvcudaAdaptiveThresholdSubmit(m_handle, stream, in.handle(), out.handle(), maxValue,
                                                           adaptiveMethod, thresholdType, blockSize, c));
}

inline void AdaptiveThreshold::operator()(cudaStream_t stream, nvcv::IImageBatchVarShape &in,
                                          nvcv::IImageBatchVarShape &out, nvcv::ITensor &maxValue,
                                          NVCVAdaptiveThresholdType adaptiveMethod, NVCVThresholdType thresholdType,
                                          nvcv::ITensor &blockSize, nvcv::ITensor &c)
{
    nvcv::detail::CheckThrow(cvcudaAdaptiveThresholdVarShapeSubmit(m_handle, stream, in.handle(), out.handle(),
                                                                   maxValue.handle(), adaptiveMethod, thresholdType,
                                                                   blockSize.handle(), c.handle()));
}

inline NVCVOperatorHandle AdaptiveThreshold::handle() const noexcept
{
    return m_handle;
}

} // namespace cvcuda

#endif // CVCUDA_ADAPTIVETHRESHOLD_HPP
