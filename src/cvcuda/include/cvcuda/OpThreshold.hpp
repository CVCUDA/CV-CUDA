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
 * @file OpThreshold.hpp
 *
 * @brief Defines the public C++ Class for the threshold operation.
 * @defgroup NVCV_CPP_ALGORITHM_THRESHOLD Threshold
 * @{
 */

#ifndef CVCUDA_THRESHOLD_HPP
#define CVCUDA_THRESHOLD_HPP

#include "IOperator.hpp"
#include "OpThreshold.h"

#include <cuda_runtime.h>
#include <nvcv/IImageBatch.hpp>
#include <nvcv/ITensor.hpp>
#include <nvcv/ImageFormat.hpp>
#include <nvcv/alloc/Requirements.hpp>

namespace cvcuda {

class Threshold final : public IOperator
{
public:
    explicit Threshold(uint32_t type, int32_t maxBatchSize);

    ~Threshold();

    void operator()(cudaStream_t stream, nvcv::ITensor &in, nvcv::ITensor &out, nvcv::ITensor &thresh,
                    nvcv::ITensor &maxval);

    void operator()(cudaStream_t stream, nvcv::IImageBatchVarShape &in, nvcv::IImageBatchVarShape &out,
                    nvcv::ITensor &thresh, nvcv::ITensor &maxval);

    virtual NVCVOperatorHandle handle() const noexcept override;

private:
    NVCVOperatorHandle m_handle;
};

inline Threshold::Threshold(uint32_t type, int32_t maxBatchSize)
{
    nvcv::detail::CheckThrow(cvcudaThresholdCreate(&m_handle, type, maxBatchSize));
    assert(m_handle);
}

inline Threshold::~Threshold()
{
    nvcvOperatorDestroy(m_handle);
    m_handle = nullptr;
}

inline void Threshold::operator()(cudaStream_t stream, nvcv::ITensor &in, nvcv::ITensor &out, nvcv::ITensor &thresh,
                                  nvcv::ITensor &maxval)
{
    nvcv::detail::CheckThrow(
        cvcudaThresholdSubmit(m_handle, stream, in.handle(), out.handle(), thresh.handle(), maxval.handle()));
}

inline void Threshold::operator()(cudaStream_t stream, nvcv::IImageBatchVarShape &in, nvcv::IImageBatchVarShape &out,
                                  nvcv::ITensor &thresh, nvcv::ITensor &maxval)
{
    nvcv::detail::CheckThrow(
        cvcudaThresholdVarShapeSubmit(m_handle, stream, in.handle(), out.handle(), thresh.handle(), maxval.handle()));
}

inline NVCVOperatorHandle Threshold::handle() const noexcept
{
    return m_handle;
}

} // namespace cvcuda

#endif // CVCUDA_THRESHOLD_HPP
