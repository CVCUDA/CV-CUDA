/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * @file OpHistogramEq.hpp
 *
 * @brief Defines the private C++ Class for the HistogramEq operation.
 */

#ifndef CVCUDA_PRIV__HISTOGRAM_EQ_HPP
#define CVCUDA_PRIV__HISTOGRAM_EQ_HPP

#include "IOperator.hpp"
#include "legacy/CvCudaLegacy.h"

#include <nvcv/Tensor.hpp>

#include <memory>

namespace cvcuda::priv {

class HistogramEq final : public IOperator
{
public:
    explicit HistogramEq(uint32_t maxBatchSize);

    void operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out) const;

    void operator()(cudaStream_t stream, const nvcv::ImageBatchVarShape &in, const nvcv::ImageBatchVarShape &out) const;

private:
    std::unique_ptr<nvcv::legacy::cuda_op::HistogramEq>         m_legacyOp;
    std::unique_ptr<nvcv::legacy::cuda_op::HistogramEqVarShape> m_legacyOpVarShape;
};

} // namespace cvcuda::priv

#endif // CVCUDA_PRIV__HISTOGRAM_EQ_HPP
