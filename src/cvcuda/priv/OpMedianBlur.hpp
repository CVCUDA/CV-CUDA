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
 * @file OpMedianBlur.hpp
 *
 * @brief Defines the private C++ Class for the median blur operation.
 */

#ifndef CVCUDA_PRIV_MEDIAN_BLUR_HPP
#define CVCUDA_PRIV_MEDIAN_BLUR_HPP

#include "IOperator.hpp"
#include "legacy/CvCudaLegacy.h"

#include <nvcv/IImageBatch.hpp>
#include <nvcv/ITensor.hpp>

#include <memory>

namespace cvcuda::priv {

class MedianBlur final : public IOperator
{
public:
    explicit MedianBlur(const int maxVarShapeBatchSize);

    void operator()(cudaStream_t stream, const nvcv::ITensor &in, const nvcv::ITensor &out,
                    const nvcv::Size2D ksize) const;

    void operator()(cudaStream_t stream, const nvcv::IImageBatchVarShape &in, const nvcv::IImageBatchVarShape &out,
                    nvcv::ITensor &ksize) const;

private:
    std::unique_ptr<nvcv::legacy::cuda_op::MedianBlur>         m_legacyOp;
    std::unique_ptr<nvcv::legacy::cuda_op::MedianBlurVarShape> m_legacyOpVarShape;
};

} // namespace cvcuda::priv

#endif // CVCUDA_PRIV_MEDIAN_BLUR_HPP
