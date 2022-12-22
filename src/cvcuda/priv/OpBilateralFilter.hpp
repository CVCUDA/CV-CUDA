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
 * @file OpBilateralFilter.hpp
 *
 * @brief Defines the private C++ Class for the Bilateral Filter operation.
 */

#ifndef CVCUDA_PRIV_BILATERAL_FILTER_HPP
#define CVCUDA_PRIV_BILATERAL_FILTER_HPP

#include "IOperator.hpp"
#include "legacy/CvCudaLegacy.h"

#include <nvcv/IImageBatch.hpp>
#include <nvcv/ITensor.hpp>

#include <memory>

namespace cvcuda::priv {

class BilateralFilter final : public IOperator
{
public:
    explicit BilateralFilter();

    void operator()(cudaStream_t stream, const nvcv::ITensor &in, const nvcv::ITensor &out, int diameter,
                    float sigmaColor, float sigmaSpace, NVCVBorderType borderMode) const;

    void operator()(cudaStream_t stream, const nvcv::IImageBatchVarShape &in, const nvcv::IImageBatchVarShape &out,
                    const nvcv::ITensor &diameter, const nvcv::ITensor &sigmaColor, const nvcv::ITensor &sigmaSpace,
                    NVCVBorderType borderMode) const;

private:
    std::unique_ptr<nvcv::legacy::cuda_op::BilateralFilter>         m_legacyOp;
    std::unique_ptr<nvcv::legacy::cuda_op::BilateralFilterVarShape> m_legacyOpVarShape;
};

} // namespace cvcuda::priv

#endif // CVCUDA_PRIV_BILATERAL_FILTER_HPP
