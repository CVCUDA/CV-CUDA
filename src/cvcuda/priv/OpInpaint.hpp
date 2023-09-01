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
 * @file OpInpaint.hpp
 *
 * @brief Defines the private C++ Class for the inpaint operation.
 */

#ifndef CVCUDA_PRIV_INPAINT_HPP
#define CVCUDA_PRIV_INPAINT_HPP

#include "IOperator.hpp"
#include "legacy/CvCudaLegacy.h"

#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>

#include <memory>

namespace cvcuda::priv {

class Inpaint final : public IOperator
{
public:
    explicit Inpaint(int maxBatchSize, nvcv::Size2D maxShape);

    void operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &masks, const nvcv::Tensor &out,
                    double inpaintRadius) const;
    void operator()(cudaStream_t stream, const nvcv::ImageBatchVarShape &in, const nvcv::ImageBatchVarShape &masks,
                    const nvcv::ImageBatchVarShape &out, double inpaintRadius) const;

private:
    std::unique_ptr<nvcv::legacy::cuda_op::Inpaint>         m_legacyOp;
    std::unique_ptr<nvcv::legacy::cuda_op::InpaintVarShape> m_legacyOpVarShape;
};

} // namespace cvcuda::priv

#endif // CVCUDA_PRIV_INPAINT_HPP
