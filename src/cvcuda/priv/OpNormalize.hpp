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
 * @file OpNormalize.hpp
 *
 * @brief Defines the private C++ Class for the normalize operation.
 */

#ifndef CVCUDA_PRIV_NORMALIZE_HPP
#define CVCUDA_PRIV_NORMALIZE_HPP

#include "IOperator.hpp"
#include "legacy/CvCudaLegacy.h"

#include <nvcv/IImageBatch.hpp>
#include <nvcv/ITensor.hpp>

#include <memory>

namespace cvcuda::priv {

class Normalize final : public IOperator
{
public:
    explicit Normalize();

    void operator()(cudaStream_t stream, const nvcv::ITensor &in, const nvcv::ITensor &base, const nvcv::ITensor &scale,
                    nvcv::ITensor &out, float global_scale, float shift, float epsilon, uint32_t flags) const;

    void operator()(cudaStream_t stream, const nvcv::IImageBatchVarShape &in, const nvcv::ITensor &base,
                    const nvcv::ITensor &scale, nvcv::IImageBatchVarShape &out, float global_scale, float shift,
                    float epsilon, uint32_t flags) const;

private:
    std::unique_ptr<nvcv::legacy::cuda_op::Normalize>         m_legacyOp;
    std::unique_ptr<nvcv::legacy::cuda_op::NormalizeVarShape> m_legacyOpVarShape;
};

} // namespace cvcuda::priv

#endif // CVCUDA_PRIV_NORMALIZE_HPP
