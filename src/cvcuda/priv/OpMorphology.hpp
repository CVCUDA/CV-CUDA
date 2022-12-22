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
 * @file OpMorphology.hpp
 *
 * @brief Defines the private C++ Class for the Morphology operation.
 */

#ifndef CVCUDA_PRIV_MORPHOLOGY_HPP
#define CVCUDA_PRIV_MORPHOLOGY_HPP

#include "IOperator.hpp"
#include "legacy/CvCudaLegacy.h"

#include <nvcv/IImageBatch.hpp>
#include <nvcv/ITensor.hpp>

#include <memory>

namespace cvcuda::priv {

class Morphology final : public IOperator
{
public:
    explicit Morphology(const int32_t maxVarShapeBatchSize);

    void operator()(cudaStream_t stream, const nvcv::ITensor &in, const nvcv::ITensor &out,
                    NVCVMorphologyType morph_type, nvcv::Size2D mask_size, int2 anchor, int32_t iteration,
                    const NVCVBorderType borderMode) const;

    void operator()(cudaStream_t stream, const nvcv::IImageBatchVarShape &in, const nvcv::IImageBatchVarShape &out,
                    NVCVMorphologyType morph_type, nvcv::ITensor &masks, nvcv::ITensor &anchors, int32_t iteration,
                    NVCVBorderType borderMode) const;

private:
    std::unique_ptr<nvcv::legacy::cuda_op::Morphology>         m_legacyOp;
    std::unique_ptr<nvcv::legacy::cuda_op::MorphologyVarShape> m_legacyOpVarShape;
};

} // namespace cvcuda::priv

#endif // CVCUDA_PRIV_MORPHOLOGY_HPP
