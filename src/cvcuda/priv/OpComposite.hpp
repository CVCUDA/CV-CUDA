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
 * @file OpComposite.hpp
 *
 * @brief Defines the private C++ Class for the composite operation.
 */

#ifndef CVCUDA_PRIV_COMPOSITE_HPP
#define CVCUDA_PRIV_COMPOSITE_HPP

#include "IOperator.hpp"
#include "legacy/CvCudaLegacy.h"

#include <cuda_runtime.h>
#include <nvcv/IImageBatch.hpp>
#include <nvcv/ITensor.hpp>

#include <memory>

namespace cvcuda::priv {

class Composite final : public IOperator
{
public:
    explicit Composite();

    void operator()(cudaStream_t stream, const nvcv::ITensor &foreground, const nvcv::ITensor &background,
                    const nvcv::ITensor &fgMask, const nvcv::ITensor &output) const;

    void operator()(cudaStream_t stream, const nvcv::IImageBatchVarShape &foreground,
                    const nvcv::IImageBatchVarShape &background, const nvcv::IImageBatchVarShape &fgMask,
                    const nvcv::IImageBatchVarShape &output) const;

private:
    std::unique_ptr<nvcv::legacy::cuda_op::Composite>         m_legacyOp;
    std::unique_ptr<nvcv::legacy::cuda_op::CompositeVarShape> m_legacyOpVarShape;
};

} // namespace cvcuda::priv

#endif // CVCUDA_PRIV_COMPOSITE_HPP
