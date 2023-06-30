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
 * @file OpRotate.hpp
 *
 * @brief Defines the private C++ Class for the rotate operation.
 */

#ifndef CVCUDA_PRIV_ROTATE_HPP
#define CVCUDA_PRIV_ROTATE_HPP

#include "IOperator.hpp"
#include "legacy/CvCudaLegacy.h"

#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>

#include <memory>

namespace cvcuda::priv {

class Rotate final : public IOperator
{
public:
    explicit Rotate(const int maxVarShapeBatchSize);

    void operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out, const double angleDeg,
                    const double2 shift, const NVCVInterpolationType interpolation) const;

    void operator()(cudaStream_t stream, const nvcv::ImageBatchVarShape &in, const nvcv::ImageBatchVarShape &out,
                    const nvcv::Tensor &angleDeg, const nvcv::Tensor &shift,
                    const NVCVInterpolationType interpolation) const;

private:
    std::unique_ptr<nvcv::legacy::cuda_op::Rotate>         m_legacyOp;
    std::unique_ptr<nvcv::legacy::cuda_op::RotateVarShape> m_legacyOpVarShape;
};

} // namespace cvcuda::priv

#endif // CVCUDA_PRIV_ROTATE_HPP
