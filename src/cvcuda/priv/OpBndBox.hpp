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
 * @file OpBndBox.hpp
 *
 * @brief Defines the private C++ Class for the BndBox operation.
 */

#ifndef CVCUDA_PRIV__BND_BOX_HPP
#define CVCUDA_PRIV__BND_BOX_HPP

#include "IOperator.hpp"
#include "legacy/CvCudaLegacy.h"

#include <nvcv/Tensor.hpp>

#include <memory>

namespace cvcuda::priv {

class BndBox final : public IOperator
{
public:
    explicit BndBox();

    void operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out,
                    const NVCVBndBoxesI &bboxes) const;

private:
    std::unique_ptr<nvcv::legacy::cuda_op::BndBox> m_legacyOp;
};

} // namespace cvcuda::priv

#endif // CVCUDA_PRIV__BND_BOX_HPP
