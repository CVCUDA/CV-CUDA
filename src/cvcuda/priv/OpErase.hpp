/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * @file OpErase.hpp
 *
 * @brief Defines the private C++ Class for the erase operation.
 */

#ifndef CVCUDA_PRIV_ERASE_HPP
#define CVCUDA_PRIV_ERASE_HPP

#include "IOperator.hpp"
#include "PerDeviceResource.hpp"
#include "legacy/CvCudaLegacy.h"

#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>

namespace cvcuda::priv {

class Erase final : public IOperator
{
public:
    explicit Erase(int num_erasing_area);

    void operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out, const nvcv::Tensor &anchor,
                    const nvcv::Tensor &erasing, const nvcv::Tensor &values, const nvcv::Tensor &imgIdx, bool random,
                    unsigned int seed) const;

    void operator()(cudaStream_t stream, const nvcv::ImageBatchVarShape &in, const nvcv::ImageBatchVarShape &out,
                    const nvcv::Tensor &anchor, const nvcv::Tensor &erasing, const nvcv::Tensor &values,
                    const nvcv::Tensor &imgIdx, bool random, unsigned int seed) const;

    void operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out, int64_t i, int64_t j,
                    int64_t h, int64_t w, const nvcv::Tensor &values) const;

private:
    mutable PerDeviceResource<nvcv::legacy::cuda_op::Erase>         m_legacyOp;
    mutable PerDeviceResource<nvcv::legacy::cuda_op::EraseVarShape> m_legacyOpVarShape;
};

} // namespace cvcuda::priv

#endif // CVCUDA_PRIV_ERASE_HPP
