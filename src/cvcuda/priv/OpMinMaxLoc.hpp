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
 * @file OpMinMaxLoc.hpp
 *
 * @brief Defines the private C++ Class for the MinMaxLoc operation.
 */

#ifndef CVCUDA_PRIV_MINMAXLOC_HPP
#define CVCUDA_PRIV_MINMAXLOC_HPP

#include "IOperator.hpp"

#include <cuda_runtime.h>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Optional.hpp>
#include <nvcv/Tensor.hpp>

namespace cvcuda::priv {

class MinMaxLoc final : public IOperator
{
public:
    explicit MinMaxLoc();

    void operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &minVal, const nvcv::Tensor &minLoc,
                    const nvcv::Tensor &numMin, const nvcv::Tensor &maxVal, const nvcv::Tensor &maxLoc,
                    const nvcv::Tensor &numMax) const;

    void operator()(cudaStream_t stream, const nvcv::ImageBatchVarShape &in, const nvcv::Tensor &minVal,
                    const nvcv::Tensor &minLoc, const nvcv::Tensor &numMin, const nvcv::Tensor &maxVal,
                    const nvcv::Tensor &maxLoc, const nvcv::Tensor &numMax) const;
};

} // namespace cvcuda::priv

#endif // CVCUDA_PRIV_MINMAXLOC_HPP
