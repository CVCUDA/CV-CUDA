/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * @file OpLabel.hpp
 *
 * @brief Defines the private C++ Class for the Label operation.
 */

#ifndef CVCUDA_PRIV_LABEL_HPP
#define CVCUDA_PRIV_LABEL_HPP

#include "IOperator.hpp"

#include <cuda_runtime.h>
#include <cvcuda/OpLabel.h>
#include <nvcv/Tensor.hpp>

namespace cvcuda::priv {

class Label final : public IOperator
{
public:
    explicit Label();

    void operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out, const nvcv::Tensor &bgLabel,
                    const nvcv::Tensor &minThresh, const nvcv::Tensor &maxThresh, const nvcv::Tensor &minSize,
                    const nvcv::Tensor &count, const nvcv::Tensor &stats, const nvcv::Tensor &mask,
                    NVCVConnectivityType connectivity, NVCVLabelType assignLabels, NVCVLabelMaskType maskType) const;
};

} // namespace cvcuda::priv

#endif // CVCUDA_PRIV_LABEL_HPP
