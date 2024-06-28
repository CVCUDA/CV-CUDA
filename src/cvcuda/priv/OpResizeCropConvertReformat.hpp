/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * @file OpResizeCropConvertReformat.hpp
 *
 * @brief Defines the private C++ class for that fuses resize, crop, data type conversion, channel manipulation, and layout reformat operations to optimize pipelines.
 */

#ifndef CVCUDA_PRIV__RESIZE_CROP_HPP
#define CVCUDA_PRIV__RESIZE_CROP_HPP

#include "IOperator.hpp"
#include "legacy/CvCudaLegacy.h"

#include <cvcuda/Types.h> // for NVCVInterpolationType, NVCVChannelManip, etc.
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>

#include <memory>

namespace cvcuda::priv {

class ResizeCropConvertReformat final : public IOperator
{
public:
    explicit ResizeCropConvertReformat();

    void operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out, const NVCVSize2D resizeDim,
                    const NVCVInterpolationType interpolation, const int2 cropPos,
                    const NVCVChannelManip manip = NVCV_CHANNEL_NO_OP, const float scale = 1, const float offset = 0,
                    const bool srcCast = true) const;

    void operator()(cudaStream_t stream, const nvcv::ImageBatchVarShape &in, const nvcv::Tensor &out,
                    const NVCVSize2D resizeDim, const NVCVInterpolationType interpolation, const int2 cropPos,
                    const NVCVChannelManip manip = NVCV_CHANNEL_NO_OP, const float scale = 1, const float offset = 0,
                    const bool srcCast = true) const;
};

} // namespace cvcuda::priv

#endif // CVCUDA_PRIV__RESIZE_CROP_HPP
