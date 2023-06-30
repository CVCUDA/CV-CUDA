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
 * @file OpCropFlipNormalizeReformat.hpp
 *
 * @brief Defines the private C++ Class for the CropFlipNormalizeReformat operation.
 */

#ifndef CVCUDA_PRIV_CROP_FLIP_NORMALIZE_REFORMAT_HPP
#define CVCUDA_PRIV_CROP_FLIP_NORMALIZE_REFORMAT_HPP

#include "IOperator.hpp"

#include <cuda_runtime.h>
#include <cvcuda/OpCropFlipNormalizeReformat.h>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>

#include <memory>

namespace cvcuda::priv {

class CropFlipNormalizeReformat final : public IOperator
{
public:
    explicit CropFlipNormalizeReformat();

    void operator()(cudaStream_t stream, const nvcv::ImageBatchVarShape &in, const nvcv::Tensor &out,
                    const nvcv::Tensor &cropRect, const NVCVBorderType borderMode, const float borderValue,
                    const nvcv::Tensor &flipCode, const nvcv::Tensor &base, const nvcv::Tensor &scale,
                    float global_scale, float shift, float epsilon, uint32_t flags) const;

private:
};

} // namespace cvcuda::priv

#endif // CVCUDA_PRIV_CROP_FLIP_NORMALIZE_REFORMAT_HPP
