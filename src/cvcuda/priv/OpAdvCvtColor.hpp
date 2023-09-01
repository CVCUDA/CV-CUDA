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
 * @file OpAdvCvtColor.hpp
 *
 * @brief Defines the private C++ Class for the AdvCvtColor operation.
 */

#ifndef CVCUDA_PRIV__ADV_CVT_COLOR_HPP
#define CVCUDA_PRIV__ADV_CVT_COLOR_HPP

#include "IOperator.hpp"
#include "legacy/CvCudaLegacy.h"

#include <nvcv/ColorSpec.hpp>
#include <nvcv/Tensor.hpp>

#include <memory>

namespace cvcuda::priv {

class AdvCvtColor final : public IOperator
{
public:
    explicit AdvCvtColor();

    void operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out, NVCVColorConversionCode code,
                    nvcv::ColorSpec spec) const;

private:
    void Yuv2Bgr(cudaStream_t stream, const nvcv::TensorDataStridedCuda &in, const nvcv::TensorDataStridedCuda &out,
                 NVCVColorConversionCode code, nvcv::ColorSpec spec) const;
    void Bgr2Yuv(cudaStream_t stream, const nvcv::TensorDataStridedCuda &in, const nvcv::TensorDataStridedCuda &out,
                 NVCVColorConversionCode code, nvcv::ColorSpec spec) const;
    void NvYuv2Bgr(cudaStream_t stream, const nvcv::TensorDataStridedCuda &in, const nvcv::TensorDataStridedCuda &out,
                   NVCVColorConversionCode code, nvcv::ColorSpec spec) const;
    void Bgr2NvYuv(cudaStream_t stream, const nvcv::TensorDataStridedCuda &in, const nvcv::TensorDataStridedCuda &out,
                   NVCVColorConversionCode code, nvcv::ColorSpec spec) const;
};

} // namespace cvcuda::priv

#endif // CVCUDA_PRIV__ADV_CVT_COLOR_HPP
