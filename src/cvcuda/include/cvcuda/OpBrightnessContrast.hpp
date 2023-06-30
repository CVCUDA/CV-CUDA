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
 * @file OpBrightnessContrast.hpp
 *
 * @brief Defines the public C++ Class for the BrightnessContrast operation.
 * @defgroup NVCV_CPP_ALGORITHM_BRIGHTNESS_CONTRAST Brightness Contrast
 * @{
 */

#ifndef CVCUDA__BRIGHTNESS_CONTRAST_HPP
#define CVCUDA__BRIGHTNESS_CONTRAST_HPP

#include "IOperator.hpp"
#include "OpBrightnessContrast.h"

#include <cuda_runtime.h>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/ImageFormat.hpp>
#include <nvcv/Tensor.hpp>

namespace cvcuda {

class BrightnessContrast final : public IOperator
{
public:
    explicit BrightnessContrast();

    ~BrightnessContrast();

    void operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out,
                    const nvcv::Tensor &brightness, const nvcv::Tensor &contrast, const nvcv::Tensor &brightnessShift,
                    const nvcv::Tensor &contrastCenter);

    void operator()(cudaStream_t stream, const nvcv::ImageBatch &in, const nvcv::ImageBatch &out,
                    const nvcv::Tensor &brightness, const nvcv::Tensor &contrast, const nvcv::Tensor &brightnessShift,
                    const nvcv::Tensor &contrastCenter);

    virtual NVCVOperatorHandle handle() const noexcept override;

private:
    NVCVOperatorHandle m_handle;
};

inline BrightnessContrast::BrightnessContrast()
{
    nvcv::detail::CheckThrow(cvcudaBrightnessContrastCreate(&m_handle));
    assert(m_handle);
}

inline BrightnessContrast::~BrightnessContrast()
{
    nvcvOperatorDestroy(m_handle);
}

inline void BrightnessContrast::operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out,
                                           const nvcv::Tensor &brightness, const nvcv::Tensor &contrast,
                                           const nvcv::Tensor &brightnessShift, const nvcv::Tensor &contrastCenter)
{
    nvcv::detail::CheckThrow(cvcudaBrightnessContrastSubmit(m_handle, stream, in.handle(), out.handle(),
                                                            brightness.handle(), contrast.handle(),
                                                            brightnessShift.handle(), contrastCenter.handle()));
}

inline void BrightnessContrast::operator()(cudaStream_t stream, const nvcv::ImageBatch &in, const nvcv::ImageBatch &out,
                                           const nvcv::Tensor &brightness, const nvcv::Tensor &contrast,
                                           const nvcv::Tensor &brightnessShift, const nvcv::Tensor &contrastCenter)
{
    nvcv::detail::CheckThrow(cvcudaBrightnessContrastVarShapeSubmit(m_handle, stream, in.handle(), out.handle(),
                                                                    brightness.handle(), contrast.handle(),
                                                                    brightnessShift.handle(), contrastCenter.handle()));
}

inline NVCVOperatorHandle BrightnessContrast::handle() const noexcept
{
    return m_handle;
}

} // namespace cvcuda

#endif // CVCUDA__BRIGHTNESS_CONTRAST_HPP
