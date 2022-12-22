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
 * @file OpCustomCrop.hpp
 *
 * @brief Defines the public C++ Class for the CustomCrop operation.
 * @defgroup NVCV_CPP_ALGORITHM_CUSTOM_CROP CustomCrop
 * @{
 */

#ifndef CVCUDA_CUSTOM_CROP_HPP
#define CVCUDA_CUSTOM_CROP_HPP

#include "IOperator.hpp"
#include "OpCustomCrop.h"

#include <cuda_runtime.h>
#include <nvcv/ITensor.hpp>
#include <nvcv/ImageFormat.hpp>
#include <nvcv/alloc/Requirements.hpp>

namespace cvcuda {

class CustomCrop final : public IOperator
{
public:
    explicit CustomCrop();

    ~CustomCrop();

    void operator()(cudaStream_t stream, nvcv::ITensor &in, nvcv::ITensor &out, const NVCVRectI cropRect);

    virtual NVCVOperatorHandle handle() const noexcept override;

private:
    NVCVOperatorHandle m_handle;
};

inline CustomCrop::CustomCrop()
{
    nvcv::detail::CheckThrow(cvcudaCustomCropCreate(&m_handle));
    assert(m_handle);
}

inline CustomCrop::~CustomCrop()
{
    nvcvOperatorDestroy(m_handle);
    m_handle = nullptr;
}

inline void CustomCrop::operator()(cudaStream_t stream, nvcv::ITensor &in, nvcv::ITensor &out, const NVCVRectI cropRect)
{
    nvcv::detail::CheckThrow(cvcudaCustomCropSubmit(m_handle, stream, in.handle(), out.handle(), cropRect));
}

inline NVCVOperatorHandle CustomCrop::handle() const noexcept
{
    return m_handle;
}

} // namespace cvcuda

#endif // CVCUDA_CUSTOM_CROP_HPP
