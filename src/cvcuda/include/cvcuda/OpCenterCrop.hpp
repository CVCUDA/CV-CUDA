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
 * @file OpCenterCrop.hpp
 *
 * @brief Defines the public C++ Class for the CenterCrop operation.
 * @defgroup NVCV_CPP_ALGORITHM_CENTER_CROP CenterCrop
 * @{
 */

#ifndef CVCUDA_CENTER_CROP_HPP
#define CVCUDA_CENTER_CROP_HPP

#include "IOperator.hpp"
#include "OpCenterCrop.h"

#include <cuda_runtime.h>
#include <nvcv/ImageFormat.hpp>
#include <nvcv/Size.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/alloc/Requirements.hpp>

#include <cassert>

namespace cvcuda {

class CenterCrop final : public IOperator
{
public:
    explicit CenterCrop();

    void operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out,
                    const nvcv::Size2D &cropSize) const;

    NVCVOperatorHandle handle() const noexcept override;

private:
    detail::OperatorHandle m_handle;
};

inline CenterCrop::CenterCrop()
{
    NVCVOperatorHandle h = nullptr;
    nvcv::detail::CheckThrow(cvcudaCenterCropCreate(&h));
    assert(h);
    m_handle = detail::OperatorHandle{h};
}

inline void CenterCrop::operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out,
                                   const nvcv::Size2D &cropSize) const
{
    nvcv::detail::CheckThrow(
        cvcudaCenterCropSubmit(m_handle.get(), stream, in.handle(), out.handle(), cropSize.w, cropSize.h));
}

inline NVCVOperatorHandle CenterCrop::handle() const noexcept
{
    return m_handle.get();
}

} // namespace cvcuda

/** @} */

#endif // CVCUDA_CENTER_CROP_HPP
