/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * @file OpAdjustHue.hpp
 *
 * @brief Defines the public C++ Class for the AdjustHue operation.
 * @defgroup NVCV_CPP_ALGORITHM__ADJUST_HUE Adjust Hue
 * @{
 */

#ifndef CVCUDA__ADJUST_HUE_HPP
#define CVCUDA__ADJUST_HUE_HPP

#include "IOperator.hpp"
#include "OpAdjustHue.h"

#include <cuda_runtime.h>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/ImageFormat.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/alloc/Requirements.hpp>

namespace cvcuda {

class AdjustHue final : public IOperator
{
public:
    explicit AdjustHue();

    void operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out, double hue) const;

    void operator()(cudaStream_t stream, const nvcv::ImageBatch &in, const nvcv::ImageBatch &out, double hue) const;

    NVCVOperatorHandle handle() const noexcept override;

private:
    detail::OperatorHandle m_handle;
};

inline AdjustHue::AdjustHue()
{
    NVCVOperatorHandle h = nullptr;
    nvcv::detail::CheckThrow(cvcudaAdjustHueCreate(&h));
    assert(h);
    m_handle = detail::OperatorHandle{h};
}

inline void AdjustHue::operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out,
                                  double hue) const
{
    nvcv::detail::CheckThrow(cvcudaAdjustHueSubmit(m_handle.get(), stream, in.handle(), out.handle(), hue));
}

inline void AdjustHue::operator()(cudaStream_t stream, const nvcv::ImageBatch &in, const nvcv::ImageBatch &out,
                                  double hue) const
{
    nvcv::detail::CheckThrow(cvcudaAdjustHueVarShapeSubmit(m_handle.get(), stream, in.handle(), out.handle(), hue));
}

inline NVCVOperatorHandle AdjustHue::handle() const noexcept
{
    return m_handle.get();
}

} // namespace cvcuda

#endif // CVCUDA__ADJUST_HUE_HPP
