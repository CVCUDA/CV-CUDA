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
 * @file OpAdvCvtColor.hpp
 *
 * @brief Defines the public C++ Class for the AdvCvtColor operation.
 * @defgroup NVCV_CPP_ALGORITHM__ADV_CVT_COLOR AdvCvtColor
 * @{
 */

#ifndef CVCUDA__ADV_CVT_COLOR_HPP
#define CVCUDA__ADV_CVT_COLOR_HPP

#include "IOperator.hpp"
#include "OpAdvCvtColor.h"

#include <cuda_runtime.h>
#include <nvcv/ImageFormat.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/alloc/Requirements.hpp>

#include <cassert>

namespace cvcuda {

class AdvCvtColor final : public IOperator
{
public:
    explicit AdvCvtColor();

    void operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out, NVCVColorConversionCode code,
                    nvcv::ColorSpec spec) const;

    NVCVOperatorHandle handle() const noexcept override;

private:
    detail::OperatorHandle m_handle;
};

inline AdvCvtColor::AdvCvtColor()
{
    NVCVOperatorHandle h = nullptr;
    nvcv::detail::CheckThrow(cvcudaAdvCvtColorCreate(&h));
    assert(h);
    m_handle = detail::OperatorHandle{h};
}

inline void AdvCvtColor::operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out,
                                    NVCVColorConversionCode code, nvcv::ColorSpec spec) const
{
    nvcv::detail::CheckThrow(cvcudaAdvCvtColorSubmit(m_handle.get(), stream, in.handle(), out.handle(), code,
                                                     static_cast<NVCVColorSpec>(spec)));
}

inline NVCVOperatorHandle AdvCvtColor::handle() const noexcept
{
    return m_handle.get();
}

} // namespace cvcuda

/** @} */

#endif // CVCUDA__ADV_CVT_COLOR_HPP
