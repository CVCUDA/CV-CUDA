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
 * @file OpColorTwist.hpp
 *
 * @brief Defines the public C++ Class for the ColorTwist operation.
 * @defgroup NVCV_CPP_ALGORITHM_COLOR_TWIST Color Twist
 * @{
 */

#ifndef CVCUDA__COLOR_TWIST_HPP
#define CVCUDA__COLOR_TWIST_HPP

#include "IOperator.hpp"
#include "OpColorTwist.h"

#include <cuda_runtime.h>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/ImageFormat.hpp>
#include <nvcv/Tensor.hpp>

namespace cvcuda {

class ColorTwist final : public IOperator
{
public:
    explicit ColorTwist();

    ~ColorTwist();

    void operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out, const nvcv::Tensor &twist);

    void operator()(cudaStream_t stream, const nvcv::ImageBatch &in, const nvcv::ImageBatch &out,
                    const nvcv::Tensor &twist);

    virtual NVCVOperatorHandle handle() const noexcept override;

private:
    NVCVOperatorHandle m_handle;
};

inline ColorTwist::ColorTwist()
{
    nvcv::detail::CheckThrow(cvcudaColorTwistCreate(&m_handle));
    assert(m_handle);
}

inline ColorTwist::~ColorTwist()
{
    nvcvOperatorDestroy(m_handle);
}

inline void ColorTwist::operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out,
                                   const nvcv::Tensor &twist)
{
    nvcv::detail::CheckThrow(cvcudaColorTwistSubmit(m_handle, stream, in.handle(), out.handle(), twist.handle()));
}

inline void ColorTwist::operator()(cudaStream_t stream, const nvcv::ImageBatch &in, const nvcv::ImageBatch &out,
                                   const nvcv::Tensor &twist)
{
    nvcv::detail::CheckThrow(
        cvcudaColorTwistVarShapeSubmit(m_handle, stream, in.handle(), out.handle(), twist.handle()));
}

inline NVCVOperatorHandle ColorTwist::handle() const noexcept
{
    return m_handle;
}

} // namespace cvcuda

#endif // CVCUDA__COLOR_TWIST_HPP
