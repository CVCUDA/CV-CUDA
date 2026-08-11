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
 * @file OpCvtColor.hpp
 *
 * @brief Defines the public C++ class for the CvtColor (convert color) operation.
 * @defgroup NVCV_CPP_ALGORITHM_CVTCOLOR CvtColor
 * @{
 */

#ifndef CVCUDA_CVTCOLOR_HPP
#define CVCUDA_CVTCOLOR_HPP

#include "IOperator.hpp"
#include "OpCvtColor.h"

#include <cuda_runtime.h>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/ImageFormat.hpp>
#include <nvcv/Size.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/alloc/Requirements.hpp>

#include <cassert>

namespace cvcuda {

class CvtColor final : public IOperator
{
public:
    explicit CvtColor();

    void operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out,
                    NVCVColorConversionCode code) const;

    void operator()(cudaStream_t stream, const nvcv::ImageBatch &in, const nvcv::ImageBatch &out,
                    NVCVColorConversionCode code) const;

    NVCVOperatorHandle handle() const noexcept override;

private:
    detail::OperatorHandle m_handle;
};

inline CvtColor::CvtColor()
{
    NVCVOperatorHandle h = nullptr;
    nvcv::detail::CheckThrow(cvcudaCvtColorCreate(&h));
    assert(h);
    m_handle = detail::OperatorHandle{h};
}

inline void CvtColor::operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out,
                                 NVCVColorConversionCode code) const
{
    nvcv::detail::CheckThrow(cvcudaCvtColorSubmit(m_handle.get(), stream, in.handle(), out.handle(), code));
}

inline void CvtColor::operator()(cudaStream_t stream, const nvcv::ImageBatch &in, const nvcv::ImageBatch &out,
                                 NVCVColorConversionCode code) const
{
    nvcv::detail::CheckThrow(cvcudaCvtColorVarShapeSubmit(m_handle.get(), stream, in.handle(), out.handle(), code));
}

inline NVCVOperatorHandle CvtColor::handle() const noexcept
{
    return m_handle.get();
}

} // namespace cvcuda

/** @} */

#endif // CVCUDA_CVTCOLOR_HPP
