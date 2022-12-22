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
#include <nvcv/IImageBatch.hpp>
#include <nvcv/ITensor.hpp>
#include <nvcv/ImageFormat.hpp>
#include <nvcv/Size.hpp>
#include <nvcv/alloc/Requirements.hpp>

namespace cvcuda {

class CvtColor final : public IOperator
{
public:
    explicit CvtColor();

    ~CvtColor();

    void operator()(cudaStream_t stream, nvcv::ITensor &in, nvcv::ITensor &out, NVCVColorConversionCode code);

    void operator()(cudaStream_t stream, nvcv::IImageBatch &in, nvcv::IImageBatch &out, NVCVColorConversionCode code);

    virtual NVCVOperatorHandle handle() const noexcept override;

private:
    NVCVOperatorHandle m_handle;
};

inline CvtColor::CvtColor()
{
    nvcv::detail::CheckThrow(cvcudaCvtColorCreate(&m_handle));
    assert(m_handle);
}

inline CvtColor::~CvtColor()
{
    nvcvOperatorDestroy(m_handle);
    m_handle = nullptr;
}

inline void CvtColor::operator()(cudaStream_t stream, nvcv::ITensor &in, nvcv::ITensor &out,
                                 NVCVColorConversionCode code)
{
    nvcv::detail::CheckThrow(cvcudaCvtColorSubmit(m_handle, stream, in.handle(), out.handle(), code));
}

inline void CvtColor::operator()(cudaStream_t stream, nvcv::IImageBatch &in, nvcv::IImageBatch &out,
                                 NVCVColorConversionCode code)
{
    nvcv::detail::CheckThrow(cvcudaCvtColorVarShapeSubmit(m_handle, stream, in.handle(), out.handle(), code));
}

inline NVCVOperatorHandle CvtColor::handle() const noexcept
{
    return m_handle;
}

} // namespace cvcuda

#endif // CVCUDA_CVTCOLOR_HPP
