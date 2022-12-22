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
 * @file OpPillowResize.hpp
 *
 * @brief Defines the public C++ Class for the pillow resize operation.
 * @defgroup NVCV_CPP_ALGORITHM_PILLOW_RESIZE Pillow Resize
 * @{
 */

#ifndef CVCUDA_PILLOW_RESIZE_HPP
#define CVCUDA_PILLOW_RESIZE_HPP

#include "IOperator.hpp"
#include "OpPillowResize.h"

#include <cuda_runtime.h>
#include <nvcv/IImageBatch.hpp>
#include <nvcv/ITensor.hpp>
#include <nvcv/ImageFormat.hpp>
#include <nvcv/Size.hpp>
#include <nvcv/alloc/Requirements.hpp>

namespace cvcuda {

class PillowResize final : public IOperator
{
public:
    explicit PillowResize(nvcv::Size2D maxSize, int32_t maxBatchSize, nvcv::ImageFormat fmt);

    ~PillowResize();

    void operator()(cudaStream_t stream, nvcv::ITensor &in, nvcv::ITensor &out,
                    const NVCVInterpolationType interpolation);

    void operator()(cudaStream_t stream, nvcv::IImageBatchVarShape &in, nvcv::IImageBatchVarShape &out,
                    const NVCVInterpolationType interpolation);

    virtual NVCVOperatorHandle handle() const noexcept override;

private:
    NVCVOperatorHandle m_handle;
};

inline PillowResize::PillowResize(nvcv::Size2D maxSize, int32_t maxBatchSize, nvcv::ImageFormat fmt)
{
    NVCVImageFormat cfmt = fmt.cvalue();
    nvcv::detail::CheckThrow(cvcudaPillowResizeCreate(&m_handle, maxSize.w, maxSize.h, maxBatchSize, cfmt));
    assert(m_handle);
}

inline PillowResize::~PillowResize()
{
    nvcvOperatorDestroy(m_handle);
    m_handle = nullptr;
}

inline void PillowResize::operator()(cudaStream_t stream, nvcv::ITensor &in, nvcv::ITensor &out,
                                     const NVCVInterpolationType interpolation)
{
    nvcv::detail::CheckThrow(cvcudaPillowResizeSubmit(m_handle, stream, in.handle(), out.handle(), interpolation));
}

inline void PillowResize::operator()(cudaStream_t stream, nvcv::IImageBatchVarShape &in, nvcv::IImageBatchVarShape &out,
                                     const NVCVInterpolationType interpolation)
{
    nvcv::detail::CheckThrow(
        nvcvopPillowResizeVarShapeSubmit(m_handle, stream, in.handle(), out.handle(), interpolation));
}

inline NVCVOperatorHandle PillowResize::handle() const noexcept
{
    return m_handle;
}

} // namespace cvcuda

#endif // CVCUDA_PILLOW_RESIZE_HPP
