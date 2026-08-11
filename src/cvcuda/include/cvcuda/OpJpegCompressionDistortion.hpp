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
 * @file OpJpegCompressionDistortion.hpp
 *
 * @brief Defines the public C++ Class for the JpegCompressionDistortion operation.
 * @defgroup NVCV_CPP_ALGORITHM__JPEG_COMPRESSION_DISTORTION JpegCompressionDistortion
 * @{
 */

#ifndef CVCUDA__JPEG_COMPRESSION_DISTORTION_HPP
#define CVCUDA__JPEG_COMPRESSION_DISTORTION_HPP

#include "IOperator.hpp"
#include "OpJpegCompressionDistortion.h"

#include <cuda_runtime.h>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/ImageFormat.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/alloc/Requirements.hpp>

namespace cvcuda {

class JpegCompressionDistortion final : public IOperator
{
public:
    explicit JpegCompressionDistortion();

    void operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out,
                    const nvcv::Tensor &quality) const;

    void operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out, int32_t quality) const;

    void operator()(cudaStream_t stream, const nvcv::ImageBatchVarShape &in, const nvcv::ImageBatchVarShape &out,
                    const nvcv::Tensor &quality) const;

    void operator()(cudaStream_t stream, const nvcv::ImageBatchVarShape &in, const nvcv::ImageBatchVarShape &out,
                    int32_t quality) const;

    NVCVOperatorHandle handle() const noexcept override;

private:
    detail::OperatorHandle m_handle;
};

inline JpegCompressionDistortion::JpegCompressionDistortion()
{
    NVCVOperatorHandle h = nullptr;
    nvcv::detail::CheckThrow(cvcudaJpegCompressionDistortionCreate(&h));
    assert(h);
    m_handle = detail::OperatorHandle{h};
}

inline void JpegCompressionDistortion::operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out,
                                                  const nvcv::Tensor &quality) const
{
    nvcv::detail::CheckThrow(
        cvcudaJpegCompressionDistortionSubmit(m_handle.get(), stream, in.handle(), out.handle(), quality.handle()));
}

inline void JpegCompressionDistortion::operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out,
                                                  int32_t quality) const
{
    nvcv::detail::CheckThrow(
        cvcudaJpegCompressionDistortionScalarSubmit(m_handle.get(), stream, in.handle(), out.handle(), quality));
}

inline void JpegCompressionDistortion::operator()(cudaStream_t stream, const nvcv::ImageBatchVarShape &in,
                                                  const nvcv::ImageBatchVarShape &out,
                                                  const nvcv::Tensor             &quality) const
{
    nvcv::detail::CheckThrow(cvcudaJpegCompressionDistortionVarShapeSubmit(m_handle.get(), stream, in.handle(),
                                                                           out.handle(), quality.handle()));
}

inline void JpegCompressionDistortion::operator()(cudaStream_t stream, const nvcv::ImageBatchVarShape &in,
                                                  const nvcv::ImageBatchVarShape &out, int32_t quality) const
{
    nvcv::detail::CheckThrow(cvcudaJpegCompressionDistortionVarShapeScalarSubmit(m_handle.get(), stream, in.handle(),
                                                                                 out.handle(), quality));
}

inline NVCVOperatorHandle JpegCompressionDistortion::handle() const noexcept
{
    return m_handle.get();
}

} // namespace cvcuda

#endif // CVCUDA__JPEG_COMPRESSION_DISTORTION_HPP
