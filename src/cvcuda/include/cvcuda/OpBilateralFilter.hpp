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
 * @file OpBilateralFilter.hpp
 *
 * @brief Defines the public C++ Class for the BilateralFilter operation.
 * @defgroup NVCV_CPP_ALGORITHM_BILATERAL_FILTER BilateralFilter
 * @{
 */

#ifndef CVCUDA_BILATERAL_FILTER_HPP
#define CVCUDA_BILATERAL_FILTER_HPP

#include "IOperator.hpp"
#include "OpBilateralFilter.h"

#include <cuda_runtime.h>
#include <nvcv/IImageBatch.hpp>
#include <nvcv/ITensor.hpp>
#include <nvcv/ImageFormat.hpp>
#include <nvcv/Size.hpp>
#include <nvcv/alloc/Requirements.hpp>

namespace cvcuda {

class BilateralFilter final : public IOperator
{
public:
    explicit BilateralFilter();

    ~BilateralFilter();

    void operator()(cudaStream_t stream, nvcv::ITensor &in, nvcv::ITensor &out, int diameter, float sigmaColor,
                    float sigmaSpace, NVCVBorderType borderMode);

    void operator()(cudaStream_t stream, nvcv::IImageBatch &in, nvcv::IImageBatch &out, nvcv::ITensor &diameterData,
                    nvcv::ITensor &sigmaColorData, nvcv::ITensor &sigmaSpace, NVCVBorderType borderMode);

    virtual NVCVOperatorHandle handle() const noexcept override;

private:
    NVCVOperatorHandle m_handle;
};

inline BilateralFilter::BilateralFilter()
{
    nvcv::detail::CheckThrow(cvcudaBilateralFilterCreate(&m_handle));
    assert(m_handle);
}

inline BilateralFilter::~BilateralFilter()
{
    nvcvOperatorDestroy(m_handle);
    m_handle = nullptr;
}

inline void BilateralFilter::operator()(cudaStream_t stream, nvcv::ITensor &in, nvcv::ITensor &out, int diameter,
                                        float sigmaColor, float sigmaSpace, NVCVBorderType borderMode)
{
    nvcv::detail::CheckThrow(cvcudaBilateralFilterSubmit(m_handle, stream, in.handle(), out.handle(), diameter,
                                                         sigmaColor, sigmaSpace, borderMode));
}

inline void BilateralFilter::operator()(cudaStream_t stream, nvcv::IImageBatch &in, nvcv::IImageBatch &out,
                                        nvcv::ITensor &diameterData, nvcv::ITensor &sigmaColorData,
                                        nvcv::ITensor &sigmaSpaceData, NVCVBorderType borderMode)
{
    nvcv::detail::CheckThrow(cvcudaBilateralFilterVarShapeSubmit(m_handle, stream, in.handle(), out.handle(),
                                                                 diameterData.handle(), sigmaColorData.handle(),
                                                                 sigmaSpaceData.handle(), borderMode));
}

inline NVCVOperatorHandle BilateralFilter::handle() const noexcept
{
    return m_handle;
}

} // namespace cvcuda

#endif // CVCUDA_BILATERAL_FILTER_HPP
