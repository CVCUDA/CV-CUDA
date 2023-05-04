/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef NVCV_TENSOR_IMPL_HPP
#define NVCV_TENSOR_IMPL_HPP

#ifndef NVCV_TENSOR_HPP
#    error "You must not include this header directly"
#endif

namespace nvcv {

// Tensor implementation -------------------------------------

inline auto Tensor::CalcRequirements(const TensorShape &shape, DataType dtype, const MemAlignment &bufAlign)
    -> Requirements
{
    Requirements reqs;
    detail::CheckThrow(nvcvTensorCalcRequirements(shape.size(), &shape[0], dtype,
                                                  static_cast<NVCVTensorLayout>(shape.layout()), bufAlign.baseAddr(),
                                                  bufAlign.rowAddr(), &reqs));
    return reqs;
}

inline auto Tensor::CalcRequirements(int numImages, Size2D imgSize, ImageFormat fmt, const MemAlignment &bufAlign)
    -> Requirements
{
    Requirements reqs;
    detail::CheckThrow(nvcvTensorCalcRequirementsForImages(numImages, imgSize.w, imgSize.h, fmt, bufAlign.baseAddr(),
                                                           bufAlign.rowAddr(), &reqs));
    return reqs;
}

inline Tensor::Tensor(const Requirements &reqs, const Allocator &alloc)
{
    detail::CheckThrow(nvcvTensorConstruct(&reqs, alloc.handle(), &m_handle));
    detail::SetObjectAssociation(nvcvTensorSetUserPointer, this, m_handle);
}

inline Tensor::Tensor(int numImages, Size2D imgSize, ImageFormat fmt, const MemAlignment &bufAlign,
                      const Allocator &alloc)
    : Tensor(CalcRequirements(numImages, imgSize, fmt, bufAlign), alloc)
{
}

inline Tensor::Tensor(const TensorShape &shape, DataType dtype, const MemAlignment &bufAlign, const Allocator &alloc)
    : Tensor(CalcRequirements(shape, dtype, bufAlign), alloc)
{
}

inline NVCVTensorHandle Tensor::doGetHandle() const
{
    return m_handle;
}

inline Tensor::~Tensor()
{
    nvcvTensorDecRef(m_handle, nullptr);
}

// TensorWrapData implementation -------------------------------------

inline TensorWrapData::TensorWrapData(const TensorData &data, TensorDataCleanupCallback &&cleanup)
{
    detail::CheckThrow(
        nvcvTensorWrapDataConstruct(&data.cdata(), cleanup.targetFunc(), cleanup.targetHandle(), &m_handle));
    cleanup.release(); // already owned by the handle
    try
    {
        detail::SetObjectAssociation(nvcvTensorSetUserPointer, this, m_handle);
    }
    catch (...)
    {
        nvcvTensorDecRef(m_handle, nullptr);
        throw;
    }
}

inline TensorWrapData::~TensorWrapData()
{
    nvcvTensorDecRef(m_handle, nullptr);
}

inline NVCVTensorHandle TensorWrapData::doGetHandle() const
{
    return m_handle;
}

// TensorWrapImage implementation -------------------------------------

inline TensorWrapImage::TensorWrapImage(const IImage &img)
{
    detail::CheckThrow(nvcvTensorWrapImageConstruct(img.handle(), &m_handle));
    detail::SetObjectAssociation(nvcvTensorSetUserPointer, this, m_handle);
}

inline TensorWrapImage::~TensorWrapImage()
{
    nvcvTensorDecRef(m_handle, nullptr);
}

inline NVCVTensorHandle TensorWrapImage::doGetHandle() const
{
    return m_handle;
}

} // namespace nvcv

#endif // NVCV_TENSOR_IMPL_HPP
