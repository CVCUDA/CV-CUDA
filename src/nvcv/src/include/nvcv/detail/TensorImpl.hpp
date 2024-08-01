/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

inline TensorShape Tensor::shape() const
{
    NVCVTensorHandle htensor = this->handle();

    int32_t rank = 0;
    detail::CheckThrow(nvcvTensorGetShape(htensor, &rank, nullptr));

    NVCVTensorLayout layout;
    detail::CheckThrow(nvcvTensorGetLayout(htensor, &layout));

    TensorShape::ShapeType shape(rank);
    detail::CheckThrow(nvcvTensorGetShape(htensor, &rank, shape.begin()));
    return {shape, layout};
}

inline int Tensor::rank() const
{
    int32_t rank = 0;
    detail::CheckThrow(nvcvTensorGetShape(this->handle(), &rank, nullptr));
    return rank;
}

inline TensorLayout Tensor::layout() const
{
    NVCVTensorLayout layout;
    detail::CheckThrow(nvcvTensorGetLayout(this->handle(), &layout));
    return static_cast<TensorLayout>(layout);
}

inline DataType Tensor::dtype() const
{
    NVCVDataType out;
    detail::CheckThrow(nvcvTensorGetDataType(this->handle(), &out));
    return DataType{out};
}

inline TensorData Tensor::exportData() const
{
    auto h = this->handle();
    if (h == nullptr)
        throw Exception(Status::ERROR_INVALID_OPERATION, "The tensor handle is null.");

    NVCVTensorData data;
    detail::CheckThrow(nvcvTensorExportData(this->handle(), &data));

    if (data.bufferType != NVCV_TENSOR_BUFFER_STRIDED_CUDA)
    {
        throw Exception(Status::ERROR_INVALID_OPERATION, "Tensor data cannot be exported, buffer type not supported");
    }

    return TensorData(data);
}

inline void Tensor::setUserPointer(void *ptr)
{
    detail::CheckThrow(nvcvTensorSetUserPointer(this->handle(), ptr));
}

inline void *Tensor::userPointer() const
{
    void *ptr;
    detail::CheckThrow(nvcvTensorGetUserPointer(this->handle(), &ptr));
    return ptr;
}

inline Tensor Tensor::reshape(const TensorShape &new_shape)
{
    NVCVTensorHandle out_handle;
    detail::CheckThrow(
        nvcvTensorReshape(this->handle(), new_shape.rank(), &new_shape.shape()[0], new_shape.layout(), &out_handle));
    Tensor out_tensor(std::move(out_handle));
    return out_tensor;
}

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
    NVCVTensorHandle handle;
    detail::CheckThrow(nvcvTensorConstruct(&reqs, alloc.handle(), &handle));
    reset(std::move(handle));
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

// Factory functions --------------------------------------------------

inline Tensor TensorWrapData(const TensorData &data, TensorDataCleanupCallback &&cleanup)
{
    NVCVTensorHandle handle;
    detail::CheckThrow(
        nvcvTensorWrapDataConstruct(&data.cdata(), cleanup.targetFunc(), cleanup.targetHandle(), &handle));
    cleanup.release(); // already owned by the tensor
    return Tensor(std::move(handle));
}

inline Tensor TensorWrapImage(const Image &img)
{
    NVCVTensorHandle handle;
    detail::CheckThrow(nvcvTensorWrapImageConstruct(img.handle(), &handle));
    return Tensor(std::move(handle));
}

} // namespace nvcv

#endif // NVCV_TENSOR_IMPL_HPP
