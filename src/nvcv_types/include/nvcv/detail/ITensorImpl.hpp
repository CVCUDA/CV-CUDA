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

#ifndef NVCV_ITENSOR_IMPL_HPP
#define NVCV_ITENSOR_IMPL_HPP

#ifndef NVCV_ITENSOR_HPP
#    error "You must not include this header directly"
#endif

namespace nvcv {

// Implementation

inline NVCVTensorHandle ITensor::handle() const
{
    return doGetHandle();
}

inline TensorShape ITensor::shape() const
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

inline int ITensor::rank() const
{
    int32_t rank = 0;
    detail::CheckThrow(nvcvTensorGetShape(this->handle(), &rank, nullptr));
    return rank;
}

inline TensorLayout ITensor::layout() const
{
    NVCVTensorLayout layout;
    detail::CheckThrow(nvcvTensorGetLayout(this->handle(), &layout));
    return static_cast<TensorLayout>(layout);
}

inline DataType ITensor::dtype() const
{
    NVCVDataType out;
    detail::CheckThrow(nvcvTensorGetDataType(this->handle(), &out));
    return DataType{out};
}

inline const ITensorData *ITensor::exportData() const
{
    NVCVTensorData data;
    detail::CheckThrow(nvcvTensorExportData(this->handle(), &data));

    if (data.bufferType != NVCV_TENSOR_BUFFER_STRIDED_CUDA)
    {
        throw Exception(Status::ERROR_INVALID_OPERATION, "Tensor data cannot be exported, buffer type not supported");
    }

    m_cacheData.emplace(TensorShape(data.shape, data.rank, data.layout), DataType{data.dtype}, data.buffer.strided);

    return &*m_cacheData;
}

inline ITensor *ITensor::cast(HandleType h)
{
    return detail::CastImpl<ITensor>(&nvcvTensorGetUserPointer, &nvcvTensorSetUserPointer, h);
}

inline void ITensor::setUserPointer(void *ptr)
{
    detail::CheckThrow(nvcvTensorSetUserPointer(this->handle(), ptr));
}

inline void *ITensor::userPointer() const
{
    void *ptr;
    detail::CheckThrow(nvcvTensorGetUserPointer(this->handle(), &ptr));
    return ptr;
}

} // namespace nvcv

#endif // NVCV_ITENSOR_IMPL_HPP
