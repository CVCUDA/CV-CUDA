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

#ifndef NVCV_TENSORDATA_IMPL_HPP
#define NVCV_TENSORDATA_IMPL_HPP

#ifndef NVCV_TENSORDATA_HPP
#    error "You must not include this header directly"
#endif

#include <algorithm>

namespace nvcv {

// Implementation - TensorData -----------------------------

inline TensorData::TensorData(const NVCVTensorData &data)
    : m_data(data)
{
}

inline int TensorData::rank() const
{
    return this->cdata().rank;
}

inline const TensorShape &TensorData::shape() const &
{
    if (!m_cacheShape)
    {
        const NVCVTensorData &data = this->cdata();
        m_cacheShape.emplace(data.shape, data.rank, data.layout);
    }

    return *m_cacheShape;
}

inline const TensorShape::DimType &TensorData::shape(int d) const &
{
    const NVCVTensorData &data = this->cdata();

    if (d < 0 || d >= data.rank)
    {
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "Index of shape dimension %d is out of bounds [0;%d]", d,
                        data.rank - 1);
    }
    return data.shape[d];
}

inline const TensorLayout &TensorData::layout() const &
{
    return this->shape().layout();
}

inline DataType TensorData::dtype() const
{
    const NVCVTensorData &data = this->cdata();
    return DataType{data.dtype};
}

inline const NVCVTensorData &TensorData::cdata() const &
{
    return m_data;
}

inline NVCVTensorData &TensorData::data() &
{
    // data contents might be modified, must reset cache
    m_cacheShape.reset();
    return m_data;
}

template<typename Derived>
bool TensorData::IsCompatible() const
{
    return Derived::IsCompatibleKind(m_data.bufferType);
}

template<typename Derived>
inline Optional<Derived> TensorData::cast() const
{
    static_assert(std::is_base_of<TensorData, Derived>::value, "Cannot cast TensorData to an unrelated type");

    static_assert(sizeof(Derived) == sizeof(TensorData), "The derived type must not add new data members.");

    if (IsCompatible<Derived>())
    {
        return Derived(m_data);
    }
    else
    {
        return NullOpt;
    }
}

// Implementation - TensorDataStrided ----------------------------

inline Byte *TensorDataStrided::basePtr() const
{
    const NVCVTensorBufferStrided &buffer = this->cdata().buffer.strided;
    return reinterpret_cast<Byte *>(buffer.basePtr);
}

inline const int64_t &TensorDataStrided::stride(int d) const
{
    const NVCVTensorData &data = this->cdata();
    if (d < 0 || d >= data.rank)
    {
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "Index of pitch %d is out of bounds [0;%d]", d, data.rank - 1);
    }

    return data.buffer.strided.strides[d];
}

// TensorDataStridedCuda implementation -----------------------

inline TensorDataStridedCuda::TensorDataStridedCuda(const TensorShape &tshape, const DataType &dtype,
                                                    const Buffer &buffer)
{
    NVCVTensorData &data = this->data();

    std::copy(tshape.shape().begin(), tshape.shape().end(), data.shape);
    data.rank   = tshape.rank();
    data.dtype  = dtype;
    data.layout = tshape.layout();

    data.bufferType     = NVCV_TENSOR_BUFFER_STRIDED_CUDA;
    data.buffer.strided = buffer;
}

inline TensorDataStridedCuda::TensorDataStridedCuda(const NVCVTensorData &data)
    : TensorDataStrided(data)
{
    if (!IsCompatibleKind(data.bufferType))
    {
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "Incompatible buffer type.");
    }
}

} // namespace nvcv

#endif // NVCV_TENSORDATA_IMPL_HPP
