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

#ifndef NVCV_ITENSORDATA_IMPL_HPP
#define NVCV_ITENSORDATA_IMPL_HPP

#ifndef NVCV_ITENSORDATA_HPP
#    error "You must not include this header directly"
#endif

namespace nvcv {

// Implementation - ITensorData -----------------------------

inline ITensorData::ITensorData(const NVCVTensorData &data)
    : m_data(data)
{
}

inline ITensorData::~ITensorData()
{
    // required dtor implementation
}

inline int ITensorData::rank() const
{
    return this->cdata().rank;
}

inline const TensorShape &ITensorData::shape() const
{
    if (!m_cacheShape)
    {
        const NVCVTensorData &data = this->cdata();
        m_cacheShape.emplace(data.shape, data.rank, data.layout);
    }

    return *m_cacheShape;
}

inline auto ITensorData::shape(int d) const -> const TensorShape::DimType &
{
    const NVCVTensorData &data = this->cdata();

    if (d < 0 || d >= data.rank)
    {
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "Index of shape dimension %d is out of bounds [0;%d]", d,
                        data.rank - 1);
    }
    return data.shape[d];
}

inline const TensorLayout &ITensorData::layout() const
{
    return this->shape().layout();
}

inline DataType ITensorData::dtype() const
{
    const NVCVTensorData &data = this->cdata();
    return DataType{data.dtype};
}

inline const NVCVTensorData &ITensorData::cdata() const
{
    return m_data;
}

inline NVCVTensorData &ITensorData::cdata()
{
    // data contents might be modified, must reset cache
    m_cacheShape.reset();
    return m_data;
}

// Implementation - ITensorDataStrided ----------------------------

inline ITensorDataStrided::~ITensorDataStrided()
{
    // required dtor implementation
}

inline Byte *ITensorDataStrided::basePtr() const
{
    const NVCVTensorBufferStrided &buffer = this->cdata().buffer.strided;
    return reinterpret_cast<Byte *>(buffer.basePtr);
}

inline const int64_t &ITensorDataStrided::stride(int d) const
{
    const NVCVTensorData &data = this->cdata();
    if (d < 0 || d >= data.rank)
    {
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "Index of pitch %d is out of bounds [0;%d]", d, data.rank - 1);
    }

    return data.buffer.strided.strides[d];
}

// Implementation - ITensorDataStridedCuda ----------------------------
inline ITensorDataStridedCuda::~ITensorDataStridedCuda()
{
    // required dtor implementation
}

} // namespace nvcv

#endif // NVCV_ITENSORDATA_IMPL_HPP
