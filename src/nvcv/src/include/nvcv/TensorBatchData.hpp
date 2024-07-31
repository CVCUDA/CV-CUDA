/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef NVCV_TENSORBATCHDATA_HPP
#define NVCV_TENSORBATCHDATA_HPP

#include "Optional.hpp"
#include "TensorBatchData.h"
#include "TensorShape.hpp"

#include <nvcv/DataType.hpp>
#include <nvcv/TensorData.hpp>

namespace nvcv {

/**
 * @brief General type represenitng data of any tensor batch.
 */
class TensorBatchData
{
public:
    TensorBatchData(const NVCVTensorBatchData &data)
        : m_data(data)
    {
    }

    /**
     * @brief Return rank of the tensors in the batch.
     */
    int rank() const
    {
        return m_data.rank;
    }

    /**
     * @brief Return the layout of the tensors in the batch.
     */
    TensorLayout layout() const
    {
        return m_data.layout;
    }

    /**
     * @brief Return the data type of the tensors in the batch.
     */
    DataType dtype() const
    {
        return DataType(m_data.dtype);
    }

    /**
     * @brief Return the number of the tensors in the batch.
     */
    int32_t numTensors() const
    {
        return m_data.numTensors;
    }

    /**
     * @brief Return underlying C struct representing the tensor batch data.
     */
    NVCVTensorBatchData cdata() const
    {
        return m_data;
    }

    static constexpr bool IsCompatibleKind(NVCVTensorBufferType kind)
    {
        return kind != NVCV_TENSOR_BUFFER_NONE;
    }

    /**
     * @brief Cast the tensor batch data to a derived type (e.g. TensorBatchDataStridedCuda)
     * @tparam Derived target type
     */
    template<typename Derived>
    Optional<Derived> cast() const
    {
        static_assert(std::is_base_of<TensorBatchData, Derived>::value,
                      "Cannot cast TensorBatchData to an unrelated type");
        static_assert(sizeof(Derived) == sizeof(TensorBatchData), "The derived type must not add new data members.");

        if (IsCompatible<Derived>())
        {
            return {Derived(m_data)};
        }
        else
        {
            return {};
        }
    }

    /**
     * @brief Checks if data can be casted to a given derived type.
     * @tparam Derived tested type
     */
    template<typename Derived>
    bool IsCompatible() const
    {
        static_assert(std::is_base_of<TensorBatchData, Derived>::value,
                      "TensorBatchData cannot be compatible with unrelated type");
        return Derived::IsCompatibleKind(m_data.type);
    }

protected:
    TensorBatchData() = default;

    NVCVTensorBatchData &data()
    {
        return m_data;
    }

private:
    NVCVTensorBatchData m_data{};
};

/**
 * @brief Data of batches of tensors with strides.
 */
class TensorBatchDataStrided : public TensorBatchData
{
public:
    using Buffer = NVCVTensorBatchBufferStrided;

    /**
     * @brief Get the buffer with the tensors' descriptors.
     */
    Buffer buffer() const
    {
        return cdata().buffer.strided;
    }

    static constexpr bool IsCompatibleKind(NVCVTensorBufferType kind)
    {
        return kind == NVCV_TENSOR_BUFFER_STRIDED_CUDA;
    }

protected:
    using TensorBatchData::TensorBatchData;
};

/**
 * @brief Data of batches of CUDA tensors with strides.
 */
class TensorBatchDataStridedCuda : public TensorBatchDataStrided
{
public:
    using TensorBatchDataStrided::TensorBatchDataStrided;

    static constexpr bool IsCompatibleKind(NVCVTensorBufferType kind)
    {
        return kind == NVCV_TENSOR_BUFFER_STRIDED_CUDA;
    }
};

} // namespace nvcv

#endif // NVCV_TENSORBATCHDATA_HPP
