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

#ifndef NVCV_TENSORDATA_HPP
#define NVCV_TENSORDATA_HPP

#include "Optional.hpp"
#include "TensorData.h"
#include "TensorShape.hpp"

#include <nvcv/DataType.hpp>

namespace nvcv {

/**
 * @brief Represents data for a tensor in the system.
 *
 * The TensorData class provides an interface to access and manage the underlying data of a tensor.
 * It offers functionalities to retrieve tensor shape, layout, data type, and other properties.
 * The tensor's data is encapsulated in an `NVCVTensorData` object.
 */
class TensorData
{
public:
    /**
     * @brief Constructs a TensorData object from an `NVCVTensorData` instance.
     *
     * @param data The underlying tensor data representation.
     */
    TensorData(const NVCVTensorData &data);

    /// @brief Retrieves the rank (number of dimensions) of the tensor.
    int rank() const;

    /// @brief Retrieves the shape of the tensor.
    const TensorShape &shape() const &;

    /**
     * @brief Retrieves a specific dimension size from the tensor shape.
     *
     * @param d The index of the dimension.
     * @return The size of the specified dimension.
     */
    const TensorShape::DimType &shape(int d) const &;

    /// @brief Retrieves the layout of the tensor.
    const TensorLayout &layout() const &;

    /// @brief Retrieves the shape of the tensor (rvalue overload).
    TensorShape shape() &&
    {
        return this->shape();
    }

    /// @brief Retrieves a specific dimension size from the tensor shape (rvalue overload).
    TensorShape::DimType shape(int d) &&
    {
        return this->shape(d);
    }

    /// @brief Retrieves the layout of the tensor (rvalue overload).
    TensorLayout layout() &&
    {
        return this->layout();
    }

    /// @brief Retrieves the data type of the tensor elements.
    DataType dtype() const;

    /// @brief Retrieves a constant reference to the underlying tensor data.
    const NVCVTensorData &cdata() const &;

    /// @brief Retrieves the underlying tensor data (rvalue overload).
    NVCVTensorData cdata() &&
    {
        return this->cdata();
    }

    /**
     * @brief Determines if a given tensor buffer type is compatible.
     *
     * @param kind The tensor buffer type to check.
     * @return true if the buffer type is compatible, false otherwise.
     */
    static bool IsCompatibleKind(NVCVTensorBufferType kind)
    {
        return kind != NVCV_TENSOR_BUFFER_NONE;
    }

    /**
     * @brief Attempts to cast the current tensor data to a derived tensor data type.
     *
     * @tparam DerivedTensorData The derived tensor data type to cast to.
     * @return An optional containing the casted tensor data if successful; otherwise, an empty optional.
     */
    template<typename DerivedTensorData>
    Optional<DerivedTensorData> cast() const;

    /**
     * @brief Checks if the current tensor data is compatible with a derived type.
     *
     * @tparam Derived The derived type to check compatibility against.
     * @return true if the current tensor data is compatible with the derived type, false otherwise.
     */
    template<typename Derived>
    bool IsCompatible() const;

protected:
    TensorData() = default;

    NVCVTensorData &data() &;

private:
    NVCVTensorData                m_data{};
    mutable Optional<TensorShape> m_cacheShape;
};

/**
 * @brief Represents strided tensor data in the system.
 *
 * The `TensorDataStrided` class extends `TensorData` to handle tensor data that is stored in a strided manner.
 * Strided tensor data allows non-contiguous storage in memory, where each dimension can have its own stride and the stride is the amount of bytes to jump each element in that dimension.
 * */
class TensorDataStrided : public TensorData
{
public:
    /**
     * @brief Retrieves the base pointer of the tensor data in memory.
     *
     * @return A pointer to the base (starting address) of the tensor data.
     */
    Byte *basePtr() const;

    /**
     * @brief Retrieves the stride for a specific dimension of the tensor.
     *
     * @param d The index of the dimension.
     * @return The stride of the specified dimension.
     */
    const int64_t &stride(int d) const;

    /**
     * @brief Determines if a given tensor buffer type is compatible with strided data.
     *
     * @param kind The tensor buffer type to check.
     * @return true if the buffer type is compatible with strided data, false otherwise.
     */
    static bool IsCompatibleKind(NVCVTensorBufferType kind)
    {
        return kind == NVCV_TENSOR_BUFFER_STRIDED_CUDA;
    }

protected:
    using TensorData::TensorData;
};

/**
 * @brief Represents strided tensor data specifically for CUDA.
 *
 * The `TensorDataStridedCuda` class extends `TensorDataStrided` to handle tensor data stored in a strided manner on CUDA devices.
 * It provides methods specific to CUDA strided tensor data.
 */
class TensorDataStridedCuda : public TensorDataStrided
{
public:
    using Buffer = NVCVTensorBufferStrided;

    /**
     * @brief Constructs a `TensorDataStridedCuda` object from an `NVCVTensorData` instance.
     *
     * @param data The underlying tensor data representation.
     */
    TensorDataStridedCuda(const NVCVTensorData &data);

    /**
     * @brief Constructs a `TensorDataStridedCuda` object from tensor shape, data type, and buffer.
     *
     * @param tshape Shape of the tensor.
     * @param dtype Data type of the tensor elements.
     * @param buffer The underlying strided buffer for CUDA.
     */

    TensorDataStridedCuda(const TensorShape &tshape, const DataType &dtype, const Buffer &buffer);

    /**
     * @brief Determines if a given tensor buffer type is compatible with CUDA strided data.
     *
     * @param kind The tensor buffer type to check.
     * @return true if the buffer type is compatible with CUDA strided data, false otherwise.
     */
    static bool IsCompatibleKind(NVCVTensorBufferType kind)
    {
        return kind == NVCV_TENSOR_BUFFER_STRIDED_CUDA;
    }
};

} // namespace nvcv

#include "detail/TensorDataImpl.hpp"

#endif // NVCV_TENSORDATA_HPP
