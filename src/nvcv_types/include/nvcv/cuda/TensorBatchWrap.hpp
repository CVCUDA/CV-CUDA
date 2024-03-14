/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * @file TensorBatchWrap.hpp
 *
 * @brief Defines a wrapper of a tensor batch.
 */

#ifndef NVCV_CUDA_TENSOR_BATCH_WRAP_HPP
#define NVCV_CUDA_TENSOR_BATCH_WRAP_HPP

#include "TypeTraits.hpp" // for HasTypeTraits, etc
#include "nvcv/TensorBatchData.hpp"
#include "nvcv/cuda/TensorWrap.hpp"

#include <type_traits>

namespace nvcv::cuda {

/**
 * @defgroup NVCV_CPP_CUDATOOLS_TENSORBATCHWRAP TensorBatchWrap classes
 * @{
 */

/**
 * TensorBatchWrap class is a non-owning wrap of a batch of N-D tensors used for easy access of its elements in CUDA device.
 *
 * TensorBatchWrap is a wrapper of a batch of multi-dimensional tensors that can have one or more of its N dimension strides, or
 * pitches, defined either at compile-time or at run-time. Each pitch in \p Strides represents the offset in bytes
 * as a compile-time template parameter that will be applied from the first (slowest changing) dimension to the
 * last (fastest changing) dimension of the tensor, in that order.  Each dimension with run-time pitch is specified
 * as -1 in the \p Strides template parameter.
 *
 * Template arguments:
 * - T type of the values inside the tensors
 * - Strides sequence of compile- or run-time pitches (-1 indicates run-time)
 *   - Y compile-time pitches
 *   - X run-time pitches
 *   - N dimensions, where N = X + Y
 *
 * For example, in the code below a wrap is defined for a batch of HWC 3D tensors where each row in H
 * has a run-time row pitch (second -1), a pixel in W has a compile-time constant pitch as
 * the size of the pixel type and a channel in C has also a compile-time constant pitch as
 * the size of the channel type.
 *
 * @code
 * using DataType = ...;
 * using ChannelType = BaseType<DataType>;
 * using TensorBatchWrap = TensorBatchWrap<ChannelType, -1, sizeof(DataType), sizeof(ChannelType)>;
 * TensorBatch tensorBatch = ...;
 * TensorBatchWrap tensorBatchWrap(tensorBatch.data());
 * // Elements may be accessed via operator[] using an int4 argument.  They can also be accessed via pointer using
 * // the ptr method with up to 4 integer arguments or by accessing each TensorWrap separately with tensor(...) method.
 * @endcode
 *
 * @sa NVCV_CPP_CUDATOOLS_TENSORBATCHWRAPS
 *
 * @tparam T Type (it can be const) of each element inside the tensor wrapper.
 * @tparam Strides Each compile-time (use -1 for run-time) pitch in bytes from first to last dimension.
 */
template<typename T, int... Strides>
class TensorBatchWrap;

template<typename T, int... Strides>
class TensorBatchWrap<const T, Strides...>
{
    static_assert(HasTypeTraits<T>, "TensorBatchWrap<T> can only be used if T has type traits");

public:
    // The type provided as template parameter is the value type, i.e. the type of each element inside this wrapper.
    using ValueType = const T;

    static constexpr int kNumDimensions   = sizeof...(Strides);
    static constexpr int kVariableStrides = ((Strides == -1) + ...);
    static constexpr int kConstantStrides = kNumDimensions - kVariableStrides;

    TensorBatchWrap() = default;

    /**
     * Constructs a constant TensorBatchWrap by wrapping a \p data argument.
     *
     * @param[in] data Tensor batch data to wrap.
     */
    __host__ TensorBatchWrap(const TensorBatchDataStridedCuda &data)
        : TensorBatchWrap(data.cdata())
    {
    }

    /**
     * Constructs a constant TensorBatchWrap by wrapping a \p data argument.
     *
     * @param[in] data Tensor batch data to wrap.
     */
    __host__ __device__ TensorBatchWrap(const NVCVTensorBatchData &data)
        : m_numTensors(data.numTensors)
        , m_tensors(data.buffer.strided.tensors)
    {
    }

    /**
     * Get a read-only proxy (as pointer) of the given tensor at the given coordinates.
     *
     * @param[in] t Tensor index in the list.
     * @param[in] c Coordinates in the given tensor;
     *
     * @return The const pointer to the beginning of the given coordinates.
     */
    template<typename... Coords>
    inline const __host__ __device__ T *ptr(int t, Coords... c) const
    {
        return doGetPtr(t, c...);
    }

    /**
     * Subscript operator for read-and-write access.
     *
     * @param[in] t Tensor index in the list.
     * @param[in] c (N+1)-D coordinates tensor index and coords (from last to first dimension) to be accessed.
     *              E.g. for a 2-dimensional tensors, the coordinates would be: {tensor_id, column, row}
     *
     * @return Accessed reference.
     */
    template<typename DimType, class = Require<std::is_same_v<int, BaseType<DimType>>>>
    inline const __host__ __device__ T &operator[](DimType c) const
    {
        static_assert(NumElements<DimType> == kNumDimensions + 1,
                      "Coordinates in the subscript operator must be (N+1)-dimensional, "
                      "where N is a dimensionality of a single tensor in the batch.");
        if constexpr (NumElements<DimType> == 1)
        {
            return *doGetPtr(c.x);
        }
        if constexpr (NumElements<DimType> == 2)
        {
            return *doGetPtr(c.x, c.y);
        }
        else if constexpr (NumElements<DimType> == 3)
        {
            return *doGetPtr(c.x, c.z, c.y);
        }
        else if constexpr (NumElements<DimType> == 4)
        {
            return *doGetPtr(c.x, c.w, c.z, c.y);
        }
    }

    /**
     * @brief Constructs a read-only wrapper for the tensor on index \p t
     * The list of static strides can be provided as a template parameter.
     * It should be a list of N outer strides (from inner to outer).
     *
     * @tparam Strides static strides
     * @param t index of the tensor
     */
    inline const __host__ __device__ auto tensor(int t) const
    {
        return TensorWrap<ValueType, Strides...>(doGetPtr(t), strides(t));
    }

    /**
     * @brief Returns a number of tensors in the batch.
     */
    inline __host__ __device__ int32_t numTensors() const
    {
        return m_numTensors;
    }

    /**
     * @brief Returns a pointer to shape buffer of the tensor at index \p t
     *
     * @param t tensor index
     */
    inline const __host__ __device__ int64_t *shape(int t) const
    {
        assert(t >= 0 && t < m_numTensors);
        return m_tensors[t].shape;
    }

    /**
     * @brief Returns a pointer to a stride buffer of the tensor at index \p t
     *
     * @param t tensor index
     */
    inline const __host__ __device__ int64_t *strides(int t) const
    {
        assert(t >= 0 && t < m_numTensors);
        return m_tensors[t].stride;
    }

protected:
    template<typename... Args>
    inline __host__ __device__ T *doGetPtr(int t, Args... c) const
    {
        static_assert(std::conjunction_v<std::is_same<int, Args>...>);
        static_assert(sizeof...(Args) <= kNumDimensions);

        constexpr int kArgSize  = sizeof...(Args);
        constexpr int kVarSize  = kArgSize < kVariableStrides ? kArgSize : kVariableStrides;
        constexpr int kDimSize  = kArgSize < kNumDimensions ? kArgSize : kNumDimensions;
        constexpr int kStride[] = {std::forward<int>(Strides)...};

        // Computing offset first potentially postpones or avoids 64-bit math during addressing
        int offset = 0;
        if constexpr (kArgSize > 0)
        {
            int            coords[] = {std::forward<int>(c)...};
            const int64_t *strides  = m_tensors[t].stride;

#pragma unroll
            for (int i = 0; i < kVarSize; ++i)
            {
                offset += coords[i] * strides[i];
            }
#pragma unroll
            for (int i = kVariableStrides; i < kDimSize; ++i)
            {
                offset += coords[i] * kStride[i];
            }
        }

        NVCVByte *dataPtr = m_tensors[t].data;
        return reinterpret_cast<T *>(dataPtr + offset);
    }

    int32_t                           m_numTensors;
    NVCVTensorBatchElementStridedRec *m_tensors;
};

/**
 * TensorBatch wrapper class specialized for non-constant value type.
 *
 * @tparam T Type (non-const) of each element inside the tensor batch wrapper.
 * @tparam Strides Each compile-time (use -1 for run-time) pitch in bytes from first to last dimension.
 */
template<typename T, int... Strides>
class TensorBatchWrap : public TensorBatchWrap<const T, Strides...>
{
    using Base = TensorBatchWrap<const T, Strides...>;

public:
    using ValueType = T;
    using Base::doGetPtr;
    using Base::kNumDimensions;
    using Base::m_tensors;
    using Base::strides;

    /**
     * Constructs a TensorBatchWrap by wrapping a \p data argument.
     *
     * @param[in] data Tensor batch data to wrap.
     */
    __host__ TensorBatchWrap(const TensorBatchDataStridedCuda &data)
        : Base(data)
    {
    }

    /**
     * Constructs a TensorBatchWrap by wrapping a \p data argument.
     *
     * @param[in] data Tensor batch data to wrap.
     */
    __host__ __device__ TensorBatchWrap(NVCVTensorBatchData &data)
        : Base(data)
    {
    }

    /**
     * Get a read-and-write proxy (as pointer) of the given tensor at the given coordinates.
     *
     * @param[in] t Tensor index in the list.
     * @param[in] c Coordinates in the given tensor;
     *
     * @return The const pointer to the beginning of the given coordinates.
     */
    template<typename... Coords>
    inline __host__ __device__ T *ptr(int t, Coords... c) const
    {
        return doGetPtr(t, c...);
    }

    /**
     * @brief Constructs a read-and-write wrapper for the tensor on index \p t
     * The list of static strides can be provided as a template parameter.
     * It should be a list of N outer strides (from inner to outer).
     *
     * @tparam Strides static strides
     * @param t index of the tensor
     */
    inline __host__ __device__ auto tensor(int t) const
    {
        return TensorWrap<ValueType, Strides...>(doGetPtr(t), strides(t));
    }

    /**
     * Subscript operator for read-and-write access.
     *
     * @param[in] t Tensor index in the list.
     * @param[in] c (N+1)-D coordinates - tensor index and coords (from inner to outer) to be accessed.
     *              E.g. for a 2-dimensional tensors, the coordinates would be: {tensor_id, column, row}
     *
     * @return Accessed reference.
     */
    template<typename DimType, class = Require<std::is_same_v<int, BaseType<DimType>>>>
    inline __host__ __device__ T &operator[](DimType c) const
    {
        static_assert(NumElements<DimType> == kNumDimensions + 1,
                      "Coordinates in the subscript operator must be (N+1)-dimensional, "
                      "where N is a dimensionality of a single tensor in the batch.");
        if constexpr (NumElements<DimType> == 1)
        {
            return *doGetPtr(c.x);
        }
        if constexpr (NumElements<DimType> == 2)
        {
            return *doGetPtr(c.x, c.y);
        }
        else if constexpr (NumElements<DimType> == 3)
        {
            return *doGetPtr(c.x, c.z, c.y);
        }
        else if constexpr (NumElements<DimType> == 4)
        {
            return *doGetPtr(c.x, c.w, c.z, c.y);
        }
    }
};

/**@}*/

/**
 *  Specializes \ref TensorBatchWrap template classes to different dimensions.
 *
 *  The specializations have the last dimension as the only compile-time dimension as size of T.  All other
 *  dimensions have run-time pitch and must be provided.
 *
 *  Template arguments:
 *  - T data type of each element in \ref TensorBatchWrap
 *
 *  @sa NVCV_CPP_CUDATOOLS_TENSORBATCHWRAP
 *
 *  @defgroup NVCV_CPP_CUDATOOLS_TENSORBATCHWRAPS TensorBatchWrap shortcuts
 *  @{
 */

template<typename T>
using TensorBatch1DWrap = TensorBatchWrap<T, sizeof(T)>;

template<typename T>
using TensorBatch2DWrap = TensorBatchWrap<T, -1, sizeof(T)>;

template<typename T>
using TensorBatch3DWrap = TensorBatchWrap<T, -1, -1, sizeof(T)>;

template<typename T>
using TensorBatch4DWrap = TensorBatchWrap<T, -1, -1, -1, sizeof(T)>;

template<typename T>
using TensorBatch5DWrap = TensorBatchWrap<T, -1, -1, -1, -1, sizeof(T)>;

template<typename T, int N>
using TensorBatchNDWrap = std::conditional_t<
    N == 1, TensorBatch1DWrap<T>,
    std::conditional_t<N == 2, TensorBatch2DWrap<T>,
                       std::conditional_t<N == 3, TensorBatch3DWrap<T>,
                                          std::conditional_t<N == 4, TensorBatch4DWrap<T>,
                                                             std::conditional_t<N == 5, TensorBatch5DWrap<T>, void>>>>>;
/**@}*/

} // namespace nvcv::cuda

#endif // NVCV_CUDA_TENSOR_BATCH_WRAP_HPP
