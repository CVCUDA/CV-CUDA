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

/**
 * @file TensorWrap.hpp
 *
 * @brief Defines N-D tensor wrapper with N pitches in bytes divided in compile- and run-time pitches.
 */

#ifndef NVCV_CUDA_TENSOR_WRAP_HPP
#define NVCV_CUDA_TENSOR_WRAP_HPP

#include "TypeTraits.hpp" // for HasTypeTraits, etc.

#include <nvcv/ImageData.hpp>        // for ImageDataStridedCuda, etc.
#include <nvcv/TensorData.hpp>       // for TensorDataStridedCuda, etc.
#include <nvcv/TensorDataAccess.hpp> // for TensorDataAccessStridedImagePlanar, etc.

#include <cassert> // for assert, etc.
#include <utility> // for forward, etc.

namespace nvcv::cuda {

/**
 * @defgroup NVCV_CPP_CUDATOOLS_TENSORWRAP TensorWrap classes
 * @{
 */

/**
 * TensorWrap class is a non-owning wrap of a N-D tensor used for easy access of its elements in CUDA device.
 *
 * TensorWrap is a wrapper of a multi-dimensional tensor that can have one or more of its N dimension strides, or
 * pitches, defined either at compile-time or at run-time.  Each pitch in \p Strides represents the offset in bytes
 * as a compile-time template parameter that will be applied from the first (slowest changing) dimension to the
 * last (fastest changing) dimension of the tensor, in that order.  Each dimension with run-time pitch is specified
 * as -1 in the \p Strides template parameter.
 *
 * Template arguments:
 * - T type of the values inside the tensor
 * - StrideT type of the stride used in the byte offset calculation
 * - Strides sequence of compile- or run-time pitches (-1 indicates run-time)
 *   - Y compile-time pitches
 *   - X run-time pitches
 *   - N dimensions, where N = X + Y
 *
 * For example, in the code below a wrap is defined for an NHWC 4D tensor where each sample image in N has a
 * run-time image pitch (first -1 in template argument), and each row in H has a run-time row pitch (second -1), a
 * pixel in W has a compile-time constant pitch as the size of the pixel type and a channel in C has also a
 * compile-time constant pitch as the size of the channel type.
 *
 * @code
 * using DataType = ...;
 * using ChannelType = BaseType<DataType>;
 * using TensorWrap = TensorWrap<ChannelType, -1, -1, sizeof(DataType), sizeof(ChannelType)>;
 * std::byte *imageData = ...;
 * int imgStride = ...;
 * int rowStride = ...;
 * TensorWrap tensorWrap(imageData, imgStride, rowStride);
 * // Elements may be accessed via operator[] using an int4 argument.  They can also be accessed via pointer using
 * // the ptr method with up to 4 integer arguments.
 * @endcode
 *
 * @sa NVCV_CPP_CUDATOOLS_TENSORWRAPS
 *
 * @tparam T Type (it can be const) of each element inside the tensor wrapper.
 * @tparam Strides Each compile-time (use -1 for run-time) pitch in bytes from first to last dimension.
 */
template<typename T, typename StrideT, StrideT... Strides>
class TensorWrapT;

template<typename T, typename StrideT, StrideT... Strides>
class TensorWrapT<const T, StrideT, Strides...>
{
    static_assert(HasTypeTraits<T>, "TensorWrap<T> can only be used if T has type traits");
    static_assert(IsStrideType<StrideT>, "StrideT must be a 64 or 32 bit signed integer type");

public:
    using ValueType  = const T;
    using StrideType = StrideT;

    static constexpr int kNumDimensions   = sizeof...(Strides);
    static constexpr int kVariableStrides = ((Strides == -1) + ...);
    static constexpr int kConstantStrides = kNumDimensions - kVariableStrides;

    TensorWrapT() = default;

    /**
     * Constructs a constant TensorWrap by wrapping a const \p data pointer argument.
     *
     * @param[in] data Pointer to the data that will be wrapped.
     * @param[in] strides0..D Each run-time pitch in bytes from first to last dimension.
     */
    template<typename DataType, typename... Args>
    explicit __host__ __device__ TensorWrapT(const DataType *data, Args... strides)
        : m_data(reinterpret_cast<const std::byte *>(data))
        , m_strides{std::forward<StrideT>(strides)...}
    {
        static_assert((IsIndexType<Args, StrideType> && ...));
        static_assert(sizeof...(Args) == kVariableStrides);
    }

    /**
     * Constructs a constant TensorWrap by wrapping a const \p data pointer argument
     * and copying the dyncamic strides from a given buffer.
     *
     * @param[in] data Pointer to the data that will be wrapped.
     * @param[in] strides Pointer to stride data
     */
    template<typename DataType, typename StrideType>
    explicit __host__ __device__ TensorWrapT(const DataType *data, StrideType *strides)
        : m_data(reinterpret_cast<const std::byte *>(data))
    {
        for (int i = 0; i < kVariableStrides; ++i)
        {
            m_strides[i] = strides[i];
        }
    }

    /**
     * Constructs a constant TensorWrap by wrapping an \p image argument.
     *
     * @param[in] image Image reference to the image that will be wrapped.
     */
    __host__ TensorWrapT(const ImageDataStridedCuda &image)
    {
        static_assert(kVariableStrides == 1 && kNumDimensions == 2);

        m_data = reinterpret_cast<const std::byte *>(image.plane(0).basePtr);

        m_strides[0] = image.plane(0).rowStride;
    }

    /**
     * Constructs a constant TensorWrap by wrapping a \p tensor argument.
     *
     * @param[in] tensor Tensor reference to the tensor that will be wrapped.
     */
    __host__ TensorWrapT(const TensorDataStridedCuda &tensor)
    {
        constexpr StrideT kStride[] = {std::forward<StrideT>(Strides)...};

        assert(tensor.rank() >= kNumDimensions);

        m_data = reinterpret_cast<const std::byte *>(tensor.basePtr());

#pragma unroll
        for (int i = 0; i < kNumDimensions; ++i)
        {
            if (kStride[i] != -1)
            {
                assert(tensor.stride(i) == kStride[i]);
            }
            else if (i < kVariableStrides)
            {
                assert(tensor.stride(i) <= TypeTraits<StrideType>::max);

                m_strides[i] = tensor.stride(i);
            }
        }
    }

    /**
     * Get run-time pitch in bytes.
     *
     * @return The const array (as a pointer) containing run-time pitches in bytes.
     */
    const __host__ __device__ StrideT *strides() const
    {
        return m_strides;
    }

    /**
     * Subscript operator for read-only access.
     *
     * @param[in] c N-D coordinate (from last to first dimension) to be accessed.
     *
     * @return Accessed const reference.
     */
    template<typename DimType, class = Require<std::is_same_v<int, BaseType<DimType>>>>
    inline const __host__ __device__ T &operator[](DimType c) const
    {
        if constexpr (NumElements<DimType> == 1)
        {
            if constexpr (NumComponents<DimType> == 0)
            {
                return *doGetPtr(c);
            }
            else
            {
                return *doGetPtr(c.x);
            }
        }
        else if constexpr (NumElements<DimType> == 2)
        {
            return *doGetPtr(c.y, c.x);
        }
        else if constexpr (NumElements<DimType> == 3)
        {
            return *doGetPtr(c.z, c.y, c.x);
        }
        else if constexpr (NumElements<DimType> == 4)
        {
            return *doGetPtr(c.w, c.z, c.y, c.x);
        }
    }

    /**
     * Get a read-only proxy (as pointer) at the Dth dimension.
     *
     * @param[in] c0..D Each coordinate from first to last dimension.
     *
     * @return The const pointer to the beginning of the Dth dimension.
     */
    template<typename... Args>
    inline const __host__ __device__ T *ptr(Args... c) const
    {
        return doGetPtr(c...);
    }

protected:
    template<typename... Args>
    inline const __host__ __device__ T *doGetPtr(Args... c) const
    {
        static_assert((IsIndexType<Args, StrideType> && ...));
        static_assert(sizeof...(Args) <= kNumDimensions);

        constexpr int     kArgSize  = sizeof...(Args);
        constexpr int     kVarSize  = kArgSize < kVariableStrides ? kArgSize : kVariableStrides;
        constexpr int     kDimSize  = kArgSize < kNumDimensions ? kArgSize : kNumDimensions;
        constexpr StrideT kStride[] = {std::forward<StrideT>(Strides)...};

        StrideType coords[] = {std::forward<StrideType>(c)...};

        // Computing offset first potentially postpones or avoids 64-bit math during addressing
        StrideT offset = 0;
#pragma unroll
        for (int i = 0; i < kVarSize; ++i)
        {
            offset += coords[i] * m_strides[i];
        }
#pragma unroll
        for (int i = kVariableStrides; i < kDimSize; ++i)
        {
            offset += coords[i] * kStride[i];
        }

        return reinterpret_cast<const T *>(m_data + offset);
    }

private:
    const std::byte *m_data                      = nullptr;
    StrideT          m_strides[kVariableStrides] = {};
};

/**
 * Tensor wrapper class specialized for non-constant value type.
 *
 * @tparam T Type (non-const) of each element inside the tensor wrapper.
 * @tparam Strides Each compile-time (use -1 for run-time) pitch in bytes from first to last dimension.
 */
template<typename T, typename StrideT, StrideT... Strides>
class TensorWrapT : public TensorWrapT<const T, StrideT, Strides...>
{
    using Base = TensorWrapT<const T, StrideT, Strides...>;

public:
    using ValueType  = T;
    using StrideType = StrideT;

    using Base::kConstantStrides;
    using Base::kNumDimensions;
    using Base::kVariableStrides;

    TensorWrapT() = default;

    /**
     * Constructs a TensorWrap by wrapping a \p data pointer argument.
     *
     * @param[in] data Pointer to the data that will be wrapped.
     * @param[in] strides0..N Each run-time pitch in bytes from first to last dimension.
     */
    template<typename DataType, typename... Args>
    explicit __host__ __device__ TensorWrapT(DataType *data, Args... strides)
        : Base(data, strides...)
    {
    }

    /**
     * Constructs a TensorWrap by wrapping a const \p data pointer argument
     * and copying the dyncamic strides from a given buffer.
     *
     * @param[in] data Pointer to the data that will be wrapped.
     * @param[in] strides Pointer to stride data
     */
    template<typename DataType, typename StrideType>
    explicit __host__ __device__ TensorWrapT(DataType *data, StrideType *strides)
        : Base(data, strides)
    {
    }

    /**
     * Constructs a TensorWrap by wrapping an \p image argument.
     *
     * @param[in] image Image reference to the image that will be wrapped.
     */
    __host__ TensorWrapT(const ImageDataStridedCuda &image)
        : Base(image)
    {
    }

    /**
     * Constructs a TensorWrap by wrapping a \p tensor argument.
     *
     * @param[in] tensor Tensor reference to the tensor that will be wrapped.
     */
    __host__ TensorWrapT(const TensorDataStridedCuda &tensor)
        : Base(tensor)
    {
    }

    /**
     * Subscript operator for read-and-write access.
     *
     * @param[in] c N-D coordinate (from last to first dimension) to be accessed.
     *
     * @return Accessed reference.
     */
    template<typename DimType, class = Require<std::is_same_v<int, BaseType<DimType>>>>
    inline __host__ __device__ T &operator[](DimType c) const
    {
        if constexpr (NumElements<DimType> == 1)
        {
            if constexpr (NumComponents<DimType> == 0)
            {
                return *doGetPtr(c);
            }
            else
            {
                return *doGetPtr(c.x);
            }
        }
        else if constexpr (NumElements<DimType> == 2)
        {
            return *doGetPtr(c.y, c.x);
        }
        else if constexpr (NumElements<DimType> == 3)
        {
            return *doGetPtr(c.z, c.y, c.x);
        }
        else if constexpr (NumElements<DimType> == 4)
        {
            return *doGetPtr(c.w, c.z, c.y, c.x);
        }
    }

    /**
     * Get a read-and-write proxy (as pointer) at the Dth dimension.
     *
     * @param[in] c0..D Each coordinate from first to last dimension.
     *
     * @return The pointer to the beginning of the Dth dimension.
     */
    template<typename... Args>
    inline __host__ __device__ T *ptr(Args... c) const
    {
        return doGetPtr(c...);
    }

protected:
    template<typename... Args>
    inline __host__ __device__ T *doGetPtr(Args... c) const
    {
        // The const_cast here is the *only* place where it is used to remove the base pointer constness
        return const_cast<T *>(Base::doGetPtr(c...));
    }
};

template<typename T, int64_t... Strides>
using TensorWrap = TensorWrapT<T, int64_t, Strides...>;

template<typename T, int32_t... Strides>
using TensorWrap32 = TensorWrapT<T, int32_t, Strides...>;

/**@}*/

/**
 *  Specializes \ref TensorWrap template classes to different dimensions.
 *
 *  The specializations have the last dimension as the only compile-time dimension as size of T.  All other
 *  dimensions have run-time pitch and must be provided.
 *
 *  Template arguments:
 *  - T data type of each element in \ref TensorWrap
 *  - StrideType stride type used in the TensorWrap
 *  - N (optional) number of dimensions
 *
 *  @sa NVCV_CPP_CUDATOOLS_TENSORWRAP
 *
 *  @defgroup NVCV_CPP_CUDATOOLS_TENSORWRAPS TensorWrap shortcuts
 *  @{
 */

template<typename T, typename StrideType = int64_t>
using Tensor1DWrap = TensorWrapT<T, StrideType, sizeof(T)>;

template<typename T, typename StrideType = int64_t>
using Tensor2DWrap = TensorWrapT<T, StrideType, -1, sizeof(T)>;

template<typename T, typename StrideType = int64_t>
using Tensor3DWrap = TensorWrapT<T, StrideType, -1, -1, sizeof(T)>;

template<typename T, typename StrideType = int64_t>
using Tensor4DWrap = TensorWrapT<T, StrideType, -1, -1, -1, sizeof(T)>;

template<typename T, typename StrideType = int64_t>
using Tensor5DWrap = TensorWrapT<T, StrideType, -1, -1, -1, -1, sizeof(T)>;

template<typename T, int N, typename StrideType = int64_t>
using TensorNDWrap = std::conditional_t<
    N == 1, Tensor1DWrap<T, StrideType>,
    std::conditional_t<
        N == 2, Tensor2DWrap<T, StrideType>,
        std::conditional_t<N == 3, Tensor3DWrap<T, StrideType>,
                           std::conditional_t<N == 4, Tensor4DWrap<T, StrideType>,
                                              std::conditional_t<N == 5, Tensor5DWrap<T, StrideType>, void>>>>>;

/**@}*/

/**
 * Factory function to create an NHW tensor wrap given a tensor data.
 *
 * The output \ref TensorWrap is an NHW 3D tensor allowing to access data per batch (N), per row (H) and per column
 * (W) of the input tensor.  The input tensor data must have either NHWC or HWC layout, where the channel C is
 * inside \p T, e.g. T=uchar3 for RGB8.
 *
 * @sa NVCV_CPP_CUDATOOLS_TENSORWRAP
 *
 * @tparam T Type of the values to be accessed in the tensor wrap.
 * @tparam StrideType Type of the stride used in the tensor wrap.
 *
 * @param[in] tensor Reference to the tensor that will be wrapped.
 *
 * @return Tensor wrap useful to access tensor data in CUDA kernels.
 */

template<typename T, typename StrideType = int64_t, class = Require<HasTypeTraits<T> && IsStrideType<StrideType>>>
__host__ auto CreateTensorWrapNHW(const TensorDataStridedCuda &tensor)
{
    auto tensorAccess = TensorDataAccessStridedImagePlanar::Create(tensor);
    assert(tensorAccess);
    assert(tensorAccess->sampleStride() <= TypeTraits<StrideType>::max);
    assert(tensorAccess->rowStride() <= TypeTraits<StrideType>::max);

    return Tensor3DWrap<T, StrideType>(tensor.basePtr(), static_cast<StrideType>(tensorAccess->sampleStride()),
                                       static_cast<StrideType>(tensorAccess->rowStride()));
}

/**
 * Factory function to create an NHWC tensor wrap given a tensor data.
 *
 * The output \ref TensorWrap is an NHWC 4D tensor allowing to access data per batch (N), per row (H), per column
 * (W) and per channel (C) of the input tensor.  The input tensor data must have either NHWC or HWC layout, where
 * the channel C is of type \p T, e.g. T=uchar for each channel of either RGB8 or RGBA8.
 *
 * @sa NVCV_CPP_CUDATOOLS_TENSORWRAP
 *
 * @tparam T Type of the values to be accessed in the tensor wrap.
 * @tparam StrideType Type of the stride used in the tensor wrap.
 *
 * @param[in] tensor Reference to the tensor that will be wrapped.
 *
 * @return Tensor wrap useful to access tensor data in CUDA kernels.
 */
template<typename T, typename StrideType = int64_t, class = Require<HasTypeTraits<T> && IsStrideType<StrideType>>>
__host__ auto CreateTensorWrapNHWC(const TensorDataStridedCuda &tensor)
{
    auto tensorAccess = TensorDataAccessStridedImagePlanar::Create(tensor);
    assert(tensorAccess);
    assert(tensorAccess->sampleStride() <= TypeTraits<StrideType>::max);
    assert(tensorAccess->rowStride() <= TypeTraits<StrideType>::max);
    assert(tensorAccess->colStride() <= TypeTraits<StrideType>::max);

    return Tensor4DWrap<T, StrideType>(tensor.basePtr(), static_cast<StrideType>(tensorAccess->sampleStride()),
                                       static_cast<StrideType>(tensorAccess->rowStride()),
                                       static_cast<StrideType>(tensorAccess->colStride()));
}

/**
 * Factory function to create an NCHW tensor wrap given a tensor data.
 *
 * The output \ref TensorWrap is an NCHW 4D tensor allowing to access data per batch (N), per channel (C), per row (H), and per column
 * (W) of the input tensor.  The input tensor data must have either NCHW or CHW layout, where
 * the channel C is of type \p T, e.g. T=uchar for each channel of either RGB8 or RGBA8.
 *
 * @sa NVCV_CPP_CUDATOOLS_TENSORWRAP
 *
 * @tparam T Type of the values to be accessed in the tensor wrap.
 *
 * @param[in] tensor Reference to the tensor that will be wrapped.
 *
 * @return Tensor wrap useful to access tensor data in CUDA kernels.
 */
template<typename T, typename StrideType = int64_t, class = Require<HasTypeTraits<T> && IsStrideType<StrideType>>>
__host__ auto CreateTensorWrapNCHW(const TensorDataStridedCuda &tensor)
{
    auto tensorAccess = TensorDataAccessStridedImagePlanar::Create(tensor);
    assert(tensorAccess);
    assert(tensorAccess->sampleStride() <= TypeTraits<StrideType>::max);
    assert(tensorAccess->chStride() <= TypeTraits<StrideType>::max);
    assert(tensorAccess->rowStride() <= TypeTraits<StrideType>::max);

    return Tensor4DWrap<T, StrideType>(tensor.basePtr(), static_cast<StrideType>(tensorAccess->sampleStride()),
                                       static_cast<StrideType>(tensorAccess->chStride()),
                                       static_cast<StrideType>(tensorAccess->rowStride()));
}

} // namespace nvcv::cuda

#endif // NVCV_CUDA_TENSOR_WRAP_HPP
