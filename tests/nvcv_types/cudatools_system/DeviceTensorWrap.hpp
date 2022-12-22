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

#ifndef NVCV_TESTS_DEVICE_TENSOR_WRAP_HPP
#define NVCV_TESTS_DEVICE_TENSOR_WRAP_HPP

#include <cuda_runtime.h>           // for int2, etc.
#include <nvcv/cuda/TensorWrap.hpp> // the object of this test
#include <nvcv/cuda/TypeTraits.hpp> // for NumElements, etc.

#include <array> // for std::array, etc.

// clang-format off

template<typename T, int H, int W>
struct PackedImage { // PackedImage extends std::array in two dimensions
    using value_type = T;
    static constexpr int rowStride = W * sizeof(T), height = H, width = W;
    std::array<T, H*W> m_data;
    T *data() { return m_data.data(); }
    const T *data() const { return m_data.data(); }
    const T& operator[](int i) const { return m_data[i]; }
    bool operator==(const PackedImage<T, H, W> &that) const { return m_data == that.m_data; }
};

template<typename T, int N, int H, int W>
struct PackedTensor3D { // PackedTensor3D extends std::array in three dimensions
    using value_type = T;
    static constexpr int batches = N, height = H, width = W;
    static constexpr int stride2 = W * sizeof(T);
    static constexpr int stride1 = H * stride2;

    std::array<T, N*H*W> m_data;
    T *data() { return m_data.data(); }
    const T *data() const { return m_data.data(); }
    const T& operator[](int i) const { return m_data[i]; }
    bool operator==(const PackedTensor3D<T, N, H, W> &that) const { return m_data == that.m_data; }
};

template<typename T, int N, int H, int W, int C>
struct PackedTensor4D { // PackedTensor4D extends std::array in four dimensions
    using value_type = T;
    static constexpr int batches = N, height = H, width = W, channels = C;
    static constexpr int stride3 = C * sizeof(T);
    static constexpr int stride2 = W * stride3;
    static constexpr int stride1 = H * stride2;

    std::array<T, N*H*W*C> m_data;
    T *data() { return m_data.data(); }
    const T *data() const { return m_data.data(); }
    const T& operator[](int i) const { return m_data[i]; }
    bool operator==(const PackedTensor4D<T, N, H, W, C> &that) const { return m_data == that.m_data; }
};

// clang-format on

template<typename ValueType, std::size_t N>
void DeviceUseTensor1DWrap(std::array<ValueType, N> &);

template<typename ValueType, int H, int W>
void DeviceUseTensor2DWrap(PackedImage<ValueType, H, W> &);

template<typename ValueType, int N, int H, int W>
void DeviceUseTensor3DWrap(PackedTensor3D<ValueType, N, H, W> &);

template<typename ValueType, int N, int H, int W, int C>
void DeviceUseTensor4DWrap(PackedTensor4D<ValueType, N, H, W, C> &);

template<typename ValueType>
void DeviceSetOnes(nvcv::cuda::Tensor1DWrap<ValueType> &, int1, cudaStream_t &);

template<typename ValueType>
void DeviceSetOnes(nvcv::cuda::Tensor2DWrap<ValueType> &, int2, cudaStream_t &);

template<typename ValueType>
void DeviceSetOnes(nvcv::cuda::Tensor3DWrap<ValueType> &, int3, cudaStream_t &);

template<typename ValueType>
void DeviceSetOnes(nvcv::cuda::Tensor4DWrap<ValueType> &, int4, cudaStream_t &);

#endif // NVCV_TESTS_DEVICE_TENSOR_WRAP_HPP
