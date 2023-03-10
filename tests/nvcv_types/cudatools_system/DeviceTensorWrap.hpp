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

#ifndef NVCV_TESTS_DEVICE_TENSOR_WRAP_HPP
#define NVCV_TESTS_DEVICE_TENSOR_WRAP_HPP

#include <cuda_runtime.h>           // for int2, etc.
#include <nvcv/cuda/TensorWrap.hpp> // the object of this test
#include <nvcv/cuda/TypeTraits.hpp> // for MakeType, etc.

#include <array> // for std::array, etc.

// clang-format off

template<typename T, int NumDim>
constexpr int TotalBytes(const int (&shapes)[NumDim])
{
    int total = sizeof(T);
    for (int i = 0; i < NumDim; ++i)
    {
        total *= shapes[i];
    }
    return total;
}

template<typename T, int N>
struct Array { // Array extends std::array to add extra properties
    using value_type = T;
    static constexpr int kNumDim           = 1;
    static constexpr int kStrides[kNumDim] = {sizeof(T)};
    static constexpr int kShapes[kNumDim]  = {N};
    static constexpr dim3 kBlocks          = {N};

    std::array<T, N> m_data;
    T *data() { return m_data.data(); }
    const T *data() const { return m_data.data(); }
    const T& operator[](int i) const { return m_data[i]; }
    bool operator==(const Array<T, N> &that) const { return m_data == that.m_data; }
};

template<typename T, int H, int W>
struct PackedImage { // PackedImage extends std::array in two dimensions
    using value_type = T;
    static constexpr int kNumDim           = 2;
    static constexpr int kStrides[kNumDim] = {W * sizeof(T), sizeof(T)};
    static constexpr int kShapes[kNumDim]  = {H, W};
    static constexpr dim3 kBlocks          = {W, H};

    std::array<T, H*W> m_data;
    T *data() { return m_data.data(); }
    const T *data() const { return m_data.data(); }
    const T& operator[](int i) const { return m_data[i]; }
    bool operator==(const PackedImage<T, H, W> &that) const { return m_data == that.m_data; }
};

template<typename T, int N, int H, int W>
struct PackedTensor3D { // PackedTensor3D extends std::array in three dimensions
    using value_type = T;
    static constexpr int kNumDim           = 3;
    static constexpr int kStrides[kNumDim] = {H * W * sizeof(T), W * sizeof(T), sizeof(T)};
    static constexpr int kShapes[kNumDim]  = {N, H, W};
    static constexpr dim3 kBlocks          = {W, H, N};

    std::array<T, N*H*W> m_data;
    T *data() { return m_data.data(); }
    const T *data() const { return m_data.data(); }
    const T& operator[](int i) const { return m_data[i]; }
    bool operator==(const PackedTensor3D<T, N, H, W> &that) const { return m_data == that.m_data; }
};

template<typename T, int N, int H, int W, int C>
struct PackedTensor4D { // PackedTensor4D extends std::array in four dimensions
    using value_type = T;
    static constexpr int kNumDim           = 4;
    static constexpr int kStrides[kNumDim] = {H * W * C * sizeof(T), W * C * sizeof(T), C * sizeof(T), sizeof(T)};
    static constexpr int kShapes[kNumDim]  = {N, H, W, C};
    static constexpr dim3 kBlocks          = {W, H, N};

    std::array<T, N*H*W*C> m_data;
    T *data() { return m_data.data(); }
    const T *data() const { return m_data.data(); }
    const T& operator[](int i) const { return m_data[i]; }
    bool operator==(const PackedTensor4D<T, N, H, W, C> &that) const { return m_data == that.m_data; }
};

// clang-format on

template<class InputType>
void DeviceUseTensorWrap(const InputType &);

template<class DstWrapper, typename DimType>
void DeviceSetOnes(DstWrapper &, DimType, cudaStream_t &);

#endif // NVCV_TESTS_DEVICE_TENSOR_WRAP_HPP
