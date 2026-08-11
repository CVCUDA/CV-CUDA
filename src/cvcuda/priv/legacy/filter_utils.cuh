/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef FILTER_UTILS_CUH
#define FILTER_UTILS_CUH

#include <cvcuda/cuda_tools/TensorWrap.hpp> // for TensorWrap, etc.

#include <type_traits>

namespace nvcv::legacy::cuda_op {

// Keep scalar planar and compound interleaved kernels on the same explicitly rounded instruction sequence.
template<typename T>
__device__ __forceinline__ T LaplacianFloatMulRN(T value, float coefficient)
{
    static_assert(std::is_same_v<cuda::BaseType<T>, float>);

    T result;
#pragma unroll
    for (int c = 0; c < cuda::NumElements<T>; ++c)
    {
        cuda::GetElement(result, c) = __fmul_rn(cuda::GetElement(value, c), coefficient);
    }
    return result;
}

template<typename T>
__device__ __forceinline__ T LaplacianFloatFmaRN(T value, float coefficient, T result)
{
    static_assert(std::is_same_v<cuda::BaseType<T>, float>);

#pragma unroll
    for (int c = 0; c < cuda::NumElements<T>; ++c)
    {
        cuda::GetElement(result, c) = __fmaf_rn(cuda::GetElement(value, c), coefficient, cuda::GetElement(result, c));
    }
    return result;
}

__global__ void computeMeanKernel(float *kernel_ptr, int k_size);

__global__ void computeGaussianKernel(float *kernel, Size2D kernelSize, double2 sigma);

__global__ void computeMeanKernelVarShape(cuda::Tensor3DWrap<float, int32_t> kernel,
                                          cuda::Tensor1DWrap<int2, int32_t>  kernelSizeArr,
                                          cuda::Tensor1DWrap<int2, int32_t>  kernelAnchorArr);

__global__ void computeGaussianKernelVarShape(cuda::Tensor3DWrap<float, int32_t> kernel, int dataKernelSize,
                                              Size2D maxKernelSize, cuda::Tensor1DWrap<int2, int32_t> kernelSizeArr,
                                              cuda::Tensor1DWrap<double2, int32_t> sigmaArr);

__global__ void computeMeanKernelVarShape(cuda::Tensor3DWrap<float, int32_t> kernel,
                                          cuda::Tensor1DWrap<int, int32_t>   blockSizeArr);

__global__ void computeGaussianKernelVarShape(cuda::Tensor3DWrap<float, int32_t> kernel,
                                              cuda::Tensor1DWrap<int, int32_t>   blockSizeArr);

} // namespace nvcv::legacy::cuda_op

#endif // FILTER_UTILS_CUH
