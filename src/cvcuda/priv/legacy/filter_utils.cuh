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

#ifndef FILTER_UTILS_CUH
#define FILTER_UTILS_CUH

#include <nvcv/cuda/TensorWrap.hpp> // for TensorWrap, etc.

namespace nvcv::legacy::cuda_op {

__global__ void computeMeanKernel(float *kernel_ptr, int k_size);

__global__ void computeGaussianKernel(float *kernel, Size2D kernelSize, double2 sigma);

__global__ void computeMeanKernelVarShape(cuda::Tensor3DWrap<float> kernel, cuda::Tensor1DWrap<int2> kernelSizeArr,
                                          cuda::Tensor1DWrap<int2> kernelAnchorArr);

__global__ void computeGaussianKernelVarShape(cuda::Tensor3DWrap<float> kernel, int dataKernelSize,
                                              Size2D maxKernelSize, cuda::Tensor1DWrap<int2> kernelSizeArr,
                                              cuda::Tensor1DWrap<double2> sigmaArr);

__global__ void computeMeanKernelVarShape(cuda::Tensor3DWrap<float> kernel, cuda::Tensor1DWrap<int> blockSizeArr);

__global__ void computeGaussianKernelVarShape(cuda::Tensor3DWrap<float> kernel, cuda::Tensor1DWrap<int> blockSizeArr);

} // namespace nvcv::legacy::cuda_op

#endif // FILTER_UTILS_CUH
