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

#ifndef NVCV_TENSORDATA_IMPL_HPP
#define NVCV_TENSORDATA_IMPL_HPP

#ifndef NVCV_TENSORDATA_HPP
#    error "You must not include this header directly"
#endif

#include <algorithm>

namespace nvcv {

// TensorDataStridedCuda implementation -----------------------

inline TensorDataStridedCuda::TensorDataStridedCuda(const TensorShape &tshape, const DataType &dtype,
                                                    const Buffer &buffer)
{
    NVCVTensorData &data = this->cdata();

    std::copy(tshape.shape().begin(), tshape.shape().end(), data.shape);
    data.rank   = tshape.rank();
    data.dtype  = dtype;
    data.layout = tshape.layout();

    data.bufferType     = NVCV_TENSOR_BUFFER_STRIDED_CUDA;
    data.buffer.strided = buffer;
}

inline TensorDataStridedCuda::TensorDataStridedCuda(const NVCVTensorData &data)
    : ITensorDataStridedCuda(data)
{
}

} // namespace nvcv

#endif // NVCV_TENSORDATA_IMPL_HPP
