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

#ifndef NVCV_TENSORDATA_HPP
#define NVCV_TENSORDATA_HPP

#include "ITensorData.hpp"
#include "TensorShape.hpp"

#include <nvcv/DataType.hpp>

namespace nvcv {

// TensorDataStridedCuda definition -----------------------

class TensorDataStridedCuda : public ITensorDataStridedCuda
{
public:
    using Buffer = NVCVTensorBufferStrided;

    explicit TensorDataStridedCuda(const TensorShape &shape, const DataType &dtype, const Buffer &data);
    explicit TensorDataStridedCuda(const NVCVTensorData &data);
};

} // namespace nvcv

#include "detail/TensorDataImpl.hpp"

#endif // NVCV_TENSORDATA_HPP
