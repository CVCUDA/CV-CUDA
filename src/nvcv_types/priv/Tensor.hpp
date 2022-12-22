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

#ifndef NVCV_CORE_PRIV_TENSOR_HPP
#define NVCV_CORE_PRIV_TENSOR_HPP

#include "ITensor.hpp"

#include <cuda_runtime.h>

namespace nvcv::priv {

class Tensor final : public CoreObjectBase<ITensor>
{
public:
    explicit Tensor(NVCVTensorRequirements reqs, IAllocator &alloc);
    ~Tensor();

    static NVCVTensorRequirements CalcRequirements(int32_t numImages, Size2D imgSize, ImageFormat fmt,
                                                   int32_t baseAlign, int32_t rowAlign);
    static NVCVTensorRequirements CalcRequirements(int rank, const int64_t *shape, const DataType &dtype,
                                                   NVCVTensorLayout layout, int32_t baseAlign, int32_t rowAlign);

    int32_t        rank() const override;
    const int64_t *shape() const override;

    const NVCVTensorLayout &layout() const override;

    DataType dtype() const override;

    IAllocator &alloc() const override;

    void exportData(NVCVTensorData &data) const override;

private:
    IAllocator            &m_alloc;
    NVCVTensorRequirements m_reqs;

    void *m_memBuffer;
};

} // namespace nvcv::priv

#endif // NVCV_CORE_PRIV_TENSOR_HPP
