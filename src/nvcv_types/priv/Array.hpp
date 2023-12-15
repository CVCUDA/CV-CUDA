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

#ifndef NVCV_CORE_PRIV_ARRAY_HPP
#define NVCV_CORE_PRIV_ARRAY_HPP

#include "IAllocator.hpp"
#include "IArray.hpp"
#include "SharedCoreObj.hpp"

#include <cuda_runtime.h>

namespace nvcv::priv {

class Array final : public CoreObjectBase<IArray>
{
public:
    explicit Array(NVCVArrayRequirements reqs, IAllocator &alloc, NVCVResourceType target);
    ~Array();

    static NVCVArrayRequirements CalcRequirements(int64_t capacity, const DataType &dtype, int32_t alignment,
                                                  NVCVResourceType target = NVCV_RESOURCE_MEM_CUDA);

    int32_t rank() const override;
    int64_t capacity() const override;
    int64_t length() const override;

    DataType dtype() const override;

    SharedCoreObj<IAllocator> alloc() const override;

    NVCVResourceType target() const override;

    void exportData(NVCVArrayData &data) const override;

    void resize(int64_t length) override;

private:
    SharedCoreObj<IAllocator> m_alloc;
    NVCVArrayRequirements     m_reqs;
    NVCVResourceType          m_target;
    NVCVArrayData             m_data;

    void *m_memBuffer;
};

} // namespace nvcv::priv

#endif // NVCV_CORE_PRIV_ARRAY_HPP
