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

#ifndef NVCV_PRIV_ARRAY_WRAPDATA_HPP
#define NVCV_PRIV_ARRAY_WRAPDATA_HPP

#include "IArray.hpp"

#include <cuda_runtime.h>

namespace nvcv::priv {

class ArrayWrapData final : public CoreObjectBase<IArray>
{
public:
    explicit ArrayWrapData(const NVCVArrayData &data, NVCVArrayDataCleanupFunc cleanup, void *ctxCleanup);
    ~ArrayWrapData();

    int32_t rank() const override;
    int64_t capacity() const override;
    int64_t length() const override;

    DataType dtype() const override;

    SharedCoreObj<IAllocator> alloc() const override;

    NVCVResourceType target() const override;

    void exportData(NVCVArrayData &data) const override;

private:
    NVCVArrayData    m_data;
    NVCVResourceType m_target;

    NVCVArrayDataCleanupFunc m_cleanup;
    void                    *m_ctxCleanup;
};

} // namespace nvcv::priv

#endif // NVCV_PRIV_ARRAY_WRAPDATA_HPP
