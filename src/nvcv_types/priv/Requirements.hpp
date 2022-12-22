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

#ifndef NVCV_CORE_PRIV_REQUIREMENTS_HPP
#define NVCV_CORE_PRIV_REQUIREMENTS_HPP

#include <nvcv/alloc/Requirements.h>

namespace nvcv::priv {

void Init(NVCVRequirements &reqs);
void Add(NVCVRequirements &reqSum, const NVCVRequirements &req);

void AddBuffer(NVCVMemRequirements &memReq, int64_t bufSize, int64_t bufAlignment);

int64_t CalcTotalSizeBytes(const NVCVMemRequirements &memReq);

} // namespace nvcv::priv

#endif // NVCV_CORE_PRIV_REQUIREMENTS_HPP
