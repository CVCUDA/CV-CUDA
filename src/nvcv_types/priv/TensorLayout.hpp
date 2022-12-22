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

#ifndef NVCV_CORE_PRIV_TENSORLAYOUT_HPP
#define NVCV_CORE_PRIV_TENSORLAYOUT_HPP

#include <nvcv/TensorData.h>

namespace nvcv::priv {

NVCVTensorLayout CreateLayout(const char *descr);
NVCVTensorLayout CreateLayout(const char *beg, const char *end);

NVCVTensorLayout CreateFirst(const NVCVTensorLayout &layout, int n);
NVCVTensorLayout CreateLast(const NVCVTensorLayout &layout, int n);
NVCVTensorLayout CreateSubRange(const NVCVTensorLayout &layout, int beg, int end);

int FindDimIndex(const NVCVTensorLayout &layout, char dimLabel);

bool IsChannelLast(const NVCVTensorLayout &layout);

bool operator==(const NVCVTensorLayout &a, const NVCVTensorLayout &b);
bool operator!=(const NVCVTensorLayout &a, const NVCVTensorLayout &b);

} // namespace nvcv::priv

#endif // NVCV_CORE_PRIV_TENSORLAYOUT_HPP
