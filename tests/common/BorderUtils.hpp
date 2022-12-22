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

#ifndef NVCV_TEST_COMMON_BORDER_UTILS_HPP
#define NVCV_TEST_COMMON_BORDER_UTILS_HPP

#include <cuda_runtime.h> // for int2, etc.
#include <cvcuda/Types.h>

namespace nvcv::test {

void ReplicateBorderIndex(int2 &coord, int2 size);

void WrapBorderIndex(int2 &coord, int2 size);

void ReflectBorderIndex(int2 &coord, int2 size);

void Reflect101BorderIndex(int2 &coord, int2 size);

bool IsInside(int2 &inCoord, int2 inSize, NVCVBorderType borderMode);

} // namespace nvcv::test

#endif // NVCV_TEST_COMMON_BORDER_UTILS_HPP
