/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef NVCV_UTIL_SIZE_HPP
#define NVCV_UTIL_SIZE_HPP

namespace nvcv::util {

struct Size2D
{
    int w, h;
};

inline bool operator==(const Size2D &a, const Size2D &b)
{
    return a.w == b.w && a.h == b.h;
}

inline bool operator!=(const Size2D &a, const Size2D &b)
{
    return !(a == b);
}

} // namespace nvcv::util

#endif // NVCV_UTIL_SIZE_HPP
