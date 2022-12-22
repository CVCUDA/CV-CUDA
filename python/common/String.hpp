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

#ifndef NVCV_PYTHON_STRING_HPP
#define NVCV_PYTHON_STRING_HPP

#include <sstream>
#include <string>

namespace nvcvpy::util {

std::string FormatString(const char *fmt, ...) __attribute__((format(printf, 1, 2)));

// Make it easier to use ostreams to define __repr__
template<class T>
std::string ToString(const T &v)
{
    std::ostringstream ss;
    ss << v;
    return ss.str();
}

} // namespace nvcvpy::util

#endif // NVCV_PYTHON_STRING_HPP
