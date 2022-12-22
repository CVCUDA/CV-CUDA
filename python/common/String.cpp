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

#include "String.hpp"

#include <cstdarg>
#include <cstdio>

namespace nvcvpy::util {

std::string FormatString(const char *fmt, ...)
{
    va_list va;
    va_start(va, fmt);

    char buffer[1024];
    vsnprintf(buffer, sizeof(buffer) - 1, fmt, va);
    buffer[sizeof(buffer) - 1] = '\0'; // better be safe against truncation

    va_end(va);

    return buffer;
}

} // namespace nvcvpy::util
