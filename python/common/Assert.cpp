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

#include "Assert.hpp"

#include <cstdio>
#include <cstdlib>

namespace nvcvpy::util {

void DoAssert(const char *file, int line, const char *cond)
{
#if NVCV_EXPOSE_CODE
    fprintf(stderr, "Fatal assertion error on %s:%d: %s\n", file, line, cond);
#else
    (void)file;
    (void)line;
    (void)cond;
    fprintf(stderr, "Fatal assertion error\n");
#endif
    abort();
}

} // namespace nvcvpy::util
