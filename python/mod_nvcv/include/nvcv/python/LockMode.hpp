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

#ifndef NVCV_PYTHON_LOCKMODE_HPP
#define NVCV_PYTHON_LOCKMODE_HPP

namespace nvcvpy {

enum LockMode : uint8_t
{
    LOCK_NONE      = 0,
    LOCK_READ      = 1,
    LOCK_WRITE     = 2,
    LOCK_READWRITE = LOCK_READ | LOCK_WRITE
};

} // namespace nvcvpy

#endif // NVCV_PYTHON_PRIV_LOCKMODE_HPP
