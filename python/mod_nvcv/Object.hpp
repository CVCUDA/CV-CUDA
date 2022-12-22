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

#ifndef NVCV_PYTHON_PRIV_OBJECT_HPP
#define NVCV_PYTHON_PRIV_OBJECT_HPP

#include <pybind11/pybind11.h>

#include <memory>

namespace nvcvpy::priv {

// Parent of all NVCV objects that are reference-counted
class PYBIND11_EXPORT Object : public std::enable_shared_from_this<Object>
{
public:
    virtual ~Object() = 0;

    Object(Object &&) = delete;

protected:
    Object() = default;
};

} // namespace nvcvpy::priv

#endif // NVCV_PYTHON_PRIV_OBJECT_HPP
