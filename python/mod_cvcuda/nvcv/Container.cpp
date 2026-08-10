/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "Container.hpp"

namespace nvcvpy::priv {

void Container::Export(py::module &m)
{
    py::class_<Container, std::shared_ptr<Container>, Resource, CacheItem> cont(m, "Container");

    py::class_<ExternalContainer, Container, std::shared_ptr<ExternalContainer>, Resource> extcont(
        nullptr, "ExternalContainer", py::module_local());
}

} // namespace nvcvpy::priv
