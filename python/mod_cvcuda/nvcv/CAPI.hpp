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

#ifndef NVCV_PYTHON_PRIV_CAPI_HPP
#define NVCV_PYTHON_PRIV_CAPI_HPP

#include <pybind11/pybind11.h>

namespace nvcvpy::priv {

namespace py = pybind11;

void ExportCAPI(py::module &m);

// Registers the deterministic C API failure-injection toggles used by the
// ResourceGuard lifetime regressions. Bound under the private cvcuda._test
// submodule; not part of the public API.
void ExportCAPITestHooks(py::module &m);

} // namespace nvcvpy::priv

#endif // NVCV_PYTHON_PRIV_CAPI_HPP
