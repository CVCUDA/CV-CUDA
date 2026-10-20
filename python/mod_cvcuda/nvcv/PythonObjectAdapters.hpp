/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef NVCV_PYTHON_PRIV_PYTHON_OBJECT_ADAPTERS_HPP
#define NVCV_PYTHON_PRIV_PYTHON_OBJECT_ADAPTERS_HPP

#include <common/Binding.hpp>

namespace nvcvpy::priv {

// Test-only failure injection for the public ResourceGuard C-API callbacks.
struct ResourceGuardFailureInjection
{
    bool streamHoldResources     = false;
    bool resourcesSubmitSyncOnly = false;
    bool resourcesSyncAndHold    = false;
};

NB_EXPORT ResourceGuardFailureInjection &GetResourceGuardFailureInjection();
NB_EXPORT void                           ThrowIfResourceGuardFailureInjected(bool armed, const char *what);

} // namespace nvcvpy::priv

#endif // NVCV_PYTHON_PRIV_PYTHON_OBJECT_ADAPTERS_HPP
