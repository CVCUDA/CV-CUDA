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

#ifndef CVCUDA_PRIV_INVERT_POLICY_HPP
#define CVCUDA_PRIV_INVERT_POLICY_HPP

#include <string_view>

namespace cvcuda::priv {

constexpr bool UsePackedU8C3VarShapeKernelForDevice(int sm, std::string_view deviceName)
{
    return sm != 89 || deviceName != "NVIDIA L4";
}

static_assert(!UsePackedU8C3VarShapeKernelForDevice(89, "NVIDIA L4"));
static_assert(UsePackedU8C3VarShapeKernelForDevice(89, "NVIDIA L40"));
static_assert(UsePackedU8C3VarShapeKernelForDevice(80, "NVIDIA A100-PCIE-40GB"));
static_assert(UsePackedU8C3VarShapeKernelForDevice(90, "NVIDIA H100 PCIe"));

} // namespace cvcuda::priv

#endif // CVCUDA_PRIV_INVERT_POLICY_HPP
