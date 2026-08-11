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

#include "Definitions.hpp"

#include <cvcuda/priv/InvertPolicy.hpp>

namespace priv = cvcuda::priv;

TEST(OpInvertPolicy, UsesScalarU8C3VarShapeKernelOnL4)
{
    EXPECT_FALSE(priv::UsePackedU8C3VarShapeKernelForDevice(89, "NVIDIA L4"));
}

TEST(OpInvertPolicy, KeepsPackedU8C3VarShapeKernelOnOtherDevices)
{
    EXPECT_TRUE(priv::UsePackedU8C3VarShapeKernelForDevice(89, "NVIDIA L40"));
    EXPECT_TRUE(priv::UsePackedU8C3VarShapeKernelForDevice(89, "NVIDIA L40S"));
    EXPECT_TRUE(priv::UsePackedU8C3VarShapeKernelForDevice(80, "NVIDIA A100-PCIE-40GB"));
    EXPECT_TRUE(priv::UsePackedU8C3VarShapeKernelForDevice(90, "NVIDIA H100 PCIe"));
    EXPECT_TRUE(priv::UsePackedU8C3VarShapeKernelForDevice(90, "NVIDIA L4"));
}
