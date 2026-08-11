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

#include "RoundMode.hpp"

#include <nvcv/RoundMode.h>

namespace cvcudapy {

void ExportRoundMode(py::module &m)
{
    py::enum_<NVCVRoundMode>(m, "Round", "Rounding modes for data-type conversion to integer types.")
        .value("NEAREST", NVCV_ROUND_NEAREST, "Round to nearest, ties to even (default)")
        .value("TRUNCATE", NVCV_ROUND_TRUNCATE, "Truncate toward zero (drop the fractional part)");
}

} // namespace cvcudapy
