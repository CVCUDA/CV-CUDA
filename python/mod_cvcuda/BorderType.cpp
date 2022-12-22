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

#include "BorderType.hpp"

#include <cvcuda/Types.h>

namespace cvcudapy {

void ExportBorderType(py::module &m)
{
    py::enum_<NVCVBorderType>(m, "Border")
        .value("CONSTANT", NVCV_BORDER_CONSTANT)
        .value("REPLICATE", NVCV_BORDER_REPLICATE)
        .value("REFLECT", NVCV_BORDER_REFLECT)
        .value("WRAP", NVCV_BORDER_WRAP)
        .value("REFLECT101", NVCV_BORDER_REFLECT101);
}

} // namespace cvcudapy
