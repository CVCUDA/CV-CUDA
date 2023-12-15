/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "NormType.hpp"

#include <cvcuda/Types.h>

namespace cvcudapy {

void ExportNormType(py::module &m)
{
    py::enum_<NVCVNormType>(m, "Norm", py::arithmetic())
        .value("HAMMING", NVCV_NORM_HAMMING, "Hamming distance")
        .value("L1", NVCV_NORM_L1, "Manhattan distance")
        .value("L2", NVCV_NORM_L2, "Euclidean distance");
}

} // namespace cvcudapy
