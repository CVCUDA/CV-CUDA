/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "ThresholdType.hpp"

#include <cvcuda/Types.h>

namespace cvcudapy {

void ExportThresholdType(py::module &m)
{
    py::enum_<NVCVThresholdType>(m, "ThresholdType", py::arithmetic())
        .value("BINARY", NVCV_THRESH_BINARY, "Value above threshold is set to maxval, otherwise set to 0")
        .value("BINARY_INV", NVCV_THRESH_BINARY_INV, "Value above threshold is set to 0, otherwise set to maxval")
        .value("TRUNC", NVCV_THRESH_TRUNC, "Value above threshold is set to threshold, otherwise unchanged")
        .value("TOZERO", NVCV_THRESH_TOZERO, "Value above threshold is unchanged, otherwise set to 0")
        .value("TOZERO_INV", NVCV_THRESH_TOZERO_INV, "Value above threshold is set to 0, otherwise unchanged")
        .value("OTSU", NVCV_THRESH_OTSU, "Use Otsu's algorithm to automatically determine threshold")
        .value("TRIANGLE", NVCV_THRESH_TRIANGLE, "Use Triangle algorithm to automatically determine threshold");
}

} // namespace cvcudapy
