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

#include "MorphologyType.hpp"

#include <cvcuda/Types.h>

namespace cvcudapy {

void ExportMorphologyType(py::module &m)
{
    py::enum_<NVCVMorphologyType>(m, "MorphologyType", "Morphological operation types.")
        .value("ERODE", NVCV_ERODE, "Replaces each pixel with the minimum over the structuring element.")
        .value("DILATE", NVCV_DILATE, "Replaces each pixel with the maximum over the structuring element.")
        .value("OPEN", NVCV_OPEN, "Erosion followed by dilation; removes small bright regions.")
        .value("CLOSE", NVCV_CLOSE, "Dilation followed by erosion; fills small dark holes.");
}

} // namespace cvcudapy
