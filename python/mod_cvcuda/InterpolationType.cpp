/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "InterpolationType.hpp"

#include <cvcuda/Types.h>

namespace cvcudapy {

void ExportInterpolationType(py::module &m)
{
    py::enum_<NVCVInterpolationType>(m, "Interp")
        .value("NEAREST", NVCV_INTERP_NEAREST, "Nearest-neighbor interpolation")
        .value("LINEAR", NVCV_INTERP_LINEAR, "Linear interpolation")
        .value("CUBIC", NVCV_INTERP_CUBIC, "Cubic interpolation")
        .value("AREA", NVCV_INTERP_AREA, "Area-based (resampling using pixels in area) interpolation")
        .value("LANCZOS", NVCV_INTERP_LANCZOS, "Lanczos interpolation")
        .value("WARP_INVERSE_MAP", NVCV_WARP_INVERSE_MAP, "Inverse transformation")
        .value("GAUSSIAN", NVCV_INTERP_GAUSSIAN, "Gaussian interpolation")
        .value("HAMMING", NVCV_INTERP_HAMMING, "Hamming interpolation")
        .value("BOX", NVCV_INTERP_BOX, "Box interpolation")
        .def("__or__", [](NVCVInterpolationType e1, NVCVInterpolationType e2) { return int(e1) | int(e2); });
}

} // namespace cvcudapy
