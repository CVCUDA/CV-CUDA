/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "ConnectivityType.hpp"

#include <cvcuda/Types.h>

namespace cvcudapy {

void ExportConnectivityType(py::module &m)
{
    py::enum_<NVCVConnectivityType>(
        m, "ConnectivityType", "Pixel/voxel connectivity types for connected-component labeling.", py::arithmetic())
        .value("CONNECTIVITY_4_2D", NVCV_CONNECTIVITY_4_2D, "4-connected: pixels sharing an edge in 2D.")
        .value("CONNECTIVITY_6_3D", NVCV_CONNECTIVITY_6_3D, "6-connected: voxels sharing a face in 3D.")
        .value("CONNECTIVITY_8_2D", NVCV_CONNECTIVITY_8_2D, "8-connected: pixels sharing an edge or corner in 2D.")
        .value("CONNECTIVITY_26_3D", NVCV_CONNECTIVITY_26_3D,
               "26-connected: voxels sharing a face, edge, or corner in 3D.")
        .export_values();
}

} // namespace cvcudapy
