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

#include "ColorSpec.hpp"

#include <nvcv/ColorSpec.h>

namespace nvcvpy::priv {

void ExportColorSpec(py::module_ &m)
{
    py::enum_compat<NVCVColorSpec>(m, "ColorSpec")
        .value("BT601", NVCV_COLOR_SPEC_BT601, "Color spec defining ITU-R BT.601 standard.")
        .value("BT601_ER", NVCV_COLOR_SPEC_BT601_ER, "Color spec defining full-range ITU-R BT.601.")
        .value("BT709", NVCV_COLOR_SPEC_BT709, "Color spec defining ITU-R BT.709 standard.")
        .value("BT709_ER", NVCV_COLOR_SPEC_BT709_ER, "Color spec defining full-range ITU-R BT.709.")
        .value("BT2020", NVCV_COLOR_SPEC_BT2020, "Color spec defining ITU-R BT.2020 standard.")
        .value("BT2020_ER", NVCV_COLOR_SPEC_BT2020_ER, "Color spec defining full-range ITU-R BT.2020.");
}

} // namespace  nvcvpy::priv
