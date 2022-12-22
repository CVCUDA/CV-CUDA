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

#ifndef NVCV_FORMAT_PRIV_COLORFORMAT_HPP
#define NVCV_FORMAT_PRIV_COLORFORMAT_HPP

#include <nvcv/ColorSpec.h>

namespace nvcv::priv {

struct ColorFormat
{
    NVCVColorModel model;

    union
    {
        NVCVColorSpec  cspec;
        NVCVRawPattern raw;
    };
};

bool operator==(const ColorFormat &a, const ColorFormat &b);
bool operator!=(const ColorFormat &a, const ColorFormat &b);

} // namespace nvcv::priv

#endif // NVCV_FORMAT_PRIV_COLORFORMAT_HPP
