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

#include "priv/TensorShape.hpp"

#include "priv/Status.hpp"
#include "priv/SymbolVersioning.hpp"

#include <nvcv/TensorData.h>

namespace priv = nvcv::priv;

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvTensorShapePermute,
                (NVCVTensorLayout srcLayout, const int64_t *srcShape, NVCVTensorLayout dstLayout, int64_t *dstShape))
{
    return priv::ProtectCall([&] { priv::PermuteShape(srcLayout, srcShape, dstLayout, dstShape); });
}
