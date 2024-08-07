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

#ifndef NVCV_CORE_PRIV_TENSORDATA_HPP
#define NVCV_CORE_PRIV_TENSORDATA_HPP

#include "IImage.hpp"
#include "ImageFormat.hpp"
#include "Size.hpp"

#include <nvcv/TensorData.h>

namespace nvcv::priv {

NVCVTensorLayout GetTensorLayoutFor(ImageFormat fmt, int nbatches);

void FillTensorData(IImage &img, NVCVTensorData &data);

void ReshapeTensorData(NVCVTensorData &tensor_data, int new_rank, const int64_t *new_shape,
                       NVCVTensorLayout new_layout);

} // namespace nvcv::priv

#endif // NVCV_CORE_PRIV_TENSORDATA_HPP
