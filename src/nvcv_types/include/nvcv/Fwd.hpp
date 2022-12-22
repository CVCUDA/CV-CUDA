
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

/**
 * @file Fwd.hpp
 *
 * @brief Forward declaration of some public C++ interface entities.
 */

#ifndef NVCV_FWD_HPP
#define NVCV_FWD_HPP

#include "alloc/Fwd.hpp"

namespace nvcv {

class IImage;
class IImageData;
class IImageDataCudaArray;
class IImageDataStrided;
class IImageDataStridedCuda;
class IImageDataStridedHost;

class IImageBatch;
class IImageBatchData;

class IImageBatchVarShape;
class IImageBatchVarShapeData;
class IImageBatchVarShapeDataStrided;
class IImageBatchVarShapeDataStridedCuda;

class ITensor;
class ITensorData;
class ITensorDataStrided;
class ITensorDataStridedCuda;

} // namespace nvcv

#endif // NVCV_FWD_HPP
