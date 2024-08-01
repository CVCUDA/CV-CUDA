
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

/**
 * @file Fwd.hpp
 *
 * @brief Forward declaration of some public C++ interface entities.
 */

#ifndef NVCV_FWD_HPP
#define NVCV_FWD_HPP

#include "alloc/Fwd.hpp"

namespace nvcv {

class Image;
class ImageData;
class ImageDataCudaArray;
class ImageDataStrided;
class ImageDataStridedCuda;
class ImageDataStridedHost;

class ImageBatch;
class ImageBatchData;

class ImageBatchVarShape;
class ImageBatchVarShapeData;
class ImageBatchVarShapeDataStrided;
class ImageBatchVarShapeDataStridedCuda;

class Tensor;
class TensorData;
class TensorDataStrided;
class TensorDataStridedCuda;

class Array;
class ArrayData;
class ArrayDataCuda;
class ArrayDataHost;
class ArrayDataHostPinned;

} // namespace nvcv

#endif // NVCV_FWD_HPP
