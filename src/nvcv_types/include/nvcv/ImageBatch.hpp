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

#ifndef NVCV_IMAGEBATCH_HPP
#define NVCV_IMAGEBATCH_HPP

#include "Casts.hpp"
#include "IImageBatch.hpp"
#include "ImageBatchData.hpp"

namespace nvcv {

// ImageBatch varshape definition -------------------------------------
class ImageBatchVarShape : public IImageBatchVarShape
{
public:
    using Requirements = NVCVImageBatchVarShapeRequirements;
    static Requirements CalcRequirements(int32_t capacity);

    explicit ImageBatchVarShape(const Requirements &reqs, IAllocator *alloc = nullptr);
    explicit ImageBatchVarShape(int32_t capacity, IAllocator *alloc = nullptr);
    ~ImageBatchVarShape();

    ImageBatchVarShape(const ImageBatchVarShape &) = delete;

private:
    NVCVImageBatchHandle doGetHandle() const final override;

    NVCVImageBatchHandle m_handle;
};

// For API backward-compatibility
using ImageBatchWrapHandle         = detail::WrapHandle<IImageBatch>;
using ImageBatchVarShapeWrapHandle = detail::WrapHandle<IImageBatchVarShape>;

} // namespace nvcv

#include "detail/ImageBatchImpl.hpp"

#endif // NVCV_IMAGEBATCH_HPP
