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

#ifndef NVCV_CORE_PRIV_IIMAGE_HPP
#define NVCV_CORE_PRIV_IIMAGE_HPP

#include "ICoreObject.hpp"
#include "ImageFormat.hpp"

#include <nvcv/Image.h>

namespace nvcv::priv {

class IAllocator;

class IImage : public ICoreObjectHandle<IImage, NVCVImageHandle>
{
public:
    virtual Size2D      size() const   = 0;
    virtual ImageFormat format() const = 0;

    virtual NVCVTypeImage type() const = 0;

    virtual IAllocator &alloc() const = 0;

    virtual void exportData(NVCVImageData &data) const = 0;
};

} // namespace nvcv::priv

#endif // NVCV_CORE_PRIV_IIMAGE_HPP
