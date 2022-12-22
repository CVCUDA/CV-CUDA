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

#ifndef NVCV_CORE_PRIV_IMAGE_HPP
#define NVCV_CORE_PRIV_IMAGE_HPP

#include "IImage.hpp"

namespace nvcv::priv {

class Image final : public CoreObjectBase<IImage>
{
public:
    explicit Image(NVCVImageRequirements reqs, IAllocator &alloc);
    ~Image();

    static NVCVImageRequirements CalcRequirements(Size2D size, ImageFormat fmt, int32_t baseAlign, int32_t rowAlign);

    Size2D        size() const override;
    ImageFormat   format() const override;
    IAllocator   &alloc() const override;
    NVCVTypeImage type() const override;

    void exportData(NVCVImageData &data) const override;

private:
    IAllocator           &m_alloc;
    NVCVImageRequirements m_reqs;
    void                 *m_memBuffer;
};

class ImageWrapData final : public CoreObjectBase<IImage>
{
public:
    explicit ImageWrapData(const NVCVImageData &data, NVCVImageDataCleanupFunc cleanup, void *ctxCleanup);

    ~ImageWrapData();

    Size2D        size() const override;
    ImageFormat   format() const override;
    IAllocator   &alloc() const override;
    NVCVTypeImage type() const override;

    void exportData(NVCVImageData &data) const override;

private:
    NVCVImageData m_data;

    NVCVImageDataCleanupFunc m_cleanup;
    void                    *m_ctxCleanup;

    void doCleanup() noexcept;

    void doValidateData(const NVCVImageData &data) const;
};

} // namespace nvcv::priv

#endif // NVCV_CORE_PRIV_IMAGE_HPP
