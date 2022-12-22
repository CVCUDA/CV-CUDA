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

#ifndef NVCV_TENSORDATAACESSOR_HPP
#define NVCV_TENSORDATAACESSOR_HPP

#include "ITensorData.hpp"
#include "TensorShapeInfo.hpp"
#include "detail/BaseFromMember.hpp"

#include <cstddef>

namespace nvcv {

// Design is similar to TensorShapeInfo hierarchy

namespace detail {

class TensorDataAccessStridedImpl
{
public:
    TensorDataAccessStridedImpl(const ITensorDataStrided &tdata, const TensorShapeInfoImpl &infoShape)
        : m_tdata(tdata)
        , m_infoShape(infoShape)
    {
    }

    TensorShape::DimType numSamples() const
    {
        return m_infoShape.numSamples();
    }

    DataType dtype() const
    {
        return m_tdata.dtype();
    }

    const TensorLayout &layout() const
    {
        return m_tdata.layout();
    }

    const TensorShape &shape() const
    {
        return m_tdata.shape();
    }

    int64_t sampleStride() const
    {
        int idx = this->infoLayout().idxSample();
        if (idx >= 0)
        {
            return m_tdata.stride(idx);
        }
        else
        {
            return 0;
        }
    }

    Byte *sampleData(int n) const
    {
        return sampleData(n, m_tdata.basePtr());
    }

    Byte *sampleData(int n, Byte *base) const
    {
        assert(0 <= n && n < this->numSamples());
        return base + this->sampleStride() * n;
    }

    bool isImage() const
    {
        return m_infoShape.isImage();
    }

    const TensorShapeInfoImpl &infoShape() const
    {
        return m_infoShape;
    }

    const TensorLayoutInfo &infoLayout() const
    {
        return m_infoShape.infoLayout();
    }

protected:
    const ITensorDataStrided &m_tdata;

    TensorDataAccessStridedImpl(const TensorDataAccessStridedImpl &that) = delete;

    TensorDataAccessStridedImpl(const TensorDataAccessStridedImpl &that, const TensorShapeInfoImpl &infoShape)
        : m_tdata(that.m_tdata)
        , m_infoShape(infoShape)
    {
    }

private:
    const TensorShapeInfoImpl &m_infoShape;
};

class TensorDataAccessStridedImageImpl : public TensorDataAccessStridedImpl
{
public:
    TensorDataAccessStridedImageImpl(const ITensorDataStrided &tdata, const TensorShapeInfoImageImpl &infoShape)
        : TensorDataAccessStridedImpl(tdata, infoShape)
    {
    }

    const TensorShapeInfoImageImpl &infoShape() const
    {
        return static_cast<const TensorShapeInfoImageImpl &>(TensorDataAccessStridedImpl::infoShape());
    }

    const TensorLayoutInfoImage &infoLayout() const
    {
        return this->infoShape().infoLayout();
    }

    int32_t numCols() const
    {
        return this->infoShape().numCols();
    }

    int32_t numRows() const
    {
        return this->infoShape().numRows();
    }

    int32_t numChannels() const
    {
        return this->infoShape().numChannels();
    }

    Size2D size() const
    {
        return this->infoShape().size();
    }

    int64_t chStride() const
    {
        int idx = this->infoLayout().idxChannel();
        if (idx >= 0)
        {
            return m_tdata.stride(idx);
        }
        else
        {
            return 0;
        }
    }

    int64_t colStride() const
    {
        int idx = this->infoLayout().idxWidth();
        if (idx >= 0)
        {
            return m_tdata.stride(idx);
        }
        else
        {
            return 0;
        }
    }

    int64_t rowStride() const
    {
        int idx = this->infoLayout().idxHeight();
        if (idx >= 0)
        {
            return m_tdata.stride(idx);
        }
        else
        {
            return 0;
        }
    }

    int64_t depthStride() const
    {
        int idx = this->infoLayout().idxDepth();
        if (idx >= 0)
        {
            return m_tdata.stride(idx);
        }
        else
        {
            return 0;
        }
    }

    Byte *rowData(int y) const
    {
        return rowData(y, m_tdata.basePtr());
    }

    Byte *rowData(int y, Byte *base) const
    {
        assert(0 <= y && y < this->numRows());
        return base + this->rowStride() * y;
    }

    Byte *chData(int c) const
    {
        return chData(c, m_tdata.basePtr());
    }

    Byte *chData(int c, Byte *base) const
    {
        assert(0 <= c && c < this->numChannels());
        return base + this->chStride() * c;
    }

protected:
    TensorDataAccessStridedImageImpl(const TensorDataAccessStridedImageImpl &that) = delete;

    TensorDataAccessStridedImageImpl(const TensorDataAccessStridedImageImpl &that,
                                     const TensorShapeInfoImageImpl         &infoShape)
        : TensorDataAccessStridedImpl(that, infoShape)
    {
    }
};

class TensorDataAccessStridedImagePlanarImpl : public TensorDataAccessStridedImageImpl
{
public:
    TensorDataAccessStridedImagePlanarImpl(const ITensorDataStrided &tdata, const TensorShapeInfoImagePlanar &infoShape)
        : TensorDataAccessStridedImageImpl(tdata, infoShape)
    {
    }

    const TensorShapeInfoImagePlanar &infoShape() const
    {
        return static_cast<const TensorShapeInfoImagePlanar &>(TensorDataAccessStridedImageImpl::infoShape());
    }

    int32_t numPlanes() const
    {
        return this->infoShape().numPlanes();
    }

    int64_t planeStride() const
    {
        if (this->infoLayout().isChannelFirst())
        {
            int ichannel = this->infoLayout().idxChannel();
            assert(ichannel >= 0);
            return m_tdata.stride(ichannel);
        }
        else
        {
            return 0;
        }
    }

    Byte *planeData(int p) const
    {
        return planeData(p, m_tdata.basePtr());
    }

    Byte *planeData(int p, Byte *base) const
    {
        assert(0 <= p && p < this->numPlanes());
        return base + this->planeStride() * p;
    }

protected:
    TensorDataAccessStridedImagePlanarImpl(const TensorDataAccessStridedImagePlanarImpl &that) = delete;

    TensorDataAccessStridedImagePlanarImpl(const TensorDataAccessStridedImagePlanarImpl &that,
                                           const TensorShapeInfoImagePlanar             &infoShape)
        : TensorDataAccessStridedImageImpl(that, infoShape)
    {
    }
};

} // namespace detail

class TensorDataAccessStrided
    // declaration order is important here
    : private detail::BaseFromMember<TensorShapeInfo>
    , public detail::TensorDataAccessStridedImpl
{
public:
    static bool IsCompatible(const ITensorData &data)
    {
        return dynamic_cast<const ITensorDataStrided *>(&data) != nullptr;
    }

    static detail::Optional<TensorDataAccessStrided> Create(const ITensorData &data)
    {
        if (auto *dataStrided = dynamic_cast<const ITensorDataStrided *>(&data))
        {
            return TensorDataAccessStrided(*dataStrided);
        }
        else
        {
            return detail::NullOpt;
        }
    }

    TensorDataAccessStrided(const TensorDataAccessStrided &that)
        : MemberShapeInfo(that)
        , detail::TensorDataAccessStridedImpl(that, MemberShapeInfo::member)
    {
    }

private:
    using MemberShapeInfo = detail::BaseFromMember<TensorShapeInfo>;

    TensorDataAccessStrided(const ITensorDataStrided &data)
        : MemberShapeInfo{*TensorShapeInfo::Create(data.shape())}
        , detail::TensorDataAccessStridedImpl(data, MemberShapeInfo::member)
    {
    }
};

class TensorDataAccessStridedImage
    // declaration order is important here
    : private detail::BaseFromMember<TensorShapeInfoImage>
    , public detail::TensorDataAccessStridedImageImpl
{
public:
    TensorDataAccessStridedImage(const TensorDataAccessStridedImage &that)
        : MemberShapeInfo(that)
        , detail::TensorDataAccessStridedImageImpl(that, MemberShapeInfo::member)
    {
    }

    static bool IsCompatible(const ITensorData &data)
    {
        return TensorDataAccessStrided::IsCompatible(data) && TensorShapeInfoImage::IsCompatible(data.shape());
    }

    static detail::Optional<TensorDataAccessStridedImage> Create(const ITensorData &data)
    {
        if (IsCompatible(data))
        {
            return TensorDataAccessStridedImage(dynamic_cast<const ITensorDataStrided &>(data));
        }
        else
        {
            return detail::NullOpt;
        }
    }

private:
    using MemberShapeInfo = detail::BaseFromMember<TensorShapeInfoImage>;

protected:
    TensorDataAccessStridedImage(const ITensorDataStrided &data)
        : MemberShapeInfo{*TensorShapeInfoImage::Create(data.shape())}
        , detail::TensorDataAccessStridedImageImpl(data, MemberShapeInfo::member)
    {
    }
};

class TensorDataAccessStridedImagePlanar
    // declaration order is important here
    : private detail::BaseFromMember<TensorShapeInfoImagePlanar>
    , public detail::TensorDataAccessStridedImagePlanarImpl
{
public:
    TensorDataAccessStridedImagePlanar(const TensorDataAccessStridedImagePlanar &that)
        : MemberShapeInfo(that)
        , detail::TensorDataAccessStridedImagePlanarImpl(that, MemberShapeInfo::member)
    {
    }

    static bool IsCompatible(const ITensorData &data)
    {
        return TensorDataAccessStridedImage::IsCompatible(data)
            && TensorShapeInfoImagePlanar::IsCompatible(data.shape());
    }

    static detail::Optional<TensorDataAccessStridedImagePlanar> Create(const ITensorData &data)
    {
        if (IsCompatible(data))
        {
            return TensorDataAccessStridedImagePlanar(dynamic_cast<const ITensorDataStrided &>(data));
        }
        else
        {
            return detail::NullOpt;
        }
    }

private:
    using MemberShapeInfo = detail::BaseFromMember<TensorShapeInfoImagePlanar>;

protected:
    TensorDataAccessStridedImagePlanar(const ITensorDataStrided &data)
        : MemberShapeInfo{*TensorShapeInfoImagePlanar::Create(data.shape())}
        , detail::TensorDataAccessStridedImagePlanarImpl(data, MemberShapeInfo::member)
    {
    }
};

} // namespace nvcv

#endif // NVCV_TENSORDATAACESSOR_HPP
