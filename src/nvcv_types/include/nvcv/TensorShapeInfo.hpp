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

#ifndef NVCV_TENSORSHAPEINFO_HPP
#define NVCV_TENSORSHAPEINFO_HPP

#include "Size.hpp"
#include "TensorLayoutInfo.hpp"
#include "TensorShape.hpp"
#include "detail/BaseFromMember.hpp"

#include <cassert>

namespace nvcv {

// The criteria followed for the design of the TensorShapeInfo hierarchy is as follows:
// - no virtual dispatches
// - no heap allocation
// - minimal memory footprint
// - fast!
//
// To achieve that, the tensor layout info class for each TensorShapeInfo child
// had to be passed to the parent's constructor, instead of using virtual
// method. This allows for the parent ctor to also use the layout info, which
// wouldn't be the case if using virtual method to get it.
// Care was taken to handle object copies, as the parent of the new class must
// use the tensor layout info object of the new object, not the old.

namespace detail {

class TensorShapeInfoImpl
{
public:
    TensorShapeInfoImpl(const TensorShape &shape, const TensorLayoutInfo &infoLayout)
        : m_shape(shape)
        , m_infoLayout(infoLayout)
    {
        // idxSample
        int idx = m_infoLayout.idxSample();
        if (idx >= 0)
        {
            m_cacheNumSamples = m_shape[idx];
        }
        else if (m_shape.layout() != TensorLayout::NONE)
        {
            m_cacheNumSamples = 1;
        }
        else
        {
            m_cacheNumSamples = 0;
        }
    }

    const TensorShape &shape() const
    {
        return m_shape;
    }

    const TensorLayout &layout() const
    {
        return m_shape.layout();
    }

    const TensorLayoutInfo &infoLayout() const
    {
        return m_infoLayout;
    }

    TensorShape::DimType numSamples() const
    {
        return m_cacheNumSamples;
    }

    bool isImage() const
    {
        return m_infoLayout.isImage();
    }

protected:
    const TensorShape &m_shape;
    int                m_cacheNumSamples;

    TensorShapeInfoImpl(const TensorShapeInfoImpl &that) = delete;

    TensorShapeInfoImpl(const TensorShapeInfoImpl &that, const TensorLayoutInfo &infoLayout)
        : m_shape(that.m_shape)
        , m_cacheNumSamples(that.m_cacheNumSamples)
        , m_infoLayout(infoLayout)
    {
    }

private:
    const TensorLayoutInfo &m_infoLayout;
};

class TensorShapeInfoImageImpl : public TensorShapeInfoImpl
{
public:
    TensorShapeInfoImageImpl(const TensorShape &shape, const TensorLayoutInfoImage &infoLayout)
        : TensorShapeInfoImpl(shape, infoLayout)
    {
        // idxChannel
        int idx = this->infoLayout().idxChannel();
        if (idx >= 0)
        {
            m_cacheNumChannels = m_shape[idx];
        }
        else
        {
            m_cacheNumChannels = 1;
        }

        // idxWidth
        idx = this->infoLayout().idxWidth();
        if (idx < 0)
        {
            throw Exception(Status::ERROR_INVALID_ARGUMENT, "Image shape must have a Width dimension");
        }
        m_cacheSize.w = m_shape[idx];

        // idxHeight
        idx = this->infoLayout().idxHeight();
        if (idx >= 0)
        {
            m_cacheSize.h = m_shape[idx];
        }
        else
        {
            m_cacheSize.h = 1;
        }
    }

    const TensorLayoutInfoImage &infoLayout() const
    {
        return static_cast<const TensorLayoutInfoImage &>(TensorShapeInfoImpl::infoLayout());
    }

    int32_t numChannels() const
    {
        return m_cacheNumChannels;
    }

    int32_t numCols() const
    {
        return m_cacheSize.w;
    }

    int32_t numRows() const
    {
        return m_cacheSize.h;
    }

    const Size2D &size() const
    {
        return m_cacheSize;
    }

protected:
    TensorShapeInfoImageImpl(const TensorShapeInfoImageImpl &that) = delete;

    TensorShapeInfoImageImpl(const TensorShapeInfoImageImpl &that, const TensorLayoutInfo &infoLayout)
        : TensorShapeInfoImpl(that, infoLayout)
        , m_cacheSize(that.m_cacheSize)
        , m_cacheNumChannels(that.m_cacheNumChannels)
    {
    }

private:
    Size2D m_cacheSize;
    int    m_cacheNumChannels;
};

} // namespace detail

class TensorShapeInfo
    // declaration order is important here
    : private detail::BaseFromMember<TensorLayoutInfo>
    , public detail::TensorShapeInfoImpl
{
public:
    static bool IsCompatible(const TensorShape &tshape)
    {
        return true;
    }

    static detail::Optional<TensorShapeInfo> Create(const TensorShape &tshape)
    {
        return TensorShapeInfo(tshape);
    }

    TensorShapeInfo(const TensorShapeInfo &that)
        : MemberLayoutInfo(that)
        , detail::TensorShapeInfoImpl(that, MemberLayoutInfo::member)
    {
    }

private:
    using MemberLayoutInfo = detail::BaseFromMember<TensorLayoutInfo>;

    TensorShapeInfo(const TensorShape &tshape)
        : MemberLayoutInfo{*TensorLayoutInfo::Create(tshape.layout())}
        , detail::TensorShapeInfoImpl(tshape, MemberLayoutInfo::member)
    {
    }

    detail::Optional<TensorLayoutInfo> m_infoLayout;
};

class TensorShapeInfoImage
    // declaration order is important here
    : private detail::BaseFromMember<TensorLayoutInfoImage>
    , public detail::TensorShapeInfoImageImpl
{
public:
    TensorShapeInfoImage(const TensorShapeInfoImage &that)
        : MemberLayoutInfo(that)
        , detail::TensorShapeInfoImageImpl(that, MemberLayoutInfo::member)
    {
    }

    static bool IsCompatible(const TensorShape &tshape)
    {
        return TensorShapeInfo::IsCompatible(tshape) && TensorLayoutInfo::IsCompatible(tshape.layout());
    }

    static detail::Optional<TensorShapeInfoImage> Create(const TensorShape &tshape)
    {
        if (IsCompatible(tshape))
        {
            return TensorShapeInfoImage(tshape);
        }
        else
        {
            return detail::NullOpt;
        }
    }

private:
    using MemberLayoutInfo = detail::BaseFromMember<TensorLayoutInfoImage>;

protected:
    TensorShapeInfoImage(const TensorShape &tshape)
        : MemberLayoutInfo{*TensorLayoutInfoImage::Create(tshape.layout())}
        , detail::TensorShapeInfoImageImpl(tshape, MemberLayoutInfo::member)
    {
    }
};

class TensorShapeInfoImagePlanar : public TensorShapeInfoImage
{
public:
    static bool IsCompatible(const TensorShape &tshape)
    {
        if (auto infoLayout = TensorLayoutInfoImage::Create(tshape.layout()))
        {
            const TensorLayout &layout = tshape.layout();

            if (infoLayout->isRowMajor() && (infoLayout->isChannelFirst() || infoLayout->isChannelLast()))
            {
                int iheight = infoLayout->idxHeight();
                // Has explicit height?
                if (iheight >= 0)
                {
                    assert(iheight + 1 < layout.rank());
                    // *HWC, [^C]*HW, *CHW
                    return layout[iheight + 1] == LABEL_WIDTH
                        && (iheight == 0 || infoLayout->isChannelLast() || layout[iheight - 1] == LABEL_CHANNEL);
                }
                else
                {
                    int ichannel = infoLayout->idxChannel();

                    // [^HC]*W, [^H]*CW, [^H]*WC
                    return ichannel == -1 || ichannel >= layout.rank() - 2;
                }
            }
        }
        return false;
    }

    static detail::Optional<TensorShapeInfoImagePlanar> Create(const TensorShape &tshape)
    {
        if (IsCompatible(tshape))
        {
            return TensorShapeInfoImagePlanar(tshape);
        }
        else
        {
            return detail::NullOpt;
        }
    }

    int32_t numPlanes() const
    {
        return m_cacheNumPlanes;
    }

private:
    int m_cacheNumPlanes;

    TensorShapeInfoImagePlanar(const TensorShape &tshape)
        : TensorShapeInfoImage(tshape)
    {
        // numPlanes
        if (this->infoLayout().isChannelLast())
        {
            m_cacheNumPlanes = 1;
        }
        else
        {
            int ichannel = this->infoLayout().idxChannel();
            if (ichannel >= 0)
            {
                m_cacheNumPlanes = m_shape[ichannel];
            }
            else
            {
                m_cacheNumPlanes = 1;
            }
        }
    }
};

} // namespace nvcv

#endif // NVCV_TENSORSHAPEINFO_HPP
