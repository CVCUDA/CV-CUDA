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

#ifndef NVCV_TENSORSHAPEINFO_HPP
#define NVCV_TENSORSHAPEINFO_HPP

#include "Size.hpp"
#include "TensorLayoutInfo.hpp"
#include "TensorShape.hpp"

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

/**
 * @class TensorShapeInfoImpl
 * @brief This class provides detailed information about the shape of a tensor.
 *
 * The class is templated on the layout information type, which allows it to be adapted to various tensor layout schemes.
 * It provides functions to retrieve the shape, layout, and additional metadata about the tensor.
 *
 * @tparam LAYOUT_INFO The type that contains layout information for the tensor.
 */
template<typename LAYOUT_INFO>
class TensorShapeInfoImpl
{
public:
    /**
    * @brief Type alias for the layout information of the tensor.
    */
    using LayoutInfo = LAYOUT_INFO;

    /**
     * @brief A constructor that initializes a `TensorShapeInfoImpl` with the provided shape and layout information.
     *
     * @param shape The shape of the tensor.
     * @param infoLayout The layout information of the tensor.
     */
    TensorShapeInfoImpl(const TensorShape &shape, const LayoutInfo &infoLayout)
        : m_shape(shape)
        , m_infoLayout(infoLayout)
    {
        // idxSample
        int idx = m_infoLayout.idxSample();
        if (idx >= 0)
        {
            m_cacheNumSamples = m_shape[idx];
        }
        else if (m_shape.layout() != TENSOR_NONE)
        {
            m_cacheNumSamples = 1;
        }
        else
        {
            m_cacheNumSamples = 0;
        }
    }

    /**
     * @brief Returns the shape of the tensor.
     *
     * @return The shape of the tensor.
     */
    const TensorShape &shape() const
    {
        return m_shape;
    }

    /**
     * @brief Returns the layout of the tensor, this is a subset of LayoutInfo and is a convenience function.
     *
     * @return The layout of the tensor.
     */
    const TensorLayout &layout() const
    {
        return m_shape.layout();
    }

    /**
     * @brief Returns the layout and additional information of the tensor @see TensorLayoutInfo.
     *
     * @return The layout information of the tensor.
     */
    const LayoutInfo &infoLayout() const
    {
        return m_infoLayout;
    }

    /**
     * @brief Returns the number of samples in the tensor. If the tensor is not batched, the number of samples is 1. (i.e Nhwc)
     *
     * @return The number of samples in the tensor.
     */
    TensorShape::DimType numSamples() const
    {
        return m_cacheNumSamples;
    }

    /**
     * @brief Checks if the tensor is an image.
     *
     * @return True if the tensor is an image, false otherwise.
     */
    bool isImage() const
    {
        return m_infoLayout.isImage();
    }

protected:
    TensorShape m_shape;
    LayoutInfo  m_infoLayout;
    int         m_cacheNumSamples;
};

} // namespace detail

/**
 * @class TensorShapeInfo
 * @brief This class provides information about the shape of a tensor.
 *
 * It inherits from TensorShapeInfoImpl and is specialized for the base tensor layout type, providing functions to
 * retrieve the shape, layout, and whether the tensor is batched or corresponds to an image.
 */
class TensorShapeInfo : public detail::TensorShapeInfoImpl<TensorLayoutInfo>
{
    using Base = detail::TensorShapeInfoImpl<TensorLayoutInfo>;

public:
    /**
     * @brief Checks if the provided tensor shape is compatible with this class.
     *        In this case, all tensor shapes are considered compatible.
     *
     * @param tshape The tensor shape to check.
     *
     * @return Always true, as all tensor shapes are considered compatible.
     */
    static bool IsCompatible(const TensorShape &tshape)
    {
        (void)tshape;
        return true;
    }

    /**
     * @brief Creates a `TensorShapeInfo` object for the provided tensor shape.
     *
     * @param tshape The tensor shape to create the `TensorShapeInfo` for.
     *
     * @return A `TensorShapeInfo` object for the provided tensor shape.
     */
    static Optional<TensorShapeInfo> Create(const TensorShape &tshape)
    {
        return TensorShapeInfo(tshape);
    }

private:
    TensorShapeInfo(const TensorShape &tshape)
        : Base(tshape, *TensorLayoutInfo::Create(tshape.layout()))
    {
    }

    Optional<TensorLayoutInfo> m_infoLayout;
};

/**
 * @class TensorShapeInfoImage
 * @brief This class provides detailed information about the shape of an image tensor.
 *
 * It inherits from TensorShapeInfoImpl and is specialized for the image tensor layout type, offering additional
 * functionality tailored to image tensors, such as retrieving the number of channels, rows, columns, and the overall size.
 */
class TensorShapeInfoImage : public detail::TensorShapeInfoImpl<TensorLayoutInfoImage>
{
    using Base = detail::TensorShapeInfoImpl<TensorLayoutInfoImage>;

public:
    /**
     * @brief Checks if the provided tensor shape is compatible with this class.
     *        A tensor shape is considered compatible if both `TensorShapeInfo` and `TensorLayoutInfo` deem it compatible.
     *
     * @param tshape The tensor shape to check.
     *
     * @return True if the tensor shape is compatible, false otherwise.
     */
    static bool IsCompatible(const TensorShape &tshape)
    {
        return TensorShapeInfo::IsCompatible(tshape) && TensorLayoutInfo::IsCompatible(tshape.layout());
    }

    /**
     * @brief Creates a `TensorShapeInfoImage` object for the provided tensor shape if it is compatible.
     *
     * @param tshape The tensor shape to create the `TensorShapeInfoImage` for.
     *
     * @return A `TensorShapeInfoImage` object for the provided tensor shape if it is compatible, NullOpt otherwise.
     */
    static Optional<TensorShapeInfoImage> Create(const TensorShape &tshape)
    {
        if (IsCompatible(tshape))
        {
            return TensorShapeInfoImage(tshape);
        }
        else
        {
            return NullOpt;
        }
    }

    /**
     * @brief Returns the number of channels in the tensor. (i.e nhwC)
     *
     * @return The number of channels.
     */
    int32_t numChannels() const
    {
        return m_cacheNumChannels;
    }

    /**
     * @brief Returns the number of columns in the tensor. (i.e nhWc)
     *
     * @return The number of columns.
     */
    int32_t numCols() const
    {
        return m_cacheSize.w;
    }

    /**
     * @brief Returns the number of rows in the tensor. (i.e nHwc)
     *
     * @return The number of rows.
     */
    int32_t numRows() const
    {
        return m_cacheSize.h;
    }

    const Size2D &size() const
    {
        return m_cacheSize;
    }

protected:
    TensorShapeInfoImage(const TensorShape &tshape)
        : TensorShapeInfoImage(tshape, *TensorLayoutInfoImage::Create(tshape.layout()))
    {
    }

    TensorShapeInfoImage(const TensorShape &shape, const TensorLayoutInfoImage &infoLayout)
        : Base(shape, infoLayout)
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

    Size2D m_cacheSize;
    int    m_cacheNumChannels;
};

/**
 * @class TensorShapeInfoImagePlanar
 * @brief This class provides information about the shape of a planar image tensor.
 *
 * It inherits from TensorShapeInfoImage and is specialized for planar image tensors. The class provides functions to
 * check the compatibility of a given tensor shape and to retrieve the number of planes in the tensor.
 */
class TensorShapeInfoImagePlanar : public TensorShapeInfoImage
{
public:
    /**
     * @brief Checks if the provided tensor shape is compatible with this class.
     *        A tensor shape is considered compatible if it matches certain criteria related to the layout of the tensor.
     *
     * @param tshape The tensor shape to check.
     *
     * @return True if the tensor shape is compatible, false otherwise.
     */
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

    /**
     * @brief Creates a `TensorShapeInfoImagePlanar` object for the provided tensor shape if it is compatible.
     *
     * @param tshape The tensor shape to create the `TensorShapeInfoImagePlanar` for.
     *
     * @return A `TensorShapeInfoImagePlanar` object for the provided tensor shape if it is compatible, NullOpt otherwise.
     */
    static Optional<TensorShapeInfoImagePlanar> Create(const TensorShape &tshape)
    {
        if (IsCompatible(tshape))
        {
            return TensorShapeInfoImagePlanar(tshape);
        }
        else
        {
            return NullOpt;
        }
    }

    /**
     * @brief Returns the number of planes in the tensor. (i.e nCwh)
     *
     * @return The number of planes.
     */
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
