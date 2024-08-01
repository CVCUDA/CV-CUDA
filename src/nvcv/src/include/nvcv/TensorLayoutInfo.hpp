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

#ifndef NVCV_TENSOR_LAYOUT_INFO_HPP
#define NVCV_TENSOR_LAYOUT_INFO_HPP

#include "Optional.hpp"
#include "TensorLayout.hpp"

namespace nvcv {

/**
 * @class TensorLayoutInfo
 * @brief Provides information and utility functions related to tensor layouts.
 *
 * The TensorLayoutInfo class provides a series of utility functions to
 * inspect and work with tensor layouts. It allows checking the compatibility
 * of a given layout, creating instances from a layout, and querying specific
 * properties of the layout, such as whether it represents a batch or an image.
 */
class TensorLayoutInfo
{
public:
    /**
     * @brief Check if the given layout is compatible.
     *
     * For this base class, all layouts are considered compatible.
     *
     * @param layout The layout to check for compatibility.
     * @return true since all layouts are compatible.
     */
    static bool IsCompatible(const TensorLayout &)
    {
        return true;
    }

    /**
     * @brief Create a TensorLayoutInfo object from the given layout.
     *
     * @param layout The layout to use for creating the TensorLayoutInfo object.
     * @return A TensorLayoutInfo object constructed with the given layout.
     */
    static Optional<TensorLayoutInfo> Create(const TensorLayout &layout)
    {
        return TensorLayoutInfo{layout};
    }

    /**
     * @brief Get the layout of the tensor.
     *
     * @return The tensor's layout.
     */
    constexpr const TensorLayout &layout() const

    {
        return m_layout;
    }

    /**
     * @brief Check if the layout includes a batch dimension.
     *
     * @return true if the layout includes a batch dimension, false otherwise.
     */
    constexpr bool isBatch() const
    {
        return m_cacheIsBatch;
    }

    /**
     * @brief Get the index of the sample dimension in the layout.
     *
     * @return The index of the sample dimension, or -1 if there is no sample dimension.
     */
    int idxSample() const
    {
        return m_cacheIdxSample;
    }

    /**
     * @brief Check if the layout corresponds to an image.
     *
     * @return true if the layout corresponds to an image, false otherwise.
     */
    bool isImage() const
    {
        return m_cacheIsImage;
    }

protected:
    TensorLayoutInfo(const TensorLayout &layout)
        : m_layout(layout)
    {
        // isBatch ----------------
        m_cacheIsBatch = m_layout.rank() > 0 && m_layout[0] == LABEL_BATCH;

        // isImage ----------------
        if (m_layout != TENSOR_NONE)
        {
            m_cacheIsImage = m_layout.find(LABEL_WIDTH) >= 0;
        }
        else
        {
            m_cacheIsImage = false;
        }

        // idxSample ----------------
        m_cacheIdxSample = m_cacheIsBatch ? 0 : -1;
    }

private:
    TensorLayout m_layout;
    bool         m_cacheIsBatch;
    bool         m_cacheIsImage;
    int          m_cacheIdxSample;
};

/**
 * @class TensorLayoutInfoImage
 * @brief This class provides more information about tensor layout for image tensors.
 *
 * The class inherits from TensorLayoutInfo and adds functions specific to image tensors.
 * It provides detailed information about the tensor layout such as the number of spatial dimensions,
 * the index of various dimensions (channel, width, height, depth), and whether the layout is row-major.
 * It also provides functions to check whether the channel is in the first or last position.
 */
class TensorLayoutInfoImage : public TensorLayoutInfo
{
public:
    /**
     * @brief Check if the given layout is compatible with the image tensor.
     *
     * @param layout The layout to check for compatibility.
     * @return true if the layout is compatible with an image tensor, false otherwise.
     */
    static bool IsCompatible(const TensorLayout &layout)
    {
        if (auto info = TensorLayoutInfo::Create(layout))
        {
            return info->isImage();
        }
        else
        {
            return false;
        }
    }

    /**
     * @brief Create a TensorLayoutInfoImage object if the provided layout is compatible.
     *
     * @param layout The layout to use for creating the TensorLayoutInfoImage object.
     * @return An optional TensorLayoutInfoImage object. The object is valid if the layout is compatible.
     */
    static Optional<TensorLayoutInfoImage> Create(const TensorLayout &layout)
    {
        if (IsCompatible(layout))
        {
            return TensorLayoutInfoImage{layout};
        }
        else
        {
            return NullOpt;
        }
    }

    /**
     * @brief Retrieves the number of spatial dimensions in the tensor layout.
     * @return Number of spatial dimensions.
     */
    int numSpatialDims() const
    {
        return m_cacheNumSpatialDims;
    }

    /**
     * @brief Checks if the tensor layout is row-major.
     * @return true if row-major, false otherwise.
     */
    bool isRowMajor() const
    {
        return m_cacheIsRowMajor;
    }

    /**
     * @brief Retrieves the index of the channel in the tensor layout.
     * @return Index of the channel or -1 if not found.
     */
    int idxChannel() const
    {
        return m_cacheIdxChannel;
    }

    /**
     * @brief Retrieves the width index in the tensor layout.
     * @return Width index or -1 if not found.
     */
    int idxWidth() const
    {
        return m_cacheIdxWidth;
    }

    /**
     * @brief Retrieves the height index in the tensor layout.
     * @return Height index or -1 if not found.
     */
    int idxHeight() const
    {
        return m_cacheIdxHeight;
    }

    /**
     * @brief Retrieves the depth index in the tensor layout.
     * @return Depth index or -1 if not found.
     */
    int idxDepth() const
    {
        return m_cacheIdxDepth;
    }

    /**
     * @brief Checks if the tensor layout contains a channel.
     * @return true if there's a channel, false otherwise.
     */
    bool hasChannel() const
    {
        return m_cacheHasChannel;
    }

    /**
     * @brief Checks if the channel appears first in the tensor layout (i.e CHW).
     * @return true if channel is first, false otherwise.
     */
    bool isChannelFirst() const
    {
        return m_cacheIsChannelFirst;
    }

    /**
     * @brief Checks if the channel appears last in the tensor layout (i.e. HWC).
     * @return true if channel is last, false otherwise.
     */
    bool isChannelLast() const
    {
        return m_cacheIsChannelLast;
    }

protected:
    TensorLayoutInfoImage(const TensorLayout &layout)
        : TensorLayoutInfo(layout)
    {
        m_cacheNumSpatialDims = std::count_if(layout.begin(), layout.end(),
                                              [](char v)
                                              {
                                                  switch (v)
                                                  {
                                                  case LABEL_WIDTH:
                                                  case LABEL_HEIGHT:
                                                  case LABEL_DEPTH:
                                                      return true;
                                                  default:
                                                      return false;
                                                  }
                                              });

        m_cacheIsRowMajor = layout.endsWith(TENSOR_W) || layout.endsWith(TENSOR_WC);
        m_cacheIdxChannel = layout.find(LABEL_CHANNEL);
        m_cacheIdxWidth   = layout.find(LABEL_WIDTH);
        m_cacheIdxHeight  = layout.find(LABEL_HEIGHT);
        m_cacheIdxDepth   = layout.find(LABEL_DEPTH);
        m_cacheHasChannel = m_cacheIdxChannel >= 0;

        // isChannelFirst --------------
        if (layout != TENSOR_NONE)
        {
            if (this->isBatch())
            {
                m_cacheIsChannelFirst = layout[1] == LABEL_CHANNEL;
            }
            else
            {
                m_cacheIsChannelFirst = layout[0] == LABEL_CHANNEL;
            }
        }
        else
        {
            m_cacheIsChannelFirst = false;
        }

        // isChannelLast --------------
        if (layout != TENSOR_NONE)
        {
            m_cacheIsChannelLast = layout[layout.rank() - 1] == LABEL_CHANNEL || !this->hasChannel();
        }
        else
        {
            m_cacheIsChannelLast = false;
        }
    }

private:
    int  m_cacheNumSpatialDims;
    bool m_cacheIsRowMajor;
    int  m_cacheIdxChannel;
    int  m_cacheIdxWidth;
    int  m_cacheIdxHeight;
    int  m_cacheIdxDepth;
    bool m_cacheHasChannel;
    bool m_cacheIsChannelFirst;
    bool m_cacheIsChannelLast;
};

} // namespace nvcv

#endif // NVCV_TENSOR_LAYOUT_INFO_HPP
