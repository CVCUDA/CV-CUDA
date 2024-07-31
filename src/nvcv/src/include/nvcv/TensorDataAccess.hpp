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

#ifndef NVCV_TENSORDATAACESSOR_HPP
#define NVCV_TENSORDATAACESSOR_HPP

#include "TensorData.hpp"
#include "TensorShapeInfo.hpp"

#include <cstddef>

namespace nvcv {

// Design is similar to TensorShapeInfo hierarchy

namespace detail {

/**
 * @class TensorDataAccessStridedImpl
 * @brief Provides access to strided tensor data, allowing for more efficient memory access patterns.
 *
 * This class offers utilities for accessing the data in a tensor using a strided memory layout.
 * It provides functions to retrieve the number of samples, data type, layout, and shape of the tensor.
 * It also contains utilities for computing strides and accessing specific samples.
 *
 * @tparam ShapeInfo  The type that contains shape information for the tensor.
 * @tparam LayoutInfo The type that contains layout information for the tensor. By default, it is derived from ShapeInfo.
 */
template<typename ShapeInfo, typename LayoutInfo = typename ShapeInfo::LayoutInfo>
class TensorDataAccessStridedImpl
{
public:
    /**
     * @brief Constructor that initializes the object with given tensor data and shape information.
     *
     * @param tdata     The strided tensor data.
     * @param infoShape The shape information of the tensor.
     */
    TensorDataAccessStridedImpl(const TensorDataStrided &tdata, const ShapeInfo &infoShape)
        : m_tdata(tdata)
        , m_infoShape(infoShape)
    {
    }

    /**
     * @brief Returns the number of samples in the tensor.
     *
     * @return The number of samples.
     */
    TensorShape::DimType numSamples() const
    {
        return m_infoShape.numSamples();
    }

    /**
     * @brief Returns the data type of the tensor.
     *
     * @return The tensor's data type.
     */
    DataType dtype() const
    {
        return m_tdata.dtype();
    }

    /**
     * @brief Returns the layout of the tensor.
     *
     * @return The tensor's layout.
     */
    const TensorLayout &layout() const
    {
        return m_tdata.layout();
    }

    /**
     * @brief Returns the shape of the tensor.
     *
     * @return The tensor's shape.
     */
    const TensorShape &shape() const
    {
        return m_tdata.shape();
    }

    /**
     * @brief Computes the stride of the sample dimension.
     *
     * @return The sample stride, or 0 if the sample dimension is not present.
     */
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

    /**
     * @brief Retrieves a pointer to the data of a specific sample.
     *
     * @param n The index of the sample.
     * @return A pointer to the sample's data.
     */
    Byte *sampleData(int n) const
    {
        return sampleData(n, m_tdata.basePtr());
    }

    /**
     * @brief Retrieves a pointer to the data of a specific sample, relative to a given base pointer.
     *
     * @param n    The index of the sample.
     * @param base The base pointer from which to compute the sample's data address.
     * @return A pointer to the sample's data.
     */
    Byte *sampleData(int n, Byte *base) const
    {
        assert(0 <= n && n < this->numSamples());
        return base + this->sampleStride() * n;
    }

    /**
     * @brief Checks if the tensor corresponds to an image.
     *
     * @return True if the tensor corresponds to an image, false otherwise.
     */
    bool isImage() const
    {
        return m_infoShape.isImage();
    }

    /**
     * @brief Returns the shape information of the tensor.
     *
     * @return The shape information.
     */
    const ShapeInfo &infoShape() const
    {
        return m_infoShape;
    }

    /**
     * @brief Returns the layout information of the tensor.
     *
     * @return The layout information.
     */
    const LayoutInfo &infoLayout() const
    {
        return m_infoShape.infoLayout();
    }

protected:
    TensorDataStrided m_tdata;

    TensorDataAccessStridedImpl(const TensorDataAccessStridedImpl &that, const TensorShapeInfo &infoShape)
        : m_tdata(that.m_tdata)
        , m_infoShape(infoShape)
    {
    }

private:
    ShapeInfo m_infoShape;
};

/**
     * @class TensorDataAccessStridedImageImpl
     * @brief Provides specialized access methods for strided tensor data representing images.
     *
     * This class is an extension of TensorDataAccessStridedImpl and offers specific utilities for accessing
     * the data in image tensors using a strided memory layout. It provides methods to retrieve the number of columns,
     * rows, channels, and other image-specific properties. Furthermore, it provides utility methods to compute strides
     * and access specific rows, channels, etc.
     *
     * @tparam ShapeInfo The type that contains shape information for the image tensor.
     */
template<typename ShapeInfo>
class TensorDataAccessStridedImageImpl : public TensorDataAccessStridedImpl<ShapeInfo>
{
    using Base = detail::TensorDataAccessStridedImpl<ShapeInfo>;

public:
    /**
     * @brief Constructor that initializes the object with the given tensor data and shape information.
     *
     * @param tdata     The strided tensor data.
     * @param infoShape The shape information of the tensor.
     */
    TensorDataAccessStridedImageImpl(const TensorDataStrided &tdata, const ShapeInfo &infoShape)
        : Base(tdata, infoShape)
    {
    }

    /**
     * @brief Returns the number of columns in the image tensor.
     *
     * @return The number of columns.
     */
    int32_t numCols() const
    {
        return this->infoShape().numCols();
    }

    /**
     * @brief Returns the number of rows in the image tensor.
     *
     * @return The number of rows.
     */
    int32_t numRows() const
    {
        return this->infoShape().numRows();
    }

    /**
     * @brief Returns the number of channels in the image tensor.
     *
     * @return The number of channels.
     */
    int32_t numChannels() const
    {
        return this->infoShape().numChannels();
    }

    /**
     * @brief Returns the size (width and height) of the image tensor.
     *
     * @return The size of the image.
     */
    Size2D size() const
    {
        return this->infoShape().size();
    }

    /**
     * @brief Computes the stride for the channel dimension.
     *
     * @return The channel stride, or 0 if the channel dimension is not present.
     */
    int64_t chStride() const
    {
        int idx = this->infoLayout().idxChannel();
        if (idx >= 0)
        {
            return this->m_tdata.stride(idx);
        }
        else
        {
            return 0;
        }
    }

    /**
     * @brief Computes the stride for the column (or width) dimension.
     *
     * @return The column stride, or 0 if the width dimension is not present.
     */
    int64_t colStride() const
    {
        int idx = this->infoLayout().idxWidth();
        if (idx >= 0)
        {
            return this->m_tdata.stride(idx);
        }
        else
        {
            return 0;
        }
    }

    /**
     * @brief Computes the stride for the row (or height) dimension.
     *
     * @return The row stride, or 0 if the height dimension is not present.
     */
    int64_t rowStride() const
    {
        int idx = this->infoLayout().idxHeight();
        if (idx >= 0)
        {
            return this->m_tdata.stride(idx);
        }
        else
        {
            return 0;
        }
    }

    /**
     * @brief Computes the stride for the depth dimension.
     *
     * @return The depth stride, or 0 if the depth dimension is not present.
     */
    int64_t depthStride() const
    {
        int idx = this->infoLayout().idxDepth();
        if (idx >= 0)
        {
            return this->m_tdata.stride(idx);
        }
        else
        {
            return 0;
        }
    }

    /**
     * @brief Retrieves a pointer to the data of a specific row.
     *
     * @param y The row index.
     * @return A pointer to the row's data.
     */
    Byte *rowData(int y) const
    {
        return rowData(y, this->m_tdata.basePtr());
    }

    /**
     * @brief Retrieves a pointer to the data of a specific row, relative to a given base pointer.
     *
     * @param y    The row index.
     * @param base The base pointer from which to compute the row's data address.
     * @return A pointer to the row's data.
     */
    Byte *rowData(int y, Byte *base) const
    {
        assert(0 <= y && y < this->numRows());
        return base + this->rowStride() * y;
    }

    /**
     * @brief Retrieves a pointer to the data of a specific channel.
     *
     * @param c The channel index.
     * @return A pointer to the channel's data.
     */
    Byte *chData(int c) const
    {
        return chData(c, this->m_tdata.basePtr());
    }

    /**
     * @brief Retrieves a pointer to the data of a specific channel, relative to a given base pointer.
     *
     * @param c    The channel index.
     * @param base The base pointer from which to compute the channel's data address.
     * @return A pointer to the channel's data.
     */
    Byte *chData(int c, Byte *base) const
    {
        assert(0 <= c && c < this->numChannels());
        return base + this->chStride() * c;
    }

protected:
    TensorDataAccessStridedImageImpl(const TensorDataAccessStridedImageImpl &that, const ShapeInfo &infoShape)
        : Base(that, infoShape)
    {
    }
};

/**
 * @class TensorDataAccessStridedImagePlanarImpl
 * @brief Provides specialized access methods for strided tensor data representing planar images.
 *
 * This class is an extension of `TensorDataAccessStridedImageImpl` and offers specific utilities for accessing
 * the data in planar image tensors using a strided memory layout. It provides methods to retrieve the number of planes,
 * compute the stride for the plane dimension, and access specific planes of the image tensor.
 *
 * @tparam ShapeInfo The type that contains shape information for the planar image tensor.
 */
template<typename ShapeInfo>
class TensorDataAccessStridedImagePlanarImpl : public TensorDataAccessStridedImageImpl<ShapeInfo>
{
    using Base = TensorDataAccessStridedImageImpl<ShapeInfo>;

public:
    /**
     * @brief Constructor that initializes the object with the given tensor data and shape information.
     *
     * @param tdata     The strided tensor data.
     * @param infoShape The shape information of the tensor.
     */
    TensorDataAccessStridedImagePlanarImpl(const TensorDataStrided &tdata, const ShapeInfo &infoShape)
        : Base(tdata, infoShape)
    {
    }

    /**
     * @brief Returns the number of planes in the planar image tensor.
     *
     * @return The number of planes.
     */
    int32_t numPlanes() const
    {
        return this->infoShape().numPlanes();
    }

    /**
     * @brief Computes the stride for the plane dimension.
     *
     * @return The plane stride, or 0 if the plane dimension is not present or is not first.
     */
    int64_t planeStride() const
    {
        if (this->infoLayout().isChannelFirst())
        {
            int ichannel = this->infoLayout().idxChannel();
            assert(ichannel >= 0);
            return this->m_tdata.stride(ichannel);
        }
        else
        {
            return 0;
        }
    }

    /**
     * @brief Retrieves a pointer to the data of a specific plane.
     *
     * @param p The plane index.
     * @return A pointer to the plane's data.
     */
    Byte *planeData(int p) const
    {
        return planeData(p, this->m_tdata.basePtr());
    }

    /**
     * @brief Retrieves a pointer to the data of a specific plane, relative to a given base pointer.
     *
     * @param p    The plane index.
     * @param base The base pointer from which to compute the plane's data address.
     * @return A pointer to the plane's data.
     */
    Byte *planeData(int p, Byte *base) const
    {
        assert(0 <= p && p < this->numPlanes());
        return base + this->planeStride() * p;
    }
};

} // namespace detail

/**
 * @class TensorDataAccessStrided
 * @brief Provides access to tensor data with a strided memory layout.
 *
 * This class is an interface for accessing tensor data that is stored in a strided memory layout.
 * It provides utilities for checking compatibility and creating instances of the class.
 */
class TensorDataAccessStrided : public detail::TensorDataAccessStridedImpl<TensorShapeInfo>
{
    using Base = detail::TensorDataAccessStridedImpl<TensorShapeInfo>;

public:
    /**
     * @brief Checks if the provided tensor data is compatible with a strided layout.
     *
     * @param data The tensor data to check.
     * @return true if the data is compatible with a strided layout, false otherwise.
     */
    static bool IsCompatible(const TensorData &data)
    {
        return data.IsCompatible<TensorDataStrided>();
    }

    /**
     * @brief Creates an instance of this class for the provided tensor data if it is compatible.
     *
     * @param data The tensor data to use for creation.
     * @return An instance of this class if the data is compatible, NullOpt otherwise.
     */
    static Optional<TensorDataAccessStrided> Create(const TensorData &data)
    {
        if (Optional<TensorDataStrided> dataStrided = data.cast<TensorDataStrided>())
        {
            return TensorDataAccessStrided(dataStrided.value());
        }
        else
        {
            return NullOpt;
        }
    }

private:
    TensorDataAccessStrided(const TensorDataStrided &data)
        : Base(data, *TensorShapeInfo::Create(data.shape()))
    {
    }
};

/**
 * @class TensorDataAccessStridedImage
 * @brief Provides access to image tensor data with a strided memory layout.
 *
 * This class extends `TensorDataAccessStrided` and provides specialized utilities for accessing image tensor data.
 */
class TensorDataAccessStridedImage : public detail::TensorDataAccessStridedImageImpl<TensorShapeInfoImage>
{
    using Base = detail::TensorDataAccessStridedImageImpl<TensorShapeInfoImage>;

public:
    /**
     * @brief Checks if the provided tensor data is compatible with an image tensor layout.
     *
     * @param data The tensor data to check.
     * @return true if the data is compatible with an image tensor layout, false otherwise.
     */
    static bool IsCompatible(const TensorData &data)
    {
        return TensorDataAccessStrided::IsCompatible(data) && TensorShapeInfoImage::IsCompatible(data.shape());
    }

    /**
     * @brief Creates an instance of this class for the provided tensor data if it is compatible.
     *
     * @param data The tensor data to use for creation.
     * @return An instance of this class if the data is compatible, NullOpt otherwise.
     */
    static Optional<TensorDataAccessStridedImage> Create(const TensorData &data)
    {
        if (IsCompatible(data))
        {
            return TensorDataAccessStridedImage(data.cast<TensorDataStrided>().value());
        }
        else
        {
            return NullOpt;
        }
    }

protected:
    TensorDataAccessStridedImage(const TensorDataStrided &data)
        : Base(data, *TensorShapeInfoImage::Create(data.shape()))
    {
    }
};

/**
 * @class TensorDataAccessStridedImagePlanar
 * @brief Provides access to planar image tensor data with a strided memory layout.
 *
 * This class extends `TensorDataAccessStridedImage` and offers specific utilities for accessing the data
 * in planar image tensors using a strided memory layout.
 */
class TensorDataAccessStridedImagePlanar
    : public detail::TensorDataAccessStridedImagePlanarImpl<TensorShapeInfoImagePlanar>
{
    using Base = detail::TensorDataAccessStridedImagePlanarImpl<TensorShapeInfoImagePlanar>;

public:
    /**
     * @brief Checks if the provided tensor data is compatible with a planar image tensor layout.
     *
     * @param data The tensor data to check.
     * @return true if the data is compatible with a planar image tensor layout, false otherwise.
     */
    static bool IsCompatible(const TensorData &data)
    {
        return TensorDataAccessStridedImage::IsCompatible(data)
            && TensorShapeInfoImagePlanar::IsCompatible(data.shape());
    }

    /**
     * @brief Creates an instance of this class for the provided tensor data if it is compatible with a planar layout.
     *
     * @param data The tensor data to use for creation.
     * @return An instance of this class if the data is compatible, NullOpt otherwise.
     */
    static Optional<TensorDataAccessStridedImagePlanar> Create(const TensorData &data)
    {
        if (IsCompatible(data))
        {
            return TensorDataAccessStridedImagePlanar(data.cast<TensorDataStrided>().value());
        }
        else
        {
            return NullOpt;
        }
    }

protected:
    TensorDataAccessStridedImagePlanar(const TensorDataStrided &data)
        : Base(data, *TensorShapeInfoImagePlanar::Create(data.shape()))
    {
    }
};

} // namespace nvcv

#endif // NVCV_TENSORDATAACESSOR_HPP
