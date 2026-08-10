/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "TensorData.hpp"

#include "DataType.hpp"
#include "Exception.hpp"
#include "TensorLayout.hpp"

#include <nvcv/TensorLayout.h>

#include <array>
#include <sstream>

namespace nvcv::priv {

NVCVTensorLayout GetTensorLayoutFor(ImageFormat fmt, int nbatches)
{
    (void)nbatches;

    int nplanes = fmt.numPlanes();

    if (nplanes == 1)
    {
        return NVCV_TENSOR_NHWC;
    }
    else if (nplanes == fmt.numChannels())
    {
        return NVCV_TENSOR_NCHW;
    }
    else
    {
        throw Exception(NVCV_ERROR_INVALID_ARGUMENT) << "Image format cannot be semi-planar, but it is: " << fmt;
    }
}

static void ValidateTensorCompatibleFormat(ImageFormat fmt)
{
    if (fmt.memLayout() != NVCV_MEM_LAYOUT_PL)
    {
        throw Exception(NVCV_ERROR_INVALID_ARGUMENT) << "Image format's memory layout must be pitch-linear";
    }

    if (fmt.css() != NVCV_CSS_444)
    {
        throw Exception(NVCV_ERROR_INVALID_ARGUMENT) << "Image format's memory layout must not have sub-sampled planes";
    }

    for (int p = 1; p < fmt.numPlanes(); ++p)
    {
        if (fmt.planeDataType(p) != fmt.planeDataType(0))
        {
            throw Exception(NVCV_ERROR_INVALID_ARGUMENT) << "Data type of all image planes must be the same";
        }
    }
}

static void ValidateTensorCompatibleData(const NVCVImageData &imgData)
{
    if (imgData.bufferType != NVCV_IMAGE_BUFFER_STRIDED_CUDA)
    {
        throw Exception(NVCV_ERROR_INVALID_ARGUMENT)
            << "Only cuda-accessible images with pitch-linear data are accepted";
    }
}

static intptr_t PlaneAddressStride(const NVCVImagePlaneStrided &prev, const NVCVImagePlaneStrided &next)
{
    return reinterpret_cast<const std::byte *>(next.basePtr) - reinterpret_cast<const std::byte *>(prev.basePtr);
}

static void ValidateTensorCompatiblePlanes(const NVCVImageBufferStrided &imgStrided)
{
    for (int p = 1; p < imgStrided.numPlanes; ++p)
    {
        if (imgStrided.planes[p].width != imgStrided.planes[0].width
            || imgStrided.planes[p].height != imgStrided.planes[0].height)
        {
            throw Exception(NVCV_ERROR_INVALID_ARGUMENT) << "All image planes must have the same dimensions";
        }

        if (imgStrided.planes[p].rowStride != imgStrided.planes[0].rowStride)
        {
            throw Exception(NVCV_ERROR_INVALID_ARGUMENT) << "All image planes must have the same row pitch";
        }

        if (imgStrided.planes[p].basePtr <= imgStrided.planes[0].basePtr)
        {
            throw Exception(NVCV_ERROR_INVALID_ARGUMENT)
                << "Consecutive image planes must have increasing memory addresses";
        }

        if (p >= 2)
        {
            intptr_t planeStride = PlaneAddressStride(imgStrided.planes[0], imgStrided.planes[1]);

            if (PlaneAddressStride(imgStrided.planes[p - 1], imgStrided.planes[p]) != planeStride)
            {
                throw Exception(NVCV_ERROR_INVALID_ARGUMENT) << "Image planes must have the same plane pitch";
            }
        }
    }
}

static bool HasSameBitsPerChannel(ImageFormat fmt)
{
    std::array<int32_t, 4> bits = fmt.bpc();
    for (int i = 1; i < fmt.numChannels(); ++i)
    {
        if (bits[i] != bits[0])
        {
            return false;
        }
    }
    return true;
}

static NVCVTensorLayout InferTensorLayout(const NVCVImageBufferStrided &imgStrided, ImageFormat fmt)
{
    if (imgStrided.numPlanes == 1 && fmt.numChannels() >= 2 && HasSameBitsPerChannel(fmt))
    {
        // If same BPC, we can have channels as its own dimension,
        // as all channels have the same type.
        return NVCV_TENSOR_NHWC;
    }

    return NVCV_TENSOR_NCHW;
}

static void FillTensorStridedNHWC(const NVCVImageBufferStrided &imgStrided, ImageFormat fmt, NVCVTensorData &tensorData)
{
    tensorData.shape[0] = 1;
    tensorData.shape[1] = imgStrided.planes[0].height;
    tensorData.shape[2] = imgStrided.planes[0].width;
    tensorData.shape[3] = fmt.numChannels();

    NVCVTensorBufferStrided &tensorStrided = tensorData.buffer.strided;
    tensorStrided.strides[3]               = fmt.planePixelStrideBytes(0) / fmt.numChannels();
    tensorStrided.strides[2]               = fmt.planePixelStrideBytes(0);
    tensorStrided.strides[1]               = imgStrided.planes[0].rowStride;
    tensorStrided.strides[0]               = tensorStrided.strides[1] * tensorData.shape[1];

    tensorData.dtype = fmt.planeDataType(0).channelType(0).value();
}

static void FillTensorStridedNCHW(const NVCVImageBufferStrided &imgStrided, ImageFormat fmt, NVCVTensorData &tensorData)
{
    tensorData.shape[0] = 1;
    tensorData.shape[1] = imgStrided.numPlanes;
    tensorData.shape[2] = imgStrided.planes[0].height;
    tensorData.shape[3] = imgStrided.planes[0].width;

    NVCVTensorBufferStrided &tensorStrided = tensorData.buffer.strided;
    tensorStrided.strides[3]               = fmt.planePixelStrideBytes(0);
    tensorStrided.strides[2]               = imgStrided.planes[0].rowStride;
    tensorStrided.strides[1]               = tensorStrided.strides[2] * tensorData.shape[2];
    tensorStrided.strides[0]               = tensorStrided.strides[1] * tensorData.shape[1];

    tensorData.dtype = fmt.planeDataType(0).value();
}

void FillTensorData(const IImage &img, NVCVTensorData &tensorData)
{
    ImageFormat fmt = img.format();
    ValidateTensorCompatibleFormat(fmt);

    NVCVImageData imgData;
    img.exportData(imgData);
    ValidateTensorCompatibleData(imgData);

    NVCVImageBufferStrided &imgStrided = imgData.buffer.strided;
    ValidateTensorCompatiblePlanes(imgStrided);

    tensorData            = {};
    tensorData.bufferType = NVCV_TENSOR_BUFFER_STRIDED_CUDA;
    tensorData.layout     = InferTensorLayout(imgStrided, fmt);
    tensorData.rank       = 4;

    if (tensorData.layout == NVCV_TENSOR_NHWC)
    {
        FillTensorStridedNHWC(imgStrided, fmt, tensorData);
    }
    else
    {
        NVCV_ASSERT(tensorData.layout == NVCV_TENSOR_NCHW);
        FillTensorStridedNCHW(imgStrided, fmt, tensorData);
    }

    // Finally, assign the pointer to the memory buffer.
    tensorData.buffer.strided.basePtr = imgStrided.planes[0].basePtr;
}

/**
 * @brief Simplifies a shape by collapsing dimensions that are not strided
 *
 * @param[in] rank number of dimensions
 * @param[in] shape
 * @param[in] stride
 * @param[out] out_shape
 * @param[out] out_strides
 * @return int out_rank
 */
static int Simplify(int rank, const int64_t *shape, const int64_t *stride, int64_t *out_shape, int64_t *out_strides)
{
    if (rank <= 1) // Nothing to simplify
    {
        if (rank == 1)
        {
            out_shape[0]   = shape[0];
            out_strides[0] = stride[0];
        }
        return rank;
    }

    int     out_rank = 0;
    int64_t vol      = shape[0];
    for (int d = 1; d < rank; d++)
    {
        if (stride[d - 1] != shape[d] * stride[d])
        {
            out_strides[out_rank] = stride[d - 1];
            out_shape[out_rank]   = vol;
            vol                   = shape[d];
            out_rank++;
        }
        else
        {
            vol *= shape[d];
        }
    }
    out_strides[out_rank] = stride[rank - 1];
    out_shape[out_rank]   = vol;
    out_rank++;
    return out_rank;
}

/**
 * @brief Reshapes a simplified shape (non-strided dimensions are collapsed) to a target shape if possible.
 *        Calculates the output strides.
 *
 * @param[in] in_rank
 * @param[in] in_shape
 * @param[in] in_strides
 * @param[in] target_rank
 * @param[in] target_shape
 * @param[out] out_strides
 *
 * @return true if reshape is possible, false otherwise
 */
static bool ReshapeSimplified(int in_rank, const int64_t *in_shape, const int64_t *in_strides, int target_rank,
                              const int64_t *target_shape, int64_t *out_strides)
{
    int i = 0;
    int j = 0;
    for (; i < in_rank && j < target_rank; i++) // NOSONAR: input and target ranks advance independently.
    {
        int64_t in_e        = in_shape[i];
        int64_t out_v       = 1;
        int     group_start = j;
        while (j < target_rank && (out_v * target_shape[j]) <= in_e) out_v *= target_shape[j++];

        if (out_v != in_e)
            return false; // reshape is not possible

        int64_t s = in_strides[i];
        for (int d = j - 1; d >= group_start; d--)
        {
            out_strides[d] = s;
            s *= target_shape[d];
        }
    }
    return true;
}

static std::string ShapeStr(int rank, const int64_t *sh)
{
    std::stringstream ss;
    ss << "(";
    for (int d = 0; d < rank; d++)
    {
        if (d > 0)
            ss << ", ";
        ss << sh[d];
    }
    ss << ")";
    return ss.str();
}

void ReshapeTensorData(NVCVTensorData &tensor_data, int new_rank, const int64_t *new_shape, NVCVTensorLayout new_layout)
{
    int64_t old_volume = 1;
    for (int d = 0; d < tensor_data.rank; d++) old_volume *= tensor_data.shape[d];

    // REVISIT: Add 0D tensor support, once it's supported accross the board
    if (new_rank < 1 || new_rank > NVCV_TENSOR_MAX_RANK)
        throw Exception(NVCV_ERROR_INVALID_ARGUMENT)
            << "Number of dimensions must be between 1 and " << NVCV_TENSOR_MAX_RANK << ", not " << new_rank;

    int64_t new_volume = 1;
    for (int d = 0; d < new_rank; d++) new_volume *= new_shape[d];

    if (new_volume != old_volume)
    {
        throw Exception(NVCV_ERROR_INVALID_ARGUMENT)
            << "The volume (" << new_volume << ") of the provided shape " << ShapeStr(new_rank, new_shape)
            << " does not match the size of the array (" << old_volume << ")";
    }

    // layout ------------
    if (new_layout.rank > 0 && new_layout.rank != new_rank)
    {
        throw Exception(NVCV_ERROR_INVALID_ARGUMENT)
            << "The number of dimensions of the provided layout and shape do not match. Got a "
               "shape with "
            << new_rank << " dimensions and a layout with " << new_layout.rank << " dimensions";
    }
    tensor_data.layout = new_layout;

    // Check strides ------------

    // right now is the only option supported
    assert(tensor_data.bufferType == NVCV_TENSOR_BUFFER_STRIDED_CUDA);

    // Collapses non-strided dimensions into groups
    // Example 1:
    // A tensor with shape (480, 640, 3) and strides (2560, 4, 1)
    // will be collapsed into (307200, 3) with strides (4, 1).
    // Example 2:
    // A tensor with shape (480, 640, 3) and strides (2560, 3, 1)
    // will be collapsed into (921600,) with strides (1,).
    std::array<int64_t, NVCV_TENSOR_MAX_RANK> simplified_shape;
    std::array<int64_t, NVCV_TENSOR_MAX_RANK> simplified_strides;
    int simplified_rank = Simplify(tensor_data.rank, tensor_data.shape, tensor_data.buffer.strided.strides,
                                   simplified_shape.data(), simplified_strides.data());

    // Calculate output strides (if reshape is possible) or throw an error
    if (bool ret = ReshapeSimplified(simplified_rank, simplified_shape.data(), simplified_strides.data(), new_rank,
                                     new_shape, tensor_data.buffer.strided.strides);
        !ret)
    {
        throw Exception(NVCV_ERROR_INVALID_ARGUMENT)
            << "Cannot reshape"
            << ". Original shape: " << ShapeStr(tensor_data.rank, tensor_data.shape)
            << ", Strides: " << ShapeStr(tensor_data.rank, tensor_data.buffer.strided.strides)
            << ", Target shape: " << ShapeStr(new_rank, new_shape);
    }

    // Set the new shape to the tensor data
    tensor_data.rank = new_rank;
    for (int d = 0; d < tensor_data.rank; d++) tensor_data.shape[d] = new_shape[d];
}

} // namespace nvcv::priv
