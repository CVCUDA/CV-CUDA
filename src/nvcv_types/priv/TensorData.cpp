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

#include "TensorData.hpp"

#include "DataType.hpp"
#include "Exception.hpp"
#include "TensorLayout.hpp"

#include <nvcv/TensorLayout.h>

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

void FillTensorData(IImage &img, NVCVTensorData &tensorData)
{
    ImageFormat fmt = img.format();

    // Must do a lot of checks to see if image is compatible with a tensor representation.

    if (img.format().memLayout() != NVCV_MEM_LAYOUT_PL)
    {
        throw Exception(NVCV_ERROR_INVALID_ARGUMENT) << "Image format's memory layout must be pitch-linear";
    }

    if (img.format().css() != NVCV_CSS_444)
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

    NVCVImageData imgData;
    img.exportData(imgData);

    if (imgData.bufferType != NVCV_IMAGE_BUFFER_STRIDED_CUDA)
    {
        throw Exception(NVCV_ERROR_INVALID_ARGUMENT)
            << "Only cuda-accessible images with pitch-linear data are accepted";
    }

    NVCVImageBufferStrided &imgStrided = imgData.buffer.strided;

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
            intptr_t planeStride = reinterpret_cast<const std::byte *>(imgStrided.planes[1].basePtr)
                                 - reinterpret_cast<const std::byte *>(imgStrided.planes[0].basePtr);

            if (reinterpret_cast<const std::byte *>(imgStrided.planes[p].basePtr)
                    - reinterpret_cast<const std::byte *>(imgStrided.planes[p - 1].basePtr)
                != planeStride)
            {
                throw Exception(NVCV_ERROR_INVALID_ARGUMENT) << "Image planes must have the same plane pitch";
            }
        }
    }

    // Now fill up tensor data with image data

    tensorData            = {}; // start everything afresh
    tensorData.bufferType = NVCV_TENSOR_BUFFER_STRIDED_CUDA;

    NVCVTensorBufferStrided &tensorStrided = tensorData.buffer.strided;

    // Infer layout and shape
    std::array<int32_t, 4> bits    = fmt.bpc();
    bool                   sameBPC = true;
    for (int i = 1; i < fmt.numChannels(); ++i)
    {
        if (bits[i] != bits[0])
        {
            sameBPC = false;
            break;
        }
    }

    if (imgStrided.numPlanes == 1)
    {
        if (fmt.numChannels() >= 2 && sameBPC)
        {
            // If same BPC, we can have channels as its own dimension,
            // as all channels have the same type.
            tensorData.layout = NVCV_TENSOR_NHWC;
        }
        else
        {
            tensorData.layout = NVCV_TENSOR_NCHW;
        }
    }
    else
    {
        tensorData.layout = NVCV_TENSOR_NCHW;
    }

    tensorData.rank = 4;
    if (tensorData.layout == NVCV_TENSOR_NHWC)
    {
        tensorData.shape[0] = 1;
        tensorData.shape[1] = imgStrided.planes[0].height;
        tensorData.shape[2] = imgStrided.planes[0].width;
        tensorData.shape[3] = fmt.numChannels();

        tensorStrided.strides[3] = fmt.planePixelStrideBytes(0) / fmt.numChannels();
        tensorStrided.strides[2] = fmt.planePixelStrideBytes(0);
        tensorStrided.strides[1] = imgStrided.planes[0].rowStride;
        tensorStrided.strides[0] = tensorStrided.strides[1] * tensorData.shape[1];

        tensorData.dtype = fmt.planeDataType(0).channelType(0).value();
    }
    else
    {
        NVCV_ASSERT(tensorData.layout == NVCV_TENSOR_NCHW);

        tensorData.shape[0] = 1;
        tensorData.shape[1] = imgStrided.numPlanes;
        tensorData.shape[2] = imgStrided.planes[0].height;
        tensorData.shape[3] = imgStrided.planes[0].width;

        tensorStrided.strides[3] = fmt.planePixelStrideBytes(0);
        tensorStrided.strides[2] = imgStrided.planes[0].rowStride;
        tensorStrided.strides[1] = tensorStrided.strides[2] * tensorData.shape[2];
        tensorStrided.strides[0] = tensorStrided.strides[1] * tensorData.shape[1];

        tensorData.dtype = fmt.planeDataType(0).value();
    }

    // Finally, assign the pointer to the memory buffer.
    tensorStrided.basePtr = imgStrided.planes[0].basePtr;
}

} // namespace nvcv::priv
