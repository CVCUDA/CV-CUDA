/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef CVCUDA_PRIV_CHANNEL_AXIS_COMMON_CUH
#define CVCUDA_PRIV_CHANNEL_AXIS_COMMON_CUH

#include <cvcuda/cuda_tools/TypeTraits.hpp>
#include <nvcv/DataType.hpp>
#include <nvcv/Exception.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/ImageBatchData.hpp>
#include <nvcv/ImageData.hpp>
#include <nvcv/Optional.hpp>
#include <nvcv/TensorData.hpp>
#include <nvcv/TensorDataAccess.hpp>
#include <nvcv/TensorLayout.hpp>

#include <cstdint>
#include <type_traits>

// Shared preamble for operators that treat channels as a *tensor axis* rather than as dtype lanes:
// the tensor dtype must be scalar, channels live on the C dimension, and only 1 or 3 of them are
// admitted. Every path is planar-aware and reports which layout it saw.
//
// This is the sibling of SameShapeCommon.cuh, and the distinction is worth stating because the two
// validator sets look alike at a glance. Both require src and dst to agree on geometry. This one
// additionally rejects vector dtypes ("use the C dimension for image channels"), caps the channel
// count at 1 or 3, returns `isPlanar`, and carries an `isEmpty` early-out. SameShapeCommon's family
// accepts a vector dtype and derives its interleaved-channel count from it. An operator belongs to
// exactly one of the two -- if a new one fits neither, add a third header rather than flagging a
// difference into either of these.
namespace cvcuda::priv::channel_axis {

namespace cuda = nvcv::cuda;

inline bool IsPlanarLayout(nvcv::TensorLayout layout)
{
    return layout == nvcv::TENSOR_NCHW || layout == nvcv::TENSOR_CHW;
}

// Resolve the runtime channel count and dtype to a compile-time pixel type and channel count, then
// hand both to `cb`. Unlike SameShapeCommon's split dispatch, this switch resolves type and channel
// together, so it has no separate type switch to defer to and hoists whole. `opName` names the
// operator in the two error messages only.
template<typename Cb>
inline void RunChannelSwitch(int numChannels, nvcv::DataType dtype, const char *opName, const Cb &cb)
{
    const bool isU8  = (dtype == nvcv::TYPE_U8 || dtype == nvcv::TYPE_3U8);
    const bool isF32 = (dtype == nvcv::TYPE_F32 || dtype == nvcv::TYPE_3F32);

    if (!isU8 && !isF32)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Invalid data type: %s supports 8-bit unsigned and 32-bit float", opName);
    }

    if (numChannels == 1)
    {
        if (isU8)
        {
            cb(uchar1{}, std::integral_constant<int, 1>{});
        }
        else
        {
            cb(float1{}, std::integral_constant<int, 1>{});
        }
    }
    else if (numChannels == 3)
    {
        if (isU8)
        {
            cb(uchar3{}, std::integral_constant<int, 3>{});
        }
        else
        {
            cb(float3{}, std::integral_constant<int, 3>{});
        }
    }
    else
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Invalid number of channels: %s supports 1 or 3 channels", opName);
    }
}

// Validation ----------------------------------------------------------------------------

// `dtype` and `numSamples` are both reported because the consumers want different ones and the body
// computes both regardless; an operator that needs only one ignores the other.
inline bool ValidateSrcDstTensors(bool &isEmpty, int &numChannels, nvcv::DataType &dtype, int &numSamples,
                                  const nvcv::Optional<nvcv::TensorDataStridedCuda> &srcData,
                                  const nvcv::Optional<nvcv::TensorDataStridedCuda> &dstData)
{
    if (!srcData)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input must be cuda-accessible, pitch-linear tensor");
    }
    if (!dstData)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Output must be cuda-accessible, pitch-linear tensor");
    }
    if (srcData->layout() != dstData->layout())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input and output must have the same layout");
    }

    const bool isPlanar = IsPlanarLayout(srcData->layout());
    if (srcData->layout() != nvcv::TENSOR_HWC && srcData->layout() != nvcv::TENSOR_NHWC && !isPlanar)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input must have (N)HWC or (N)CHW layout");
    }
    if (srcData->dtype() != dstData->dtype())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input and output must have the same data type");
    }
    if (srcData->dtype().numChannels() != 1)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Tensor data type must be scalar; use the C dimension for image channels");
    }

    auto srcAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*srcData);
    auto dstAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*dstData);
    if (!srcAccess || !dstAccess)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input/output must be accessible as strided images");
    }
    if (srcAccess->numSamples() != dstAccess->numSamples())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Incompatible input/output number of samples");
    }

    numChannels = srcAccess->numChannels();
    if (numChannels != dstAccess->numChannels())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Incompatible input/output number of channels");
    }
    if (numChannels != 1 && numChannels != 3)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input must have 1 or 3 channels");
    }
    if (srcAccess->numCols() != dstAccess->numCols() || srcAccess->numRows() != dstAccess->numRows())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input and output must have matching width and height");
    }

    numSamples = static_cast<int>(srcAccess->numSamples());
    isEmpty    = numSamples == 0 || srcAccess->numRows() == 0 || srcAccess->numCols() == 0;
    dtype      = srcData->dtype();
    return isPlanar;
}

inline void ValidateImagePlanes(const nvcv::Image &image, const nvcv::ImageFormat &format)
{
    auto data = image.exportData<nvcv::ImageDataStridedCuda>();
    if (!data || data->numPlanes() != format.numPlanes())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Image plane descriptors must match the image format");
    }

    constexpr int64_t limit = cuda::TypeTraits<int32_t>::max;
    for (int p = 0; p < data->numPlanes(); ++p)
    {
        const nvcv::ImagePlaneStrided &plane    = data->plane(p);
        const nvcv::Size2D             expected = format.planeSize(image.size(), p);
        if (plane.width != expected.w || plane.height != expected.h)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Image plane descriptors must match the image format and size");
        }
        if (expected != image.size())
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "All image planes must have matching width and height");
        }

        const int64_t pixelStride = format.planePixelStrideBytes(p);
        if (plane.rowStride < 0
            || (static_cast<int64_t>(plane.height - 1) * plane.rowStride
                    + static_cast<int64_t>(plane.width - 1) * pixelStride
                > limit))
        {
            throw nvcv::Exception(nvcv::Status::ERROR_OVERFLOW,
                                  "Input or output image-plane maximum byte offset exceeds %d",
                                  static_cast<int>(limit));
        }
    }
}

inline bool ValidateSrcDstVarBatch(bool &isEmpty, int &numChannels, nvcv::DataType &dtype,
                                   const nvcv::ImageBatchVarShape &src, const nvcv::ImageBatchVarShape &dst,
                                   const nvcv::Optional<nvcv::ImageBatchVarShapeDataStridedCuda> &srcData,
                                   const nvcv::Optional<nvcv::ImageBatchVarShapeDataStridedCuda> &dstData)
{
    if (!srcData)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input must be cuda-accessible, varshape pitch-linear image batch");
    }
    if (!dstData)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Output must be cuda-accessible, varshape pitch-linear image batch");
    }
    if (srcData->numImages() != dstData->numImages())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Incompatible input/output number of samples");
    }
    isEmpty = srcData->numImages() == 0;
    if (isEmpty)
    {
        return false;
    }

    const auto &srcFormat = srcData->uniqueFormat();
    const auto &dstFormat = dstData->uniqueFormat();
    if (!srcFormat || !dstFormat)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "All images in a batch must have the same format");
    }
    if (srcFormat != dstFormat)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input and output must have the same format");
    }

    numChannels = srcFormat.numChannels();
    if (numChannels != 1 && numChannels != 3)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "The input must have 1 or 3 channels");
    }
    if (numChannels == 3
        && (srcFormat.colorModel() != nvcv::ColorModel::RGB
            || (srcFormat.swizzle() != nvcv::Swizzle::S_XYZ0 && srcFormat.swizzle() != nvcv::Swizzle::S_XYZ1)))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Three-channel input must use RGB channel order");
    }

    nvcv::ExtraChannelInfo extraChannels{};
    srcFormat.extraChannelInfo(&extraChannels);
    if (extraChannels.numChannels != 0)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Image formats with extra channels are not supported");
    }
    if (srcFormat.chromaSubsampling() != nvcv::ChromaSubsampling::CSS_444)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Chroma-subsampled image formats are not supported");
    }

    const int numPlanes = srcFormat.numPlanes();
    dtype               = srcFormat.planeDataType(0);
    if (numPlanes == 1)
    {
        if (dtype.numChannels() != numChannels || srcFormat.planePixelStrideBytes(0) != dtype.strideBytes())
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Single-plane images must have one packed element per pixel");
        }
    }
    else if (numPlanes != numChannels || dtype.numChannels() != 1
             || srcFormat.planePixelStrideBytes(0) != dtype.strideBytes())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Planar images must have one scalar, full-resolution plane per channel");
    }
    for (int p = 1; p < numPlanes; ++p)
    {
        if (srcFormat.planeDataType(p) != dtype || srcFormat.planeDataType(p).numChannels() != 1
            || srcFormat.planePixelStrideBytes(p) != dtype.strideBytes())
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "All image planes must have the same packed scalar data type");
        }
    }

    for (int i = 0; i < src.numImages(); ++i)
    {
        const nvcv::Size2D srcSize = src[i].size();
        const nvcv::Size2D dstSize = dst[i].size();
        if (srcSize != dstSize)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Input and output image %d sizes must match: input is %dx%d, output is %dx%d", i,
                                  srcSize.w, srcSize.h, dstSize.w, dstSize.h);
        }
        ValidateImagePlanes(src[i], srcFormat);
        ValidateImagePlanes(dst[i], dstFormat);
    }

    return numPlanes > 1;
}

} // namespace cvcuda::priv::channel_axis

#endif // CVCUDA_PRIV_CHANNEL_AXIS_COMMON_CUH
