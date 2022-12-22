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

#include "CvCudaLegacyHelpers.hpp"

#include "CvCudaLegacy.h"
#include "nvcv/TensorDataAccess.hpp"

#include <nvcv/DataType.hpp>
#include <nvcv/Exception.hpp>

#include <iostream>
#include <optional>

using namespace std;

namespace nvcv::legacy::helpers {

cuda_op::DataFormat GetLegacyDataFormat(int32_t numberChannels, int32_t numberPlanes, int32_t numberInBatch)
{
    if (numberPlanes == 1) // test for packed
    {
        return ((numberInBatch > 1) ? legacy::cuda_op::DataFormat::kNHWC : legacy::cuda_op::DataFormat::kHWC);
    }
    if (numberChannels == numberPlanes) //test for planar
    {
        return ((numberInBatch > 1) ? legacy::cuda_op::DataFormat::kNCHW : legacy::cuda_op::DataFormat::kCHW);
    }

    throw Exception(Status::ERROR_INVALID_ARGUMENT,
                    "Only planar or packed formats supported CH = %d, planes = %d, batch = %d", numberChannels,
                    numberPlanes, numberInBatch);
}

static cuda_op::DataType GetLegacyCvFloatType(int32_t bpc)
{
    if (bpc == 64)
        return cuda_op::DataType::kCV_64F;
    if (bpc == 32)
        return cuda_op::DataType::kCV_32F;
    if (bpc == 16)
        return cuda_op::DataType::kCV_16F;

    throw Exception(Status::ERROR_INVALID_ARGUMENT, "Invalid bpc(%d) for float cuda op type ", bpc);
}

static cuda_op::DataType GetLegacyCvSignedType(int32_t bpc)
{
    if (bpc == 8)
        return cuda_op::DataType::kCV_8S;
    if (bpc == 16)
        return cuda_op::DataType::kCV_16S;
    if (bpc == 32)
        return cuda_op::DataType::kCV_32S;

    throw Exception(Status::ERROR_INVALID_ARGUMENT, "Invalid bpc(%d) for signed cuda op type ", bpc);
}

static cuda_op::DataType GetLegacyCvUnsignedType(int32_t bpc)
{
    if (bpc == 8)
        return cuda_op::DataType::kCV_8U;
    if (bpc == 16)
        return cuda_op::DataType::kCV_16U;

    throw Exception(Status::ERROR_INVALID_ARGUMENT, "Invalid bpc(%d) for unsigned cuda op type ", bpc);
}

cuda_op::DataType GetLegacyDataType(int32_t bpc, nvcv::DataKind kind)
{
    switch (kind)
    {
    case nvcv::DataKind::FLOAT:
        return GetLegacyCvFloatType(bpc);

    case nvcv::DataKind::SIGNED:
        return GetLegacyCvSignedType(bpc);

    case nvcv::DataKind::UNSIGNED:
        return GetLegacyCvUnsignedType(bpc);
    }
    throw Exception(Status::ERROR_INVALID_ARGUMENT, "Only planar formats supported ");
}

cuda_op::DataType GetLegacyDataType(DataType dtype)
{
    auto bpc = dtype.bitsPerChannel();

    for (int i = 1; i < dtype.numChannels(); ++i)
    {
        if (bpc[i] != bpc[0])
        {
            throw Exception(Status::ERROR_INVALID_ARGUMENT, "All channels must have same bit-depth");
        }
    }

    return GetLegacyDataType(bpc[0], (nvcv::DataKind)dtype.dataKind());
}

cuda_op::DataType GetLegacyDataType(ImageFormat fmt)
{
    for (int i = 1; i < fmt.numPlanes(); ++i)
    {
        if (fmt.planeDataType(i) != fmt.planeDataType(0))
        {
            throw Exception(Status::ERROR_INVALID_ARGUMENT, "All planes must have the same data type");
        }
    }

    return GetLegacyDataType(fmt.planeDataType(0));
}

cuda_op::DataShape GetLegacyDataShape(const TensorShapeInfoImage &shapeInfo)
{
    return cuda_op::DataShape(shapeInfo.numSamples(), shapeInfo.numChannels(), shapeInfo.numRows(),
                              shapeInfo.numCols());
}

cuda_op::DataFormat GetLegacyDataFormat(const IImageBatchVarShape &imgBatch)
{
    ImageFormat fmt = imgBatch.uniqueFormat();
    if (!fmt)
    {
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "All images must have the same format");
    }

    for (int i = 1; i < fmt.numPlanes(); ++i)
    {
        if (fmt.planeDataType(i) != fmt.planeDataType(0))
        {
            throw Exception(Status::ERROR_INVALID_ARGUMENT, "All planes must have the same data type");
        }
    }

    if (fmt.numPlanes() >= 2)
    {
        if (fmt.numPlanes() != fmt.numChannels())
        {
            throw Exception(Status::ERROR_INVALID_ARGUMENT, "Planar images must have one channel per plane");
        }

        if (imgBatch.numImages() >= 2)
        {
            return legacy::cuda_op::DataFormat::kNCHW;
        }
        else
        {
            return legacy::cuda_op::DataFormat::kCHW;
        }
    }
    else
    {
        if (imgBatch.numImages() >= 2)
        {
            return legacy::cuda_op::DataFormat::kNHWC;
        }
        else
        {
            return legacy::cuda_op::DataFormat::kHWC;
        }
    }
}

cuda_op::DataFormat GetLegacyDataFormat(const IImageBatchVarShapeDataStridedCuda &imgBatch)
{
    ImageFormat fmt = imgBatch.uniqueFormat();
    if (!fmt)
    {
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "All images must have the same format");
    }

    for (int i = 1; i < fmt.numPlanes(); ++i)
    {
        if (fmt.planeDataType(i) != fmt.planeDataType(0))
        {
            throw Exception(Status::ERROR_INVALID_ARGUMENT, "All planes must have the same data type");
        }
    }

    if (fmt.numPlanes() >= 2)
    {
        if (fmt.numPlanes() != fmt.numChannels())
        {
            throw Exception(Status::ERROR_INVALID_ARGUMENT, "Planar images must have one channel per plane");
        }

        if (imgBatch.numImages() >= 2)
        {
            return legacy::cuda_op::DataFormat::kNCHW;
        }
        else
        {
            return legacy::cuda_op::DataFormat::kCHW;
        }
    }
    else
    {
        if (imgBatch.numImages() >= 2)
        {
            return legacy::cuda_op::DataFormat::kNHWC;
        }
        else
        {
            return legacy::cuda_op::DataFormat::kHWC;
        }
    }
}

cuda_op::DataFormat GetLegacyDataFormat(const TensorLayout &layout)
{
    if (layout == TensorLayout::NCHW)
    {
        return legacy::cuda_op::DataFormat::kNCHW;
    }
    else if (layout == TensorLayout::CHW)
    {
        return legacy::cuda_op::DataFormat::kCHW;
    }
    else if (layout == TensorLayout::NHWC)
    {
        return legacy::cuda_op::DataFormat::kNHWC;
    }
    else if (layout == TensorLayout::HWC)
    {
        return legacy::cuda_op::DataFormat::kHWC;
    }
    else
    {
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "Tensor layout not supported");
    }
}

cuda_op::DataFormat GetLegacyDataFormat(const ITensorDataStridedCuda &container)
{
    return GetLegacyDataFormat(container.layout());
}

Size2D GetMaxImageSize(const ITensorDataStridedCuda &tensor)
{
    //tensor must be NHWC or HWC
    if (auto access = TensorDataAccessStridedImagePlanar::Create(tensor))
    {
        return access->size();
    }
    else
    {
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "Get size failed.");
    }
}

Size2D GetMaxImageSize(const IImageBatchVarShapeDataStridedCuda &imageBatch)
{
    return imageBatch.maxSize();
}

} // namespace nvcv::legacy::helpers

namespace nvcv::util {

NVCVStatus TranslateError(legacy::cuda_op::ErrorCode err)
{
    using legacy::cuda_op::ErrorCode;

    switch (err)
    {
    case ErrorCode::INVALID_PARAMETER:
    case ErrorCode::INVALID_DATA_FORMAT:
    case ErrorCode::INVALID_DATA_SHAPE:
    case ErrorCode::INVALID_DATA_TYPE:
        return NVCV_ERROR_INVALID_ARGUMENT;
    default:
        return NVCV_ERROR_INTERNAL;
    }
}

const char *ToString(legacy::cuda_op::ErrorCode err, const char **perrdescr)
{
    const char *errorName = "UNKNOWN", *errorDescr = "Unknown error";

    using legacy::cuda_op::ErrorCode;

    switch (err)
    {
    case ErrorCode::SUCCESS:
        errorName  = "SUCCESS";
        errorDescr = "Operation executed successfully";
        break;
    case ErrorCode::INVALID_PARAMETER:
        errorName  = "INVALID_PARAMETER";
        errorDescr = "Some parameter is outside its acceptable range";
        break;
    case ErrorCode::INVALID_DATA_FORMAT:
        errorName  = "INVALID_DATA_FORMAT";
        errorDescr = "Data format is outside its acceptable range";
        break;

    case ErrorCode::INVALID_DATA_SHAPE:
        errorName  = "INVALID_DATA_SHAPE";
        errorDescr = "Tensor shape is outside its acceptable range";
        break;

    case ErrorCode::INVALID_DATA_TYPE:
        errorName  = "INVALID_DATA_TYPE";
        errorDescr = "Data type is outside its acceptable range";
        break;
    }

    if (perrdescr != nullptr)
    {
        *perrdescr = errorDescr;
    }

    return errorName;
}

} // namespace nvcv::util
