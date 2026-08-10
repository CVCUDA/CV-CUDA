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

#include "CvtColorUtil.hpp"

#include <common/String.hpp>
#include <cvcuda/Types.h>
#include <cvcuda/cuda_tools/TypeTraits.hpp>

#include <array>
#include <map>
#include <stdexcept>
#include <unordered_map>

namespace {

class CvtColorError : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

} // namespace

const std::unordered_map<NVCVColorConversionCode, NVCVImageFormat> kOutputFormat = {
    {     NVCV_COLOR_BGR2BGRA, NVCV_IMAGE_FORMAT_BGRA8},
    {     NVCV_COLOR_RGB2RGBA, NVCV_IMAGE_FORMAT_RGBA8},
    {     NVCV_COLOR_BGRA2BGR,  NVCV_IMAGE_FORMAT_BGR8},
    {     NVCV_COLOR_RGBA2RGB,  NVCV_IMAGE_FORMAT_RGB8},
    {     NVCV_COLOR_BGR2RGBA, NVCV_IMAGE_FORMAT_RGBA8},
    {     NVCV_COLOR_RGB2BGRA, NVCV_IMAGE_FORMAT_BGRA8},
    {     NVCV_COLOR_RGBA2BGR,  NVCV_IMAGE_FORMAT_BGR8},
    {     NVCV_COLOR_BGRA2RGB,  NVCV_IMAGE_FORMAT_RGB8},
    {      NVCV_COLOR_BGR2RGB,  NVCV_IMAGE_FORMAT_RGB8},
    {      NVCV_COLOR_RGB2BGR,  NVCV_IMAGE_FORMAT_BGR8},
    {    NVCV_COLOR_BGRA2RGBA, NVCV_IMAGE_FORMAT_RGBA8},
    {    NVCV_COLOR_RGBA2BGRA, NVCV_IMAGE_FORMAT_BGRA8},
    {     NVCV_COLOR_BGR2GRAY, NVCV_IMAGE_FORMAT_Y8_ER},
    {     NVCV_COLOR_RGB2GRAY, NVCV_IMAGE_FORMAT_Y8_ER},
    {     NVCV_COLOR_GRAY2BGR,  NVCV_IMAGE_FORMAT_BGR8},
    {     NVCV_COLOR_GRAY2RGB,  NVCV_IMAGE_FORMAT_RGB8},
    {      NVCV_COLOR_BGR2HSV,  NVCV_IMAGE_FORMAT_HSV8},
    {      NVCV_COLOR_RGB2HSV,  NVCV_IMAGE_FORMAT_HSV8},
    {      NVCV_COLOR_HSV2BGR,  NVCV_IMAGE_FORMAT_BGR8},
    {      NVCV_COLOR_HSV2RGB,  NVCV_IMAGE_FORMAT_RGB8},
    { NVCV_COLOR_BGR2HSV_FULL,  NVCV_IMAGE_FORMAT_HSV8},
    { NVCV_COLOR_RGB2HSV_FULL,  NVCV_IMAGE_FORMAT_HSV8},
    { NVCV_COLOR_HSV2BGR_FULL,  NVCV_IMAGE_FORMAT_BGR8},
    { NVCV_COLOR_HSV2RGB_FULL,  NVCV_IMAGE_FORMAT_RGB8},
    {      NVCV_COLOR_BGR2YUV,  NVCV_IMAGE_FORMAT_YUV8},
    {      NVCV_COLOR_RGB2YUV,  NVCV_IMAGE_FORMAT_YUV8},
    {      NVCV_COLOR_YUV2BGR,  NVCV_IMAGE_FORMAT_BGR8},
    {      NVCV_COLOR_YUV2RGB,  NVCV_IMAGE_FORMAT_RGB8},
    { NVCV_COLOR_YUV2RGB_NV12,  NVCV_IMAGE_FORMAT_RGB8},
    { NVCV_COLOR_YUV2BGR_NV12,  NVCV_IMAGE_FORMAT_BGR8},
    { NVCV_COLOR_YUV2RGB_NV21,  NVCV_IMAGE_FORMAT_RGB8},
    { NVCV_COLOR_YUV2BGR_NV21,  NVCV_IMAGE_FORMAT_BGR8},
    { NVCV_COLOR_YUV420sp2RGB,  NVCV_IMAGE_FORMAT_RGB8},
    { NVCV_COLOR_YUV420sp2BGR,  NVCV_IMAGE_FORMAT_BGR8},
    {NVCV_COLOR_YUV2RGBA_NV12, NVCV_IMAGE_FORMAT_RGBA8},
    {NVCV_COLOR_YUV2BGRA_NV12, NVCV_IMAGE_FORMAT_BGRA8},
    {NVCV_COLOR_YUV2RGBA_NV21, NVCV_IMAGE_FORMAT_RGBA8},
    {NVCV_COLOR_YUV2BGRA_NV21, NVCV_IMAGE_FORMAT_BGRA8},
    {NVCV_COLOR_YUV420sp2RGBA, NVCV_IMAGE_FORMAT_RGBA8},
    {NVCV_COLOR_YUV420sp2BGRA, NVCV_IMAGE_FORMAT_BGRA8},
    { NVCV_COLOR_YUV2RGB_YV12,  NVCV_IMAGE_FORMAT_RGB8},
    { NVCV_COLOR_YUV2BGR_YV12,  NVCV_IMAGE_FORMAT_BGR8},
    { NVCV_COLOR_YUV2RGB_IYUV,  NVCV_IMAGE_FORMAT_RGB8},
    { NVCV_COLOR_YUV2BGR_IYUV,  NVCV_IMAGE_FORMAT_BGR8},
    { NVCV_COLOR_YUV2RGB_I420,  NVCV_IMAGE_FORMAT_RGB8},
    { NVCV_COLOR_YUV2BGR_I420,  NVCV_IMAGE_FORMAT_BGR8},
    { NVCV_COLOR_YUV420sp2RGB,  NVCV_IMAGE_FORMAT_RGB8},
    { NVCV_COLOR_YUV420sp2BGR,  NVCV_IMAGE_FORMAT_BGR8},
    {NVCV_COLOR_YUV2RGBA_YV12, NVCV_IMAGE_FORMAT_RGBA8},
    {NVCV_COLOR_YUV2BGRA_YV12, NVCV_IMAGE_FORMAT_BGRA8},
    {NVCV_COLOR_YUV2RGBA_IYUV, NVCV_IMAGE_FORMAT_RGBA8},
    {NVCV_COLOR_YUV2BGRA_IYUV, NVCV_IMAGE_FORMAT_BGRA8},
    {NVCV_COLOR_YUV2RGBA_I420, NVCV_IMAGE_FORMAT_RGBA8},
    {NVCV_COLOR_YUV2BGRA_I420, NVCV_IMAGE_FORMAT_BGRA8},
    {NVCV_COLOR_YUV420sp2RGBA, NVCV_IMAGE_FORMAT_RGBA8},
    {NVCV_COLOR_YUV420sp2BGRA, NVCV_IMAGE_FORMAT_BGRA8},
    { NVCV_COLOR_YUV2GRAY_420, NVCV_IMAGE_FORMAT_Y8_ER},
    {NVCV_COLOR_YUV2GRAY_NV21, NVCV_IMAGE_FORMAT_Y8_ER},
    {NVCV_COLOR_YUV2GRAY_NV12, NVCV_IMAGE_FORMAT_Y8_ER},
    {NVCV_COLOR_YUV2GRAY_YV12, NVCV_IMAGE_FORMAT_Y8_ER},
    {NVCV_COLOR_YUV2GRAY_IYUV, NVCV_IMAGE_FORMAT_Y8_ER},
    {NVCV_COLOR_YUV2GRAY_I420, NVCV_IMAGE_FORMAT_Y8_ER},
    {NVCV_COLOR_YUV420sp2GRAY, NVCV_IMAGE_FORMAT_Y8_ER},
    { NVCV_COLOR_YUV420p2GRAY, NVCV_IMAGE_FORMAT_Y8_ER},
    { NVCV_COLOR_YUV2RGB_UYVY,  NVCV_IMAGE_FORMAT_RGB8},
    { NVCV_COLOR_YUV2BGR_UYVY,  NVCV_IMAGE_FORMAT_BGR8},
    { NVCV_COLOR_YUV2RGB_Y422,  NVCV_IMAGE_FORMAT_RGB8},
    { NVCV_COLOR_YUV2BGR_Y422,  NVCV_IMAGE_FORMAT_BGR8},
    { NVCV_COLOR_YUV2RGB_UYNV,  NVCV_IMAGE_FORMAT_RGB8},
    { NVCV_COLOR_YUV2BGR_UYNV,  NVCV_IMAGE_FORMAT_BGR8},
    {NVCV_COLOR_YUV2RGBA_UYVY, NVCV_IMAGE_FORMAT_RGBA8},
    {NVCV_COLOR_YUV2BGRA_UYVY, NVCV_IMAGE_FORMAT_BGRA8},
    {NVCV_COLOR_YUV2RGBA_Y422, NVCV_IMAGE_FORMAT_RGBA8},
    {NVCV_COLOR_YUV2BGRA_Y422, NVCV_IMAGE_FORMAT_BGRA8},
    {NVCV_COLOR_YUV2RGBA_UYNV, NVCV_IMAGE_FORMAT_RGBA8},
    {NVCV_COLOR_YUV2BGRA_UYNV, NVCV_IMAGE_FORMAT_BGRA8},
    { NVCV_COLOR_YUV2RGB_YUY2,  NVCV_IMAGE_FORMAT_RGB8},
    { NVCV_COLOR_YUV2BGR_YUY2,  NVCV_IMAGE_FORMAT_BGR8},
    { NVCV_COLOR_YUV2RGB_YVYU,  NVCV_IMAGE_FORMAT_RGB8},
    { NVCV_COLOR_YUV2BGR_YVYU,  NVCV_IMAGE_FORMAT_BGR8},
    { NVCV_COLOR_YUV2RGB_YUYV,  NVCV_IMAGE_FORMAT_RGB8},
    { NVCV_COLOR_YUV2BGR_YUYV,  NVCV_IMAGE_FORMAT_BGR8},
    { NVCV_COLOR_YUV2RGB_YUNV,  NVCV_IMAGE_FORMAT_RGB8},
    { NVCV_COLOR_YUV2BGR_YUNV,  NVCV_IMAGE_FORMAT_BGR8},
    {NVCV_COLOR_YUV2RGBA_YUY2, NVCV_IMAGE_FORMAT_RGBA8},
    {NVCV_COLOR_YUV2BGRA_YUY2, NVCV_IMAGE_FORMAT_BGRA8},
    {NVCV_COLOR_YUV2RGBA_YVYU, NVCV_IMAGE_FORMAT_RGBA8},
    {NVCV_COLOR_YUV2BGRA_YVYU, NVCV_IMAGE_FORMAT_BGRA8},
    {NVCV_COLOR_YUV2RGBA_YUYV, NVCV_IMAGE_FORMAT_RGBA8},
    {NVCV_COLOR_YUV2BGRA_YUYV, NVCV_IMAGE_FORMAT_BGRA8},
    {NVCV_COLOR_YUV2RGBA_YUNV, NVCV_IMAGE_FORMAT_RGBA8},
    {NVCV_COLOR_YUV2BGRA_YUNV, NVCV_IMAGE_FORMAT_BGRA8},
    {NVCV_COLOR_YUV2GRAY_UYVY, NVCV_IMAGE_FORMAT_Y8_ER},
    {NVCV_COLOR_YUV2GRAY_YUY2, NVCV_IMAGE_FORMAT_Y8_ER},
    {NVCV_COLOR_YUV2GRAY_Y422, NVCV_IMAGE_FORMAT_Y8_ER},
    {NVCV_COLOR_YUV2GRAY_UYNV, NVCV_IMAGE_FORMAT_Y8_ER},
    {NVCV_COLOR_YUV2GRAY_YVYU, NVCV_IMAGE_FORMAT_Y8_ER},
    {NVCV_COLOR_YUV2GRAY_YUYV, NVCV_IMAGE_FORMAT_Y8_ER},
    {NVCV_COLOR_YUV2GRAY_YUNV, NVCV_IMAGE_FORMAT_Y8_ER},
    { NVCV_COLOR_RGB2YUV_NV12,    NVCV_IMAGE_FORMAT_Y8},
    { NVCV_COLOR_BGR2YUV_NV12,    NVCV_IMAGE_FORMAT_Y8},
    { NVCV_COLOR_RGB2YUV_NV21,    NVCV_IMAGE_FORMAT_Y8},
    { NVCV_COLOR_BGR2YUV_NV21,    NVCV_IMAGE_FORMAT_Y8},
    {NVCV_COLOR_RGBA2YUV_NV12,    NVCV_IMAGE_FORMAT_Y8},
    {NVCV_COLOR_BGRA2YUV_NV12,    NVCV_IMAGE_FORMAT_Y8},
};

nvcv::ImageFormat GetOutputFormat(nvcv::DataType in, NVCVColorConversionCode code)
{
    auto outFormatIt = kOutputFormat.find(code);
    if (outFormatIt == kOutputFormat.end())
    {
        throw CvtColorError("Invalid color conversion code");
    }
    nvcv::ImageFormat outFormat{outFormatIt->second};

    auto inPackingParams = nvcv::GetParams(in.packing());
    int  inNumBits       = 0;
    for (const auto &numBits : inPackingParams.bits)
    {
        if (numBits > 0)
        {
            inNumBits = (inNumBits == 0) ? numBits : inNumBits;
            if (numBits != inNumBits)
            {
                throw CvtColorError("Invalid input format, all channels must have the same bit-depth");
            }
        }
    }
    auto outPackingParams = nvcv::GetParams(outFormat.planePacking(0));
    for (auto &numBits : outPackingParams.bits)
    {
        numBits = (numBits > 0) ? inNumBits : numBits;
    }
    auto outPacking = nvcv::MakePacking(outPackingParams);
    outFormat = outFormat.swizzleAndPacking(outFormat.swizzle(), outPacking, nvcv::Packing::NONE, nvcv::Packing::NONE,
                                            nvcv::Packing::NONE);

    return outFormat;
}

int64_t GetOutputHeight(int64_t height, NVCVColorConversionCode code)
{
    switch (code)
    {
    case NVCVColorConversionCode::NVCV_COLOR_YUV2RGB_NV12:
    case NVCVColorConversionCode::NVCV_COLOR_YUV2BGR_NV12:
    case NVCVColorConversionCode::NVCV_COLOR_YUV2RGB_NV21:
    case NVCVColorConversionCode::NVCV_COLOR_YUV2BGR_NV21:
        return (2 * height) / 3; // output height must be 2/3 of input height from NV12 or NV21

    case NVCVColorConversionCode::NVCV_COLOR_RGB2YUV_NV12:
    case NVCVColorConversionCode::NVCV_COLOR_BGR2YUV_NV12:
    case NVCVColorConversionCode::NVCV_COLOR_RGB2YUV_NV21:
    case NVCVColorConversionCode::NVCV_COLOR_BGR2YUV_NV21:
        return (3 * height) / 2; // output height must be 3/2 of input height for UV plane

    default:
        return height;
    }
}

nvcv::TensorShape GetOutputTensorShape(const nvcv::TensorShape &inputShape, nvcv::ImageFormat outputFormat,
                                       NVCVColorConversionCode code)
{
    if (inputShape.rank() < 3 || inputShape.rank() > 4)
    {
        throw CvtColorError("Invalid input tensor shape, only NHWC, HWC, NCHW, or CHW are supported");
    }

    std::array<int64_t, 4> outputShape  = {};
    auto                   layout       = inputShape.layout();
    int                    heightIndex  = layout.find('H');
    int                    channelIndex = layout.find('C');
    if (heightIndex < 0 || channelIndex < 0)
    {
        throw CvtColorError("Invalid input tensor shape, layout must contain H and C axes");
    }

    for (int i = 0; i < inputShape.rank(); i++)
    {
        outputShape[i] = inputShape[i];
    }

    outputShape[heightIndex]  = GetOutputHeight(outputShape[heightIndex], code);
    outputShape[channelIndex] = outputFormat.numChannels();

    return nvcv::TensorShape(outputShape.data(), inputShape.rank(), layout);
}
