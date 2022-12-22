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

#include "Operators.hpp"

#include <common/PyUtil.hpp>
#include <common/String.hpp>
#include <cvcuda/OpCvtColor.hpp>
#include <cvcuda/Types.h>
#include <nvcv/cuda/TypeTraits.hpp>
#include <nvcv/python/ImageBatchVarShape.hpp>
#include <nvcv/python/ResourceGuard.hpp>
#include <nvcv/python/Stream.hpp>
#include <nvcv/python/Tensor.hpp>
#include <pybind11/stl.h>

#include <map>

namespace cvcudapy {

namespace {

// All commented mappings below "// {...}" are not implemented in legacy code
// All commented codes below "//NVCV..." do not have clear output format

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
 //    {      NVCV_COLOR_GRAY2BGRA, NVCV_IMAGE_FORMAT_BGRA8},
  //    {      NVCV_COLOR_GRAY2RGBA, NVCV_IMAGE_FORMAT_RGBA8},
  //    {      NVCV_COLOR_BGRA2GRAY, NVCV_IMAGE_FORMAT_Y8_ER},
  //    {      NVCV_COLOR_RGBA2GRAY, NVCV_IMAGE_FORMAT_Y8_ER},
  //NVCV_COLOR_BGR2BGR565
  //NVCV_COLOR_RGB2BGR565
  //    {     NVCV_COLOR_BGR5652BGR,  NVCV_IMAGE_FORMAT_BGR8},
  //    {     NVCV_COLOR_BGR5652RGB,  NVCV_IMAGE_FORMAT_RGB8},
  //NVCV_COLOR_BGRA2BGR565
  //NVCV_COLOR_RGBA2BGR565
  //    {    NVCV_COLOR_BGR5652BGRA, NVCV_IMAGE_FORMAT_BGRA8},
  //    {    NVCV_COLOR_BGR5652RGBA, NVCV_IMAGE_FORMAT_RGBA8},
  //NVCV_COLOR_GRAY2BGR565
  //    {    NVCV_COLOR_BGR5652GRAY, NVCV_IMAGE_FORMAT_Y8_ER},
  //NVCV_COLOR_BGR2BGR555
  //NVCV_COLOR_RGB2BGR555
  //    {     NVCV_COLOR_BGR5552BGR,  NVCV_IMAGE_FORMAT_BGR8},
  //    {     NVCV_COLOR_BGR5552RGB,  NVCV_IMAGE_FORMAT_RGB8},
  //NVCV_COLOR_BGRA2BGR555
  //NVCV_COLOR_RGBA2BGR555
  //    {    NVCV_COLOR_BGR5552BGRA, NVCV_IMAGE_FORMAT_BGRA8},
  //    {    NVCV_COLOR_BGR5552RGBA, NVCV_IMAGE_FORMAT_RGBA8},
  //NVCV_COLOR_GRAY2BGR555
  //    {    NVCV_COLOR_BGR5552GRAY, NVCV_IMAGE_FORMAT_Y8_ER},
  //NVCV_COLOR_BGR2XYZ
  //NVCV_COLOR_RGB2XYZ
  //    {        NVCV_COLOR_XYZ2BGR,  NVCV_IMAGE_FORMAT_BGR8},
  //    {        NVCV_COLOR_XYZ2RGB,  NVCV_IMAGE_FORMAT_RGB8},
  //NVCV_COLOR_BGR2YCrCb
  //NVCV_COLOR_RGB2YCrCb
  //    {      NVCV_COLOR_YCrCb2BGR,  NVCV_IMAGE_FORMAT_BGR8},
  //    {      NVCV_COLOR_YCrCb2RGB,  NVCV_IMAGE_FORMAT_RGB8},
    {      NVCV_COLOR_BGR2HSV,  NVCV_IMAGE_FORMAT_HSV8},
    {      NVCV_COLOR_RGB2HSV,  NVCV_IMAGE_FORMAT_HSV8},
 //NVCV_COLOR_BGR2Lab
  //NVCV_COLOR_RGB2Lab
  //NVCV_COLOR_BGR2Luv
  //NVCV_COLOR_RGB2Luv
  //NVCV_COLOR_BGR2HLS
  //NVCV_COLOR_RGB2HLS
    {      NVCV_COLOR_HSV2BGR,  NVCV_IMAGE_FORMAT_BGR8},
    {      NVCV_COLOR_HSV2RGB,  NVCV_IMAGE_FORMAT_RGB8},
 //    {        NVCV_COLOR_Lab2BGR,  NVCV_IMAGE_FORMAT_BGR8},
  //    {        NVCV_COLOR_Lab2RGB,  NVCV_IMAGE_FORMAT_RGB8},
  //    {        NVCV_COLOR_Luv2BGR,  NVCV_IMAGE_FORMAT_BGR8},
  //    {        NVCV_COLOR_Luv2RGB,  NVCV_IMAGE_FORMAT_RGB8},
  //    {        NVCV_COLOR_HLS2BGR,  NVCV_IMAGE_FORMAT_BGR8},
  //    {        NVCV_COLOR_HLS2RGB,  NVCV_IMAGE_FORMAT_RGB8},
    { NVCV_COLOR_BGR2HSV_FULL,  NVCV_IMAGE_FORMAT_HSV8},
    { NVCV_COLOR_RGB2HSV_FULL,  NVCV_IMAGE_FORMAT_HSV8},
 //NVCV_COLOR_BGR2HLS_FULL
  //NVCV_COLOR_RGB2HLS_FULL
    { NVCV_COLOR_HSV2BGR_FULL,  NVCV_IMAGE_FORMAT_BGR8},
    { NVCV_COLOR_HSV2RGB_FULL,  NVCV_IMAGE_FORMAT_RGB8},
 //    {   NVCV_COLOR_HLS2BGR_FULL,  NVCV_IMAGE_FORMAT_BGR8},
  //    {   NVCV_COLOR_HLS2RGB_FULL,  NVCV_IMAGE_FORMAT_RGB8},
  //NVCV_COLOR_LBGR2Lab
  //NVCV_COLOR_LRGB2Lab
  //NVCV_COLOR_LBGR2Luv
  //NVCV_COLOR_LRGB2Luv
  //    {       NVCV_COLOR_Lab2LBGR,  NVCV_IMAGE_FORMAT_BGR8},
  //    {       NVCV_COLOR_Lab2LRGB,  NVCV_IMAGE_FORMAT_RGB8},
  //    {       NVCV_COLOR_Luv2LBGR,  NVCV_IMAGE_FORMAT_BGR8},
  //    {       NVCV_COLOR_Luv2LRGB,  NVCV_IMAGE_FORMAT_RGB8},
  //NVCV_COLOR_BGR2YUV
  //NVCV_COLOR_RGB2YUV
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
 //    {     NVCV_COLOR_RGBA2mRGBA, NVCV_IMAGE_FORMAT_RGBA8},
  //    {     NVCV_COLOR_mRGBA2RGBA, NVCV_IMAGE_FORMAT_RGBA8},
  //NVCV_COLOR_RGB2YUV_I420
  //NVCV_COLOR_BGR2YUV_I420
  //NVCV_COLOR_RGB2YUV_IYUV
  //NVCV_COLOR_BGR2YUV_IYUV
  //NVCV_COLOR_RGBA2YUV_I420
  //NVCV_COLOR_BGRA2YUV_I420
  //NVCV_COLOR_RGBA2YUV_IYUV
  //NVCV_COLOR_BGRA2YUV_IYUV
  //NVCV_COLOR_RGB2YUV_YV12
  //NVCV_COLOR_BGR2YUV_YV12
  //NVCV_COLOR_RGBA2YUV_YV12
  //NVCV_COLOR_BGRA2YUV_YV12
  //    {    NVCV_COLOR_BayerBG2BGR,  NVCV_IMAGE_FORMAT_BGR8},
  //    {    NVCV_COLOR_BayerGB2BGR,  NVCV_IMAGE_FORMAT_BGR8},
  //    {    NVCV_COLOR_BayerRG2BGR,  NVCV_IMAGE_FORMAT_BGR8},
  //    {    NVCV_COLOR_BayerGR2BGR,  NVCV_IMAGE_FORMAT_BGR8},
  //    {    NVCV_COLOR_BayerBG2RGB,  NVCV_IMAGE_FORMAT_RGB8},
  //   {    NVCV_COLOR_BayerGB2RGB,  NVCV_IMAGE_FORMAT_RGB8},
  //    {    NVCV_COLOR_BayerRG2RGB,  NVCV_IMAGE_FORMAT_RGB8},
  //    {    NVCV_COLOR_BayerGR2RGB,  NVCV_IMAGE_FORMAT_RGB8},
  //    {   NVCV_COLOR_BayerBG2GRAY, NVCV_IMAGE_FORMAT_Y8_ER},
  //    {   NVCV_COLOR_BayerGB2GRAY, NVCV_IMAGE_FORMAT_Y8_ER},
  //    {   NVCV_COLOR_BayerRG2GRAY, NVCV_IMAGE_FORMAT_Y8_ER},
  //    {   NVCV_COLOR_BayerGR2GRAY, NVCV_IMAGE_FORMAT_Y8_ER},
  //    {NVCV_COLOR_BayerBG2BGR_VNG,  NVCV_IMAGE_FORMAT_BGR8},
  //    {NVCV_COLOR_BayerGB2BGR_VNG,  NVCV_IMAGE_FORMAT_BGR8},
  //    {NVCV_COLOR_BayerRG2BGR_VNG,  NVCV_IMAGE_FORMAT_BGR8},
  //    {NVCV_COLOR_BayerGR2BGR_VNG,  NVCV_IMAGE_FORMAT_BGR8},
  //    {NVCV_COLOR_BayerBG2RGB_VNG,  NVCV_IMAGE_FORMAT_RGB8},
  //    {NVCV_COLOR_BayerGB2RGB_VNG,  NVCV_IMAGE_FORMAT_RGB8},
  //    {NVCV_COLOR_BayerRG2RGB_VNG,  NVCV_IMAGE_FORMAT_RGB8},
  //    {NVCV_COLOR_BayerGR2RGB_VNG,  NVCV_IMAGE_FORMAT_RGB8},
  //    { NVCV_COLOR_BayerBG2BGR_EA,  NVCV_IMAGE_FORMAT_BGR8},
  //    { NVCV_COLOR_BayerGB2BGR_EA,  NVCV_IMAGE_FORMAT_BGR8},
  //    { NVCV_COLOR_BayerRG2BGR_EA,  NVCV_IMAGE_FORMAT_BGR8},
  //    { NVCV_COLOR_BayerGR2BGR_EA,  NVCV_IMAGE_FORMAT_BGR8},
  //    { NVCV_COLOR_BayerBG2RGB_EA,  NVCV_IMAGE_FORMAT_RGB8},
  //    { NVCV_COLOR_BayerGB2RGB_EA,  NVCV_IMAGE_FORMAT_RGB8},
  //    { NVCV_COLOR_BayerRG2RGB_EA,  NVCV_IMAGE_FORMAT_RGB8},
  //    { NVCV_COLOR_BayerGR2RGB_EA,  NVCV_IMAGE_FORMAT_RGB8},
  //NVCV_COLOR_COLORCVT_MAX
    { NVCV_COLOR_RGB2YUV_NV12,  NVCV_IMAGE_FORMAT_NV12},
    { NVCV_COLOR_BGR2YUV_NV12,  NVCV_IMAGE_FORMAT_NV12},
 //NVCV_COLOR_RGB2YUV_NV21
  //NVCV_COLOR_RGB2YUV420sp
  //NVCV_COLOR_BGR2YUV_NV21
  //NVCV_COLOR_BGR2YUV420sp
    {NVCV_COLOR_RGBA2YUV_NV12,  NVCV_IMAGE_FORMAT_NV12},
    {NVCV_COLOR_BGRA2YUV_NV12,  NVCV_IMAGE_FORMAT_NV12},
 //NVCV_COLOR_RGBA2YUV_NV21
  //NVCV_COLOR_RGBA2YUV420sp
  //NVCV_COLOR_BGRA2YUV_NV21
  //NVCV_COLOR_BGRA2YUV420sp
  //NVCV_COLORCVT_MAX = 148,
};

nvcv::ImageFormat GetOutputFormat(nvcv::DataType in, NVCVColorConversionCode code)
{
    auto outFormatIt = kOutputFormat.find(code);
    if (outFormatIt == kOutputFormat.end())
    {
        throw std::runtime_error("Invalid color conversion code");
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
                throw std::runtime_error("Invalid input format, all channels must have the same bit-depth");
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

Tensor CvtColorInto(Tensor &output, Tensor &input, NVCVColorConversionCode code, std::optional<Stream> pstream)
{
    if (!pstream)
    {
        pstream = Stream::Current();
    }

    auto cvtColor = CreateOperator<cvcuda::CvtColor>();

    ResourceGuard guard(*pstream);
    guard.add(LockMode::LOCK_READ, {input});
    guard.add(LockMode::LOCK_WRITE, {output});
    guard.add(LockMode::LOCK_NONE, {*cvtColor});

    cvtColor->submit(pstream->cudaHandle(), input, output, code);

    return output;
}

Tensor CvtColor(Tensor &input, NVCVColorConversionCode code, std::optional<Stream> pstream)
{
    auto outFormat = GetOutputFormat(input.dtype(), code);

    if (input.shape().size() < 3)
    {
        throw std::runtime_error("Invalid input tensor shape");
    }
    int          numImgs{static_cast<int>(input.shape()[0])};
    nvcv::Size2D size{static_cast<int>(input.shape()[2]), static_cast<int>(input.shape()[1])};

    Tensor output = Tensor::CreateForImageBatch(numImgs, size, outFormat);

    return CvtColorInto(output, input, code, pstream);
}

ImageBatchVarShape CvtColorVarShapeInto(ImageBatchVarShape &output, ImageBatchVarShape &input,
                                        NVCVColorConversionCode code, std::optional<Stream> pstream)
{
    if (!pstream)
    {
        pstream = Stream::Current();
    }

    auto cvtColor = CreateOperator<cvcuda::CvtColor>();

    ResourceGuard guard(*pstream);
    guard.add(LockMode::LOCK_READ, {input});
    guard.add(LockMode::LOCK_WRITE, {output});
    guard.add(LockMode::LOCK_NONE, {*cvtColor});

    cvtColor->submit(pstream->cudaHandle(), input, output, code);

    return output;
}

ImageBatchVarShape CvtColorVarShape(ImageBatchVarShape &input, NVCVColorConversionCode code,
                                    std::optional<Stream> pstream)
{
    auto inFormat = input.uniqueFormat();
    if (!inFormat || inFormat.numPlanes() != 1)
    {
        throw std::runtime_error("All images in input must have the same single-plane format");
    }
    auto outFormat = GetOutputFormat(inFormat.planeDataType(0), code);

    ImageBatchVarShape output = ImageBatchVarShape::Create(input.capacity());

    for (int i = 0; i < input.numImages(); ++i)
    {
        nvcv::Size2D size = input[i].size();

        auto img = Image::Create(size, outFormat);
        output.pushBack(img);
    }

    return CvtColorVarShapeInto(output, input, code, pstream);
}

} // namespace

void ExportOpCvtColor(py::module &m)
{
    using namespace pybind11::literals;

    m.def("cvtcolor", &CvtColor, "src"_a, "code"_a, py::kw_only(), "stream"_a = nullptr);
    m.def("cvtcolor_into", &CvtColorInto, "dst"_a, "src"_a, "code"_a, py::kw_only(), "stream"_a = nullptr);

    m.def("cvtcolor", &CvtColorVarShape, "src"_a, "code"_a, py::kw_only(), "stream"_a = nullptr);
    m.def("cvtcolor_into", &CvtColorVarShapeInto, "dst"_a, "src"_a, "code"_a, py::kw_only(), "stream"_a = nullptr);
}

} // namespace cvcudapy
