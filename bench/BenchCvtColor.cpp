/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "BenchUtils.hpp"

#include <cvcuda/OpCvtColor.hpp>

#include <nvbench/nvbench.cuh>

#include <map>
#include <stdexcept>
#include <tuple>

using ConvCodeToFormat = std::tuple<NVCVColorConversionCode, NVCVImageFormat, NVCVImageFormat>;
using CodeMap          = std::map<std::string, ConvCodeToFormat>;

inline static ConvCodeToFormat str2Frmt(const std::string &str)
{
    // clang-format off
    static const CodeMap codeMap {
        {     "RGB2BGR", {NVCV_COLOR_RGB2BGR,      NVCV_IMAGE_FORMAT_RGB8,  NVCV_IMAGE_FORMAT_BGR8 }},
        {    "RGB2RGBA", {NVCV_COLOR_RGB2RGBA,     NVCV_IMAGE_FORMAT_RGB8,  NVCV_IMAGE_FORMAT_RGBA8}},
        {    "RGBA2RGB", {NVCV_COLOR_RGBA2RGB,     NVCV_IMAGE_FORMAT_RGBA8, NVCV_IMAGE_FORMAT_RGB8 }},
        {    "RGB2GRAY", {NVCV_COLOR_RGB2GRAY,     NVCV_IMAGE_FORMAT_RGB8,  NVCV_IMAGE_FORMAT_Y8   }},
        {    "GRAY2RGB", {NVCV_COLOR_GRAY2RGB,     NVCV_IMAGE_FORMAT_Y8,    NVCV_IMAGE_FORMAT_RGB8 }},
        {     "RGB2HSV", {NVCV_COLOR_RGB2HSV,      NVCV_IMAGE_FORMAT_RGB8,  NVCV_IMAGE_FORMAT_HSV8 }},
        {     "HSV2RGB", {NVCV_COLOR_HSV2RGB,      NVCV_IMAGE_FORMAT_HSV8,  NVCV_IMAGE_FORMAT_RGB8 }},
        {     "RGB2YUV", {NVCV_COLOR_RGB2YUV,      NVCV_IMAGE_FORMAT_RGB8,  NVCV_IMAGE_FORMAT_YUV8 }},
        {     "YUV2RGB", {NVCV_COLOR_YUV2RGB,      NVCV_IMAGE_FORMAT_YUV8,  NVCV_IMAGE_FORMAT_RGB8 }},
        {"RGB2YUV_NV12", {NVCV_COLOR_RGB2YUV_NV12, NVCV_IMAGE_FORMAT_RGB8,  NVCV_IMAGE_FORMAT_NV12 }},
        {"YUV2RGB_NV12", {NVCV_COLOR_YUV2RGB_NV12, NVCV_IMAGE_FORMAT_NV12,  NVCV_IMAGE_FORMAT_RGB8 }},
    };
    // clang-format on

    if (auto it = codeMap.find(str); it != codeMap.end())
    {
        return it->second;
    }
    else
    {
        throw std::invalid_argument("Unrecognized color code");
    }
}

template<typename BT>
inline float bytesPerPixel(NVCVImageFormat imgFormat)
{
#define BPP_CASE(frmt, bytes) \
    case frmt:                \
        return bytes * sizeof(BT)

    switch (imgFormat)
    {
        BPP_CASE(NVCV_IMAGE_FORMAT_RGB8, 3);
        BPP_CASE(NVCV_IMAGE_FORMAT_BGR8, 3);
        BPP_CASE(NVCV_IMAGE_FORMAT_HSV8, 3);
        BPP_CASE(NVCV_IMAGE_FORMAT_RGBA8, 4);
        BPP_CASE(NVCV_IMAGE_FORMAT_YUV8, 3);
        BPP_CASE(NVCV_IMAGE_FORMAT_NV12, 1.5f);
        BPP_CASE(NVCV_IMAGE_FORMAT_Y8, 1);
    default:
        throw std::invalid_argument("Unrecognized format");
    }
#undef BPP_CASE
}

// Adapted from src/util/TensorDataUtils.hpp
inline static nvcv::Tensor CreateTensor(int numImages, int imgWidth, int imgHeight, const nvcv::ImageFormat &imgFormat)
{
    if (imgFormat == NVCV_IMAGE_FORMAT_NV12 || imgFormat == NVCV_IMAGE_FORMAT_NV12_ER
        || imgFormat == NVCV_IMAGE_FORMAT_NV21 || imgFormat == NVCV_IMAGE_FORMAT_NV21_ER)
    {
        if (imgHeight % 2 != 0 || imgWidth % 2 != 0)
        {
            throw std::invalid_argument("Invalid height");
        }

        int height420 = (imgHeight * 3) / 2;

        return nvcv::Tensor(numImages, {imgWidth, height420}, nvcv::ImageFormat(NVCV_IMAGE_FORMAT_Y8));
    }
    else
    {
        return nvcv::Tensor(numImages, {imgWidth, imgHeight}, imgFormat);
    }
}

template<typename BT>
inline void CvtColor(nvbench::state &state, nvbench::type_list<BT>)
try
{
    long3 shape    = benchutils::GetShape<3>(state.get_string("shape"));
    long  varShape = state.get_int64("varShape");

    ConvCodeToFormat formats = str2Frmt(state.get_string("code"));

    NVCVColorConversionCode code = std::get<0>(formats);
    nvcv::ImageFormat       inFormat{std::get<1>(formats)};
    nvcv::ImageFormat       outFormat{std::get<2>(formats)};

    state.add_global_memory_reads(shape.x * shape.y * shape.z * bytesPerPixel<BT>(inFormat));
    state.add_global_memory_writes(shape.x * shape.y * shape.z * bytesPerPixel<BT>(outFormat));

    cvcuda::CvtColor op;

    if (varShape < 0) // negative var shape means use Tensor
    {
        nvcv::Tensor src = CreateTensor(shape.x, shape.z, shape.y, inFormat);
        nvcv::Tensor dst = CreateTensor(shape.x, shape.z, shape.y, outFormat);

        benchutils::FillTensor<BT>(src, benchutils::RandomValues<BT>());

        state.exec(nvbench::exec_tag::sync,
                   [&op, &src, &dst, &code](nvbench::launch &launch) { op(launch.get_stream(), src, dst, code); });
    }
    else // zero and positive var shape means use ImageBatchVarShape
    {
        if (inFormat.chromaSubsampling() != nvcv::ChromaSubsampling::CSS_444
            || outFormat.chromaSubsampling() != nvcv::ChromaSubsampling::CSS_444)
        {
            state.skip("Skipping formats that have subsampled planes for the varshape benchmark");
        }

        std::vector<nvcv::Image>          imgSrc;
        std::vector<nvcv::Image>          imgDst;
        nvcv::ImageBatchVarShape          src(shape.x);
        nvcv::ImageBatchVarShape          dst(shape.x);
        std::vector<std::vector<uint8_t>> srcVec(shape.x);

        auto randomValuesU8 = benchutils::RandomValues<uint8_t>();

        for (int i = 0; i < shape.x; i++)
        {
            imgSrc.emplace_back(nvcv::Size2D{(int)shape.z, (int)shape.y}, inFormat);
            imgDst.emplace_back(nvcv::Size2D{(int)shape.z, (int)shape.y}, outFormat);

            int srcRowStride = imgSrc[i].size().w * inFormat.planePixelStrideBytes(0);
            int srcBufSize   = imgSrc[i].size().h * srcRowStride;
            srcVec[i].resize(srcBufSize);
            for (int idx = 0; idx < srcBufSize; idx++)
            {
                srcVec[i][idx] = randomValuesU8();
            }

            auto imgData = imgSrc[i].exportData<nvcv::ImageDataStridedCuda>();
            CUDA_CHECK_ERROR(cudaMemcpy2D(imgData->plane(0).basePtr, imgData->plane(0).rowStride, srcVec[i].data(),
                                          srcRowStride, srcRowStride, imgSrc[i].size().h, cudaMemcpyHostToDevice));
        }
        src.pushBack(imgSrc.begin(), imgSrc.end());
        dst.pushBack(imgDst.begin(), imgDst.end());

        state.exec(nvbench::exec_tag::sync,
                   [&op, &src, &dst, &code](nvbench::launch &launch) { op(launch.get_stream(), src, dst, code); });
    }
}

catch (const std::exception &err)
{
    state.skip(err.what());
}

using BaseTypes = nvbench::type_list<uint8_t>;

NVBENCH_BENCH_TYPES(CvtColor, NVBENCH_TYPE_AXES(BaseTypes))
    .set_type_axes_names({"BaseType"})
    .add_string_axis("shape", {"1x1080x1920", "64x720x1280"})
    .add_string_axis("code", {"RGB2BGR", "RGB2RGBA", "RGBA2RGB", "RGB2GRAY", "GRAY2RGB", "RGB2HSV", "HSV2RGB",
                              "RGB2YUV", "YUV2RGB", "RGB2YUV_NV12", "YUV2RGB_NV12"})
    .add_int64_axis("varShape", {-1, 0});
