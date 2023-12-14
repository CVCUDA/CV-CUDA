/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "Definitions.hpp"

#include <common/ValueTests.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorDataAccess.hpp>
#include <stdint.h>
#include <util/TensorDataUtils.hpp>

#include <iostream>
#include <random>

namespace gt   = ::testing;
namespace test = nvcv::test;
namespace util = nvcv::util;

// clang-format off
NVCV_TEST_SUITE_P(TensorDataUtils, test::ValueList<int, int, int, uint8_t, nvcv::ImageFormat>
{
    //width, height, numImages, fill byte, format
    {     2,      2,         2,         2, nvcv::FMT_RGB8},
    {     3,      3,         5,         2, nvcv::FMT_BGR8},
    {     10,    11,         2,         2, nvcv::FMT_RGBA8},
    {     5,      5,         1,         2, nvcv::FMT_BGRA8},
    {     2,      4,         2,         2, nvcv::FMT_BGR8p},
    {     4,      9,         2,         2, nvcv::FMT_RGB8p},
    {     43,     1,         4,         2, nvcv::FMT_RGBA8p},
    {     4,      42,        3,         2, nvcv::FMT_BGRA8p},
    {     2,      2,         2,         2, nvcv::FMT_U32},
    {     4,      2,         2,         2, nvcv::FMT_RGBf32},
    {     6,      6,         2,         2, nvcv::FMT_BGRf32},
    {     5,      7,         5,         2, nvcv::FMT_RGBAf32},
    {     2,      2,         2,         2, nvcv::FMT_BGRAf32},
    {     7,      8,         2,         2, nvcv::FMT_RGBf32p},
    {     2,      2,         1,         2, nvcv::FMT_BGRf32p},
    {     6,      3,         2,         2, nvcv::FMT_RGBAf32p},
    {     2,      2,         2,         2, nvcv::FMT_BGRAf32p},
    {     2,      7,         2,         2, nvcv::FMT_F64},
    {     3,      5,         3,         2, nvcv::FMT_RGBA8},
});

// clang-format on

template<typename DT>
static void compareTensor(nvcv::Tensor &tensor, DT fillVal)
{
    auto            ac       = nvcv::TensorDataAccessStrided::Create(tensor.exportData());
    int             elements = ac->sampleStride() / sizeof(DT);
    std::vector<DT> goldVec(elements, static_cast<DT>(fillVal));
    for (int i = 0; i < ac->numSamples(); ++i)
    {
        std::vector<DT> readVec(elements);
        if (cudaMemcpy(readVec.data(), ac->sampleData(i), ac->sampleStride(), cudaMemcpyDeviceToHost) != cudaSuccess)
            throw std::runtime_error("CudaMemcpy failed");
        if (goldVec != readVec)
            throw std::runtime_error("Vectors not equal");
    }
    return;
}

// This function will set the provided tensor (including any strides) to random values, then read it back and compare using
// TensorDataUtils::SetXXXFromVector, GetXXXFromTensor functions
template<typename DT>
static void GetSetTensor(const nvcv::Tensor &tensor)
{
    auto                          ac          = nvcv::TensorDataAccessStrided::Create(tensor.exportData());
    int                           numElements = ac->sampleStride() / sizeof(DT);
    std::vector<DT>               vec(numElements);
    std::default_random_engine    rng;
    std::uniform_int_distribution rand;
    generate(vec.begin(), vec.end(), [&rng, &rand] { return rand(rng); });

    std::vector<DT> vecOut(numElements, 0);
    util::SetTensorFromVector<DT>(tensor.exportData(), vec);
    util::GetVectorFromTensor<DT>(tensor.exportData(), 0, vecOut);
    if (vec != vecOut)
        throw std::runtime_error("Vectors not equal");

    return;
}

// This function will set the provided tensor image area to random values, then read it back and compare using
// TensorDataUtils::SetXXXFromVector, GetXXXFromTensor functions
template<typename DT>
static void setGetTensorImage(const nvcv::Tensor &tensor)
{
    auto                          ac              = nvcv::TensorDataAccessStridedImage::Create(tensor.exportData());
    int                           numElements     = ac->numCols() * ac->numRows() * ac->numChannels();
    int                           numElementsFull = ac->sampleStride() / sizeof(DT);
    std::vector<DT>               vec(numElements, 0);
    std::default_random_engine    rng;
    std::uniform_int_distribution rand;
    generate(vec.begin(), vec.end(), [&rng, &rand] { return rand(rng); });
    std::vector<DT> vecOutImage(numElements, 0);

    std::vector<DT> vecImageOriginal(numElements, 0);

    std::vector<DT> vecOriginal(numElementsFull, 0);
    std::vector<DT> vecOutFull(numElementsFull, 0);

    // copy the original data out
    util::GetImageVectorFromTensor<DT>(tensor.exportData(), 0, vecImageOriginal);
    util::GetVectorFromTensor<DT>(tensor.exportData(), 0, vecOriginal);

    // set the image data
    util::SetImageTensorFromVector<DT>(tensor.exportData(), vec);
    util::GetImageVectorFromTensor<DT>(tensor.exportData(), 0, vecOutImage);

    if (vec != vecOutImage)
        throw std::runtime_error(
            "Vectors not equal vector out image does note contain the same data as the input vector");

    util::SetImageTensorFromVector<DT>(tensor.exportData(), vecImageOriginal);
    util::GetVectorFromTensor<DT>(tensor.exportData(), 0, vecOutFull);

    if (vecOriginal != vecOutFull)
        throw std::runtime_error("Vectors not equal, vector not restored to original state");

    return;
}

template<typename DT>
static void checkRndRange(const nvcv::Tensor &tensor, DT lowBound, DT highBound)
{
    auto tDataAc = nvcv::TensorDataAccessStrided::Create(tensor.exportData());

    for (int sample = 0; sample < tDataAc->numSamples(); sample++)
    {
        util::TensorImageData img(tensor.exportData(), sample);

        for (int x = 0; x < img.size().w; x++)
            for (int y = 0; y < img.size().h; y++)
                for (int c = 0; c < img.numC(); c++)
                {
                    DT value = *img.item<DT>(x, y, c);
                    //note floats are [a,b), while ints are (a,b) but this should be sufficient
                    if (value < lowBound || value > highBound)
                    {
                        throw std::runtime_error("Value out of bounds");
                    }
                }
    }
    return;
}

TEST_P(TensorDataUtils, SetTensorTo)
{
    int               width   = GetParamValue<0>();
    int               height  = GetParamValue<1>();
    int               number  = GetParamValue<2>();
    uint8_t           fillVal = GetParamValue<3>();
    nvcv::ImageFormat fmt     = GetParamValue<4>();

    nvcv::Tensor tensor(number, {width, height}, fmt);

    EXPECT_NO_THROW(util::SetTensorTo<uint8_t>(tensor.exportData(), fillVal));
    EXPECT_NO_THROW(compareTensor<uint8_t>(tensor, (uint8_t)fillVal));

    EXPECT_NO_THROW(util::SetTensorTo<uint16_t>(tensor.exportData(), fillVal));
    EXPECT_NO_THROW(compareTensor<uint16_t>(tensor, (uint16_t)fillVal));

    EXPECT_NO_THROW(util::SetTensorTo<int>(tensor.exportData(), fillVal));
    EXPECT_NO_THROW(compareTensor<int>(tensor, (int)fillVal));

    EXPECT_NO_THROW(util::SetTensorTo<float>(tensor.exportData(), fillVal));
    EXPECT_NO_THROW(compareTensor<float>(tensor, (float)fillVal));
}

TEST_P(TensorDataUtils, SetTensorToRandom)
{
    int width  = GetParamValue<0>();
    int height = GetParamValue<1>();
    int number = GetParamValue<2>();

    nvcv::Tensor tensor(number, {width, height}, nvcv::FMT_RGBA8);
    nvcv::Tensor tensorFloat(number, {width, height}, nvcv::FMT_RGBAf32p);

    EXPECT_NO_THROW(util::SetTensorToRandomValue<uint8_t>(tensor.exportData(), 0, 0xFF));
    EXPECT_NO_THROW(checkRndRange<uint8_t>(tensor, 0, 0xFF));

    EXPECT_NO_THROW(util::SetTensorToRandomValue<uint8_t>(tensor.exportData(), 0x05, 0x11));
    EXPECT_NO_THROW(checkRndRange<uint8_t>(tensor, 0x05, 0x11));

    EXPECT_NO_THROW(util::SetTensorToRandomValue<float>(tensorFloat.exportData(), .01f, 1.0f));
    EXPECT_NO_THROW(checkRndRange<float>(tensorFloat, .01f, 1.0f));
}

TEST_P(TensorDataUtils, SetGetTensorFromVector)
{
    int               width  = GetParamValue<0>();
    int               height = GetParamValue<1>();
    int               number = GetParamValue<2>();
    nvcv::ImageFormat fmt    = GetParamValue<4>();

    nvcv::Tensor tensor(number, {width, height}, fmt);
    EXPECT_NO_THROW(GetSetTensor<uint8_t>(tensor));
    EXPECT_NO_THROW(GetSetTensor<uint16_t>(tensor));
    EXPECT_NO_THROW(GetSetTensor<int>(tensor));
    EXPECT_NO_THROW(GetSetTensor<float>(tensor));
}

TEST_P(TensorDataUtils, SetGetTensorFromImageVector)
{
    int               width  = GetParamValue<0>();
    int               height = GetParamValue<1>();
    int               number = GetParamValue<2>();
    nvcv::ImageFormat fmt    = GetParamValue<4>();

    nvcv::Tensor tensor(number, {width, height}, fmt);
    // Just put in some random data
    EXPECT_NO_THROW(util::SetTensorToRandomValue<uint8_t>(tensor.exportData(), 0, 0xFF));
    switch (fmt)
    {
    case nvcv::FMT_BGR8:
    case nvcv::FMT_RGBA8:
    case nvcv::FMT_BGRA8:
    case nvcv::FMT_BGR8p:
    case nvcv::FMT_RGB8p:
    case nvcv::FMT_RGBA8p:
    case nvcv::FMT_BGRA8p:
    {
        EXPECT_NO_THROW(setGetTensorImage<uint8_t>(tensor));
        break;
    }

    case nvcv::FMT_U32:
    {
        EXPECT_NO_THROW(setGetTensorImage<uint32_t>(tensor));
        break;
    }

    case nvcv::FMT_RGBf32:
    case nvcv::FMT_BGRf32:
    case nvcv::FMT_RGBAf32:
    case nvcv::FMT_BGRAf32:
    case nvcv::FMT_RGBf32p:
    case nvcv::FMT_BGRf32p:
    case nvcv::FMT_RGBAf32p:
    case nvcv::FMT_BGRAf32p:
    {
        EXPECT_NO_THROW(setGetTensorImage<float>(tensor));
        break;
    }

    case nvcv::FMT_F64:

    {
        EXPECT_NO_THROW(setGetTensorImage<double>(tensor));
        break;
    }

    default:

        break;
    }
}

TEST(TensorDataUtils, SanityCvImageData)
{
    int width  = 10;
    int height = 20;
    int number = 2;

    nvcv::Tensor tensor1(number, {width, height}, nvcv::FMT_RGBAf32p);
    nvcv::Tensor tensor2(number, {width, height}, nvcv::FMT_BGR8);

    EXPECT_NO_THROW(util::SetTensorTo<float>(tensor1.exportData(), 0.5f));
    EXPECT_NO_THROW(util::SetTensorTo<uint8_t>(tensor2.exportData(), 0xa0));
    EXPECT_NO_THROW(util::SetTensorTo<uint8_t>(tensor2.exportData(), 0xb0, 1));

    util::TensorImageData cvImage1(tensor1.exportData());
    util::TensorImageData cvImage2(tensor1.exportData(), 1);
    util::TensorImageData cvImage3(tensor2.exportData());
    util::TensorImageData cvImage4(tensor2.exportData(), 1);

    EXPECT_EQ(cvImage1, cvImage2);
    EXPECT_NE(cvImage2, cvImage3);
    EXPECT_NE(cvImage3, cvImage4);

    auto tDataAc1 = nvcv::TensorDataAccessStridedImagePlanar::Create(tensor1.exportData());
    EXPECT_EQ(tDataAc1->rowStride(), cvImage1.rowStride());
    EXPECT_EQ(tDataAc1->planeStride(), cvImage1.planeStride());
    EXPECT_EQ(cvImage1.size().w, width);
    EXPECT_EQ(cvImage1.size().h, height);
    EXPECT_EQ(cvImage1.bytesPerC(), 4);
    EXPECT_EQ(cvImage1.numC(), 4);
    EXPECT_EQ(cvImage4.bytesPerC(), 1);
    EXPECT_EQ(cvImage4.numC(), 3);
    EXPECT_EQ(cvImage1.imageCHW(), true);
    EXPECT_EQ(cvImage4.imageCHW(), false);
}

TEST(TensorDataUtils, SetCvImageData)
{
    int width  = 5;
    int height = 5;

    nvcv::Tensor tensor(1, {width, height}, nvcv::FMT_RGBA8);
    EXPECT_NO_THROW(util::SetTensorTo<uint8_t>(tensor.exportData(), 0xCA));
    util::TensorImageData cvTensor(tensor.exportData());

    nvcv::Size2D region = {width - 1, height - 1};
    EXPECT_NO_THROW(
        util::SetCvDataTo<uint8_t>(cvTensor, 0xFF, region, util::chflags::C0 | util::chflags::C2 | util::chflags::C3));

    uint8_t *dataPtr = cvTensor.getVector().data();
    //1st Col
    EXPECT_EQ(*dataPtr, 0xFF);
    dataPtr += sizeof(uint8_t);
    EXPECT_EQ(*dataPtr, 0xCA);
    dataPtr += sizeof(uint8_t);
    EXPECT_EQ(*dataPtr, 0xFF);
    dataPtr += sizeof(uint8_t);
    EXPECT_EQ(*dataPtr, 0xFF);

    //last col
    dataPtr = cvTensor.getVector().data() + (cvTensor.size().w - 1) * cvTensor.bytesPerC() * cvTensor.numC();
    EXPECT_EQ(*dataPtr, 0xCA);
    dataPtr += sizeof(uint8_t);
    EXPECT_EQ(*dataPtr, 0xCA);
    dataPtr += sizeof(uint8_t);
    EXPECT_EQ(*dataPtr, 0xCA);
    dataPtr += sizeof(uint8_t);
    EXPECT_EQ(*dataPtr, 0xCA);

    //last row
    dataPtr = cvTensor.getVector().data() + (cvTensor.size().h - 1) * cvTensor.rowStride();
    EXPECT_EQ(*dataPtr, 0xCA);
    dataPtr += sizeof(uint8_t);
    EXPECT_EQ(*dataPtr, 0xCA);
    dataPtr += sizeof(uint8_t);
    EXPECT_EQ(*dataPtr, 0xCA);
    dataPtr += sizeof(uint8_t);
    EXPECT_EQ(*dataPtr, 0xCA);

    EXPECT_EQ(*cvTensor.item<uint8_t>(0, 0, 0), 0xFF);
    EXPECT_EQ(*cvTensor.item<uint8_t>(0, 0, 1), 0xCA);
    EXPECT_EQ(*cvTensor.item<uint8_t>(0, 0, 2), 0xFF);
    EXPECT_EQ(*cvTensor.item<uint8_t>(0, 0, 3), 0xFF);

    EXPECT_EQ(*cvTensor.item<uint8_t>(1, 1, 0), 0xFF);
    EXPECT_EQ(*cvTensor.item<uint8_t>(1, 1, 1), 0xCA);
    EXPECT_EQ(*cvTensor.item<uint8_t>(1, 1, 2), 0xFF);
    EXPECT_EQ(*cvTensor.item<uint8_t>(1, 1, 3), 0xFF);

    EXPECT_EQ(*cvTensor.item<uint8_t>(width - 1, height - 1, 0), 0xCA);
    EXPECT_EQ(*cvTensor.item<uint8_t>(width - 1, height - 1, 1), 0xCA);
    EXPECT_EQ(*cvTensor.item<uint8_t>(width - 1, height - 1, 2), 0xCA);
    EXPECT_EQ(*cvTensor.item<uint8_t>(width - 1, height - 1, 3), 0xCA);
}

TEST(TensorDataUtils, SetCvImageDataP)
{
    int          width  = 5;
    int          height = 5;
    nvcv::Tensor tensor(1, {width, height}, nvcv::FMT_RGBAf32p);
    EXPECT_NO_THROW(util::SetTensorTo<float>(tensor.exportData(), 1.0f));
    util::TensorImageData cvTensorFp(tensor.exportData());
    nvcv::Size2D          region = {width - 1, height - 1};
    EXPECT_NO_THROW(
        util::SetCvDataTo<float>(cvTensorFp, .5f, region, util::chflags::C0 | util::chflags::C2 | util::chflags::C3));

    float *dataPtr = (float *)cvTensorFp.getVector().data();
    EXPECT_EQ(*dataPtr, .5f);
    EXPECT_EQ(*(dataPtr + cvTensorFp.planeStride() / sizeof(float)), 1.0f);
    EXPECT_EQ(*(dataPtr + 2 * cvTensorFp.planeStride() / sizeof(float)), .5f);
    EXPECT_EQ(*(dataPtr + 3 * cvTensorFp.planeStride() / sizeof(float)), .5f);

    // last col should be 1.0
    float *lastCol = (float *)(cvTensorFp.getVector().data() + (cvTensorFp.size().w - 1) * sizeof(float));
    EXPECT_EQ(*lastCol, 1.0f);
    EXPECT_EQ(*(lastCol + cvTensorFp.planeStride() / sizeof(float)), 1.0f);
    EXPECT_EQ(*(lastCol + 2 * cvTensorFp.planeStride() / sizeof(float)), 1.0f);
    EXPECT_EQ(*(lastCol + 3 * cvTensorFp.planeStride() / sizeof(float)), 1.0f);

    // last row should be 1.0
    float *lastRow = (float *)(cvTensorFp.getVector().data() + (cvTensorFp.size().h - 1) * cvTensorFp.rowStride());
    EXPECT_EQ((float)*lastRow, 1.0f);
    EXPECT_EQ(*(lastRow + cvTensorFp.planeStride() / sizeof(float)), 1.0f);
    EXPECT_EQ(*(lastRow + 2 * cvTensorFp.planeStride() / sizeof(float)), 1.0f);
    EXPECT_EQ(*(lastRow + 3 * cvTensorFp.planeStride() / sizeof(float)), 1.0f);

    EXPECT_EQ(*cvTensorFp.item<float>(0, 0, 0), .5f);
    EXPECT_EQ(*cvTensorFp.item<float>(0, 0, 1), 1.0f);
    EXPECT_EQ(*cvTensorFp.item<float>(0, 0, 2), .5f);
    EXPECT_EQ(*cvTensorFp.item<float>(0, 0, 3), .5f);

    EXPECT_EQ(*cvTensorFp.item<float>(1, 1, 0), .5f);
    EXPECT_EQ(*cvTensorFp.item<float>(1, 1, 1), 1.0f);
    EXPECT_EQ(*cvTensorFp.item<float>(1, 1, 2), .5f);
    EXPECT_EQ(*cvTensorFp.item<float>(1, 1, 3), .5f);

    EXPECT_EQ(*cvTensorFp.item<float>(width - 1, height - 1, 0), 1.0f);
    EXPECT_EQ(*cvTensorFp.item<float>(width - 1, height - 1, 1), 1.0f);
    EXPECT_EQ(*cvTensorFp.item<float>(width - 1, height - 1, 2), 1.0f);
    EXPECT_EQ(*cvTensorFp.item<float>(width - 1, height - 1, 3), 1.0f);
}

TEST(TensorDataUtils, SetCvImageDataPrint)
{
    int          width  = 2;
    int          height = 2;
    nvcv::Tensor tensorFp(1, {width, height}, nvcv::FMT_RGBAf32p);
    EXPECT_NO_THROW(util::SetTensorTo<float>(tensorFp.exportData(), 3.0f));
    util::TensorImageData cvTensorFp(tensorFp.exportData());
    std::cout << cvTensorFp;

    nvcv::Tensor tensor(1, {width, height}, nvcv::FMT_RGB8);
    EXPECT_NO_THROW(util::SetTensorTo<uint8_t>(tensor.exportData(), 0x55));
    util::TensorImageData cvTensor(tensor.exportData());
    std::cout << cvTensor;
}
