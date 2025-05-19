/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <common/TensorDataUtils.hpp>
#include <common/ValueTests.hpp>
#include <cvcuda/OpErase.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorDataAccess.hpp>

#include <deque>
#include <iostream>

NVCV_TEST_SUITE_P(OpErase, nvcv::test::ValueList<int, bool, bool>{
  // N, random, isInplace
                               {1, false, false},
                               {2, false, false},
                               {1,  true, false},
                               {2,  true, false},
                               {1, false,  true},
                               {2, false,  true},
                               {1,  true,  true},
                               {2,  true,  true},
});

TEST_P(OpErase, correct_output)
{
    int          N                    = GetParamValue<0>();
    bool         random               = GetParamValue<1>();
    bool         isInplace            = GetParamValue<2>();
    int          max_num_erasing_area = 2;
    unsigned int seed                 = 0;

    cudaStream_t stream;
    EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    nvcv::Tensor  imgIn   = nvcv::util::CreateTensor(N, 640, 480, nvcv::FMT_U8);
    nvcv::Tensor  _imgOut = nvcv::util::CreateTensor(N, 640, 480, nvcv::FMT_U8);
    nvcv::Tensor &imgOut  = isInplace ? imgIn : _imgOut;

    auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(imgIn.exportData());
    ASSERT_TRUE(inAccess);

    ASSERT_EQ(N, inAccess->numSamples());

    // setup the buffer
    EXPECT_EQ(cudaSuccess, cudaMemset2D(inAccess->planeData(0), inAccess->rowStride(), 0,
                                        inAccess->numCols() * inAccess->colStride(), inAccess->numRows()));

    auto outData = imgOut.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_NE(nullptr, outData);

    auto outAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*outData);
    ASSERT_TRUE(outAccess);

    int64_t outSampleStride = outAccess->sampleStride();

    if (outData->rank() == 3)
    {
        outSampleStride = outAccess->numRows() * outAccess->rowStride();
    }

    int64_t outBufferSize = outSampleStride * outAccess->numSamples();

    // Set output buffer to dummy value
    if (!isInplace)
    {
        EXPECT_EQ(cudaSuccess, cudaMemset(outAccess->sampleData(0), 0xFA, outBufferSize));
    }

    //parameters
    int          num_erasing_area = 2;
    nvcv::Tensor anchor({{num_erasing_area}, "N"}, nvcv::TYPE_2S32);
    nvcv::Tensor erasing({{num_erasing_area}, "N"}, nvcv::TYPE_3S32);
    nvcv::Tensor values({{num_erasing_area}, "N"}, nvcv::TYPE_F32);
    nvcv::Tensor imgIdx({{num_erasing_area}, "N"}, nvcv::TYPE_S32);

    auto anchorData  = anchor.exportData<nvcv::TensorDataStridedCuda>();
    auto erasingData = erasing.exportData<nvcv::TensorDataStridedCuda>();
    auto valuesData  = values.exportData<nvcv::TensorDataStridedCuda>();
    auto imgIdxData  = imgIdx.exportData<nvcv::TensorDataStridedCuda>();

    ASSERT_NE(nullptr, anchorData);
    ASSERT_NE(nullptr, erasingData);
    ASSERT_NE(nullptr, valuesData);
    ASSERT_NE(nullptr, imgIdxData);

    std::vector<int2>  anchorVec(num_erasing_area);
    std::vector<int3>  erasingVec(num_erasing_area);
    std::vector<int>   imgIdxVec(num_erasing_area);
    std::vector<float> valuesVec(num_erasing_area);

    anchorVec[0].x  = 0;
    anchorVec[0].y  = 0;
    erasingVec[0].x = 10;
    erasingVec[0].y = 10;
    erasingVec[0].z = 0x1;
    imgIdxVec[0]    = 0;
    valuesVec[0]    = 1.f;

    anchorVec[1].x  = 10;
    anchorVec[1].y  = 10;
    erasingVec[1].x = 20;
    erasingVec[1].y = 20;
    erasingVec[1].z = 0x1;
    imgIdxVec[1]    = 0;
    valuesVec[1]    = 1.f;

    // Copy vectors to the GPU
    ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(anchorData->basePtr(), anchorVec.data(), anchorVec.size() * sizeof(int2),
                                           cudaMemcpyHostToDevice, stream));
    ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(erasingData->basePtr(), erasingVec.data(), erasingVec.size() * sizeof(int3),
                                           cudaMemcpyHostToDevice, stream));
    ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(imgIdxData->basePtr(), imgIdxVec.data(), imgIdxVec.size() * sizeof(int),
                                           cudaMemcpyHostToDevice, stream));
    ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(valuesData->basePtr(), valuesVec.data(), valuesVec.size() * sizeof(float),
                                           cudaMemcpyHostToDevice, stream));

    // Call operator
    cvcuda::Erase eraseOp(max_num_erasing_area);
    EXPECT_NO_THROW(eraseOp(stream, imgIn, imgOut, anchor, erasing, values, imgIdx, random, seed));

    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    std::vector<uint8_t> test(outBufferSize, 0xA);

    //Check data
    if (!random)
    {
        EXPECT_EQ(cudaSuccess, cudaMemcpy(test.data(), outData->basePtr(), outBufferSize, cudaMemcpyDeviceToHost));

        EXPECT_EQ(test[0], 1);
        EXPECT_EQ(test[9], 1);
        EXPECT_EQ(test[10], 0);
        EXPECT_EQ(test[9 * 640], 1);
        EXPECT_EQ(test[9 * 640 + 9], 1);
        EXPECT_EQ(test[9 * 640 + 10], 0);
        EXPECT_EQ(test[10 * 640], 0);
        EXPECT_EQ(test[10 * 640 + 10], 1);
    }
    EXPECT_EQ(cudaSuccess, cudaMemcpy(test.data(), outData->basePtr(), outBufferSize, cudaMemcpyDeviceToHost));

    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpErase, OpErase_Varshape)
{
    cudaStream_t stream;
    EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream));
    std::deque<bool> isInplaces{true, false};
    std::deque<bool> isRandoms{true, false};
    for (bool isInplace : isInplaces)
    {
        for (bool random : isRandoms)
        {
            std::vector<nvcv::Image> imgSrc, imgDst;
            imgSrc.emplace_back(nvcv::Size2D{640, 480}, nvcv::FMT_U8);
            imgDst.emplace_back(nvcv::Size2D{640, 480}, nvcv::FMT_U8);

            nvcv::ImageBatchVarShape  batchSrc(1);
            nvcv::ImageBatchVarShape  _batchDst(1);
            nvcv::ImageBatchVarShape &batchDst = isInplace ? batchSrc : _batchDst;
            batchSrc.pushBack(imgSrc.begin(), imgSrc.end());
            _batchDst.pushBack(imgDst.begin(), imgDst.end());

            for (int i = 0; i < 1; ++i)
            {
                const auto srcData = imgSrc[i].exportData<nvcv::ImageDataStridedCuda>();
                assert(srcData->numPlanes() == 1);

                int srcWidth  = srcData->plane(0).width;
                int srcHeight = srcData->plane(0).height;

                int srcRowStride = srcWidth * nvcv::FMT_U8.planePixelStrideBytes(0);

                EXPECT_EQ(cudaSuccess,
                          cudaMemset2D(srcData->plane(0).basePtr, srcRowStride, 0, srcRowStride, srcHeight));
            }

            if (!isInplace)
            {
                for (int i = 0; i < 1; ++i)
                {
                    const auto dstData      = imgSrc[i].exportData<nvcv::ImageDataStridedCuda>();
                    int        dstWidth     = dstData->plane(0).width;
                    int        dstHeight    = dstData->plane(0).height;
                    int        dstRowStride = dstWidth * nvcv::FMT_U8.planePixelStrideBytes(0);
                    EXPECT_EQ(cudaSuccess,
                              cudaMemset2D(dstData->plane(0).basePtr, dstRowStride, 0, dstRowStride, dstHeight));
                }
            }

            //parameters
            int          num_erasing_area = 2;
            nvcv::Tensor anchor({{num_erasing_area}, "N"}, nvcv::TYPE_2S32);
            nvcv::Tensor erasing({{num_erasing_area}, "N"}, nvcv::TYPE_3S32);
            nvcv::Tensor values({{num_erasing_area}, "N"}, nvcv::TYPE_F32);
            nvcv::Tensor imgIdx({{num_erasing_area}, "N"}, nvcv::TYPE_S32);

            auto anchorData  = anchor.exportData<nvcv::TensorDataStridedCuda>();
            auto erasingData = erasing.exportData<nvcv::TensorDataStridedCuda>();
            auto valuesData  = values.exportData<nvcv::TensorDataStridedCuda>();
            auto imgIdxData  = imgIdx.exportData<nvcv::TensorDataStridedCuda>();

            ASSERT_NE(nullptr, anchorData);
            ASSERT_NE(nullptr, erasingData);
            ASSERT_NE(nullptr, valuesData);
            ASSERT_NE(nullptr, imgIdxData);

            std::vector<int2>  anchorVec(num_erasing_area);
            std::vector<int3>  erasingVec(num_erasing_area);
            std::vector<int>   imgIdxVec(num_erasing_area);
            std::vector<float> valuesVec(num_erasing_area);

            anchorVec[0].x  = 0;
            anchorVec[0].y  = 0;
            erasingVec[0].x = 10;
            erasingVec[0].y = 10;
            erasingVec[0].z = 0x1;
            imgIdxVec[0]    = 0;
            valuesVec[0]    = 1.f;

            anchorVec[1].x  = 10;
            anchorVec[1].y  = 10;
            erasingVec[1].x = 20;
            erasingVec[1].y = 20;
            erasingVec[1].z = 0x1;
            imgIdxVec[1]    = 0;
            valuesVec[1]    = 1.f;

            // Copy vectors to the GPU
            ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(anchorData->basePtr(), anchorVec.data(),
                                                   anchorVec.size() * sizeof(int2), cudaMemcpyHostToDevice, stream));
            ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(erasingData->basePtr(), erasingVec.data(),
                                                   erasingVec.size() * sizeof(int3), cudaMemcpyHostToDevice, stream));
            ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(imgIdxData->basePtr(), imgIdxVec.data(),
                                                   imgIdxVec.size() * sizeof(int), cudaMemcpyHostToDevice, stream));
            ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(valuesData->basePtr(), valuesVec.data(),
                                                   valuesVec.size() * sizeof(float), cudaMemcpyHostToDevice, stream));

            // Call operator
            unsigned int  seed                 = 0;
            int           max_num_erasing_area = 2;
            cvcuda::Erase eraseOp(max_num_erasing_area);
            EXPECT_NO_THROW(eraseOp(stream, batchSrc, batchDst, anchor, erasing, values, imgIdx, random, seed));

            EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

            const auto dstData = isInplace ? imgSrc[0].exportData<nvcv::ImageDataStridedCuda>()
                                           : imgDst[0].exportData<nvcv::ImageDataStridedCuda>();
            assert(dstData->numPlanes() == 1);

            int dstWidth  = dstData->plane(0).width;
            int dstHeight = dstData->plane(0).height;

            int dstRowStride = dstWidth * nvcv::FMT_U8.planePixelStrideBytes(0);

            std::vector<uint8_t> test(dstHeight * dstRowStride, 0xFF);

            // Copy output data to Host
            if (!random)
            {
                ASSERT_EQ(cudaSuccess,
                          cudaMemcpy2D(test.data(), dstRowStride, dstData->plane(0).basePtr,
                                       dstData->plane(0).rowStride, dstRowStride, dstHeight, cudaMemcpyDeviceToHost));

                EXPECT_EQ(test[0], 1);
                EXPECT_EQ(test[9], 1);
                EXPECT_EQ(test[10], 0);
                EXPECT_EQ(test[9 * 640], 1);
                EXPECT_EQ(test[9 * 640 + 9], 1);
                EXPECT_EQ(test[9 * 640 + 10], 0);
                EXPECT_EQ(test[10 * 640], 0);
                EXPECT_EQ(test[10 * 640 + 10], 1);
            }
        }
    }

    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

// clang-format off
NVCV_TEST_SUITE_P(OpErase_Negative, nvcv::test::ValueList<nvcv::ImageFormat, nvcv::ImageFormat, std::string, nvcv::DataType, std::string, nvcv::DataType, std::string, nvcv::DataType, std::string, nvcv::DataType, int>
{
    //   in_format, out_format, anchor_layout, anchor_datatype, erasingData_layout, erasingData_datatype, imgIdxData_layout, imgIdxData_datatype, valuesData_layout, valuesData_type, num_erasing_area
    { nvcv::FMT_RGB8p, nvcv::FMT_RGB8, "N", nvcv::TYPE_2S32, "N", nvcv::TYPE_3S32, "N", nvcv::TYPE_S32, "N", nvcv::TYPE_F32, 2}, // invalid in layout
    { nvcv::FMT_RGBf16, nvcv::FMT_RGBf16, "N", nvcv::TYPE_2S32, "N", nvcv::TYPE_3S32, "N", nvcv::TYPE_S32, "N", nvcv::TYPE_F32, 2}, // invalid in layout
    { nvcv::FMT_RGB8, nvcv::FMT_RGB8p, "N", nvcv::TYPE_2S32, "N", nvcv::TYPE_3S32, "N", nvcv::TYPE_S32, "N", nvcv::TYPE_F32, 2}, // invalid out layout
    { nvcv::FMT_RGB8, nvcv::FMT_RGBf32, "N", nvcv::TYPE_2S32, "N", nvcv::TYPE_3S32, "N", nvcv::TYPE_S32, "N", nvcv::TYPE_F32, 2}, // different datatype
    { nvcv::FMT_RGB8, nvcv::FMT_RGB8, "N", nvcv::TYPE_2F32, "N", nvcv::TYPE_3S32, "N", nvcv::TYPE_S32, "N", nvcv::TYPE_F32, 2}, // invalid anchor datatype
    { nvcv::FMT_RGB8, nvcv::FMT_RGB8, "NHW", nvcv::TYPE_2S32, "N", nvcv::TYPE_3S32, "N", nvcv::TYPE_S32, "N", nvcv::TYPE_F32, 2}, // invalid anchor dim
    { nvcv::FMT_RGB8, nvcv::FMT_RGB8, "N", nvcv::TYPE_2S32, "N", nvcv::TYPE_3S32, "N", nvcv::TYPE_S32, "N", nvcv::TYPE_F32, 3}, // Invalid num of erasing area 3 (> max)
    { nvcv::FMT_RGB8, nvcv::FMT_RGB8, "N", nvcv::TYPE_2S32, "N", nvcv::TYPE_3F32, "N", nvcv::TYPE_S32, "N", nvcv::TYPE_F32, 2}, // invalid erasing datatype
    { nvcv::FMT_RGB8, nvcv::FMT_RGB8, "N", nvcv::TYPE_2S32, "NHW", nvcv::TYPE_3S32, "N", nvcv::TYPE_S32, "N", nvcv::TYPE_F32, 2}, // invalid erasing dim
    { nvcv::FMT_RGB8, nvcv::FMT_RGB8, "N", nvcv::TYPE_2S32, "N", nvcv::TYPE_3S32, "N", nvcv::TYPE_F32, "N", nvcv::TYPE_F32, 2}, // invalid imgIdx datatype
    { nvcv::FMT_RGB8, nvcv::FMT_RGB8, "N", nvcv::TYPE_2S32, "N", nvcv::TYPE_3S32, "NHW", nvcv::TYPE_S32, "N", nvcv::TYPE_F32, 2}, // invalid imgIdx datatype
    { nvcv::FMT_RGB8, nvcv::FMT_RGB8, "N", nvcv::TYPE_2S32, "N", nvcv::TYPE_3S32, "N", nvcv::TYPE_S32, "N", nvcv::TYPE_S32, 2}, // invalid values datatype
    { nvcv::FMT_RGB8, nvcv::FMT_RGB8, "N", nvcv::TYPE_2S32, "N", nvcv::TYPE_3S32, "N", nvcv::TYPE_S32, "NHW", nvcv::TYPE_F32, 2}, // invalid values datatype
});

// clang-format on

TEST(OpErase_Negative, create_null_handle)
{
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, cvcudaEraseCreate(nullptr, 1));
}

TEST(OpErase_Negative, create_negative_area)
{
    NVCVOperatorHandle handle;
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, cvcudaEraseCreate(&handle, -1));
}

TEST_P(OpErase_Negative, infer_negative_parameter)
{
    nvcv::ImageFormat inputFmt             = GetParamValue<0>();
    nvcv::ImageFormat outputFmt            = GetParamValue<1>();
    std::string       anchor_layout        = GetParamValue<2>();
    nvcv::DataType    anchor_datatype      = GetParamValue<3>();
    std::string       erasingData_layout   = GetParamValue<4>();
    nvcv::DataType    erasingData_datatype = GetParamValue<5>();
    std::string       imgIdxData_layout    = GetParamValue<6>();
    nvcv::DataType    imgIdxData_datatype  = GetParamValue<7>();
    std::string       valuesData_layout    = GetParamValue<8>();
    nvcv::DataType    valuesData_datatype  = GetParamValue<9>();
    int               num_erasing_area     = GetParamValue<10>();

    int          max_num_erasing_area = 2;
    unsigned int seed                 = 0;

    nvcv::Tensor imgIn  = nvcv::util::CreateTensor(1, 24, 24, inputFmt);
    nvcv::Tensor imgOut = nvcv::util::CreateTensor(1, 24, 24, outputFmt);

    //parameters
    nvcv::TensorShape anchorShape = anchor_layout.size() == 3 ? nvcv::TensorShape{{num_erasing_area, num_erasing_area, num_erasing_area}, anchor_layout.c_str()} : nvcv::TensorShape{{num_erasing_area}, anchor_layout.c_str()};
    nvcv::TensorShape erasingShape = erasingData_layout.size() == 3 ? nvcv::TensorShape{{num_erasing_area, num_erasing_area, num_erasing_area}, erasingData_layout.c_str()} : nvcv::TensorShape{{num_erasing_area}, erasingData_layout.c_str()};
    nvcv::TensorShape imgIdxShape = imgIdxData_layout.size() == 3 ? nvcv::TensorShape{{num_erasing_area, num_erasing_area, num_erasing_area}, imgIdxData_layout.c_str()} : nvcv::TensorShape{{num_erasing_area}, imgIdxData_layout.c_str()};
    nvcv::TensorShape valuesShape = valuesData_layout.size() == 3 ? nvcv::TensorShape{{num_erasing_area, num_erasing_area, num_erasing_area}, valuesData_layout.c_str()} : nvcv::TensorShape{{num_erasing_area}, valuesData_layout.c_str()};
    nvcv::Tensor anchor(anchorShape, anchor_datatype);
    nvcv::Tensor erasing(erasingShape, erasingData_datatype);
    nvcv::Tensor values(valuesShape, valuesData_datatype);
    nvcv::Tensor imgIdx(imgIdxShape, imgIdxData_datatype);

    // Call operator
    cvcuda::Erase eraseOp(max_num_erasing_area);
    EXPECT_EQ(
        NVCV_ERROR_INVALID_ARGUMENT,
        nvcv::ProtectCall([&] { eraseOp(nullptr, imgIn, imgOut, anchor, erasing, values, imgIdx, false, seed); }));
}

TEST_P(OpErase_Negative, varshape_infer_negative_parameter)
{
    nvcv::ImageFormat inputFmt             = GetParamValue<0>();
    nvcv::ImageFormat outputFmt            = GetParamValue<1>();
    std::string       anchor_layout        = GetParamValue<2>();
    nvcv::DataType    anchor_datatype      = GetParamValue<3>();
    std::string       erasingData_layout   = GetParamValue<4>();
    nvcv::DataType    erasingData_datatype = GetParamValue<5>();
    std::string       imgIdxData_layout    = GetParamValue<6>();
    nvcv::DataType    imgIdxData_datatype  = GetParamValue<7>();
    std::string       valuesData_layout    = GetParamValue<8>();
    nvcv::DataType    valuesData_datatype  = GetParamValue<9>();
    int               num_erasing_area     = GetParamValue<10>();

    int          max_num_erasing_area = 2;
    unsigned int seed                 = 0;

    nvcv::ImageBatchVarShape batchSrc(1);
    nvcv::ImageBatchVarShape batchDst(1);
    batchSrc.pushBack(nvcv::Image{
        nvcv::Size2D{32, 32},
        inputFmt
    });
    batchDst.pushBack(nvcv::Image{
        nvcv::Size2D{32, 32},
        outputFmt
    });

    //parameters
    nvcv::TensorShape anchorShape = anchor_layout.size() == 3 ? nvcv::TensorShape{{num_erasing_area, num_erasing_area, num_erasing_area}, anchor_layout.c_str()} : nvcv::TensorShape{{num_erasing_area}, anchor_layout.c_str()};
    nvcv::TensorShape erasingShape = erasingData_layout.size() == 3 ? nvcv::TensorShape{{num_erasing_area, num_erasing_area, num_erasing_area}, erasingData_layout.c_str()} : nvcv::TensorShape{{num_erasing_area}, erasingData_layout.c_str()};
    nvcv::TensorShape imgIdxShape = imgIdxData_layout.size() == 3 ? nvcv::TensorShape{{num_erasing_area, num_erasing_area, num_erasing_area}, imgIdxData_layout.c_str()} : nvcv::TensorShape{{num_erasing_area}, imgIdxData_layout.c_str()};
    nvcv::TensorShape valuesShape = valuesData_layout.size() == 3 ? nvcv::TensorShape{{num_erasing_area, num_erasing_area, num_erasing_area}, valuesData_layout.c_str()} : nvcv::TensorShape{{num_erasing_area}, valuesData_layout.c_str()};
    nvcv::Tensor anchor(anchorShape, anchor_datatype);
    nvcv::Tensor erasing(erasingShape, erasingData_datatype);
    nvcv::Tensor values(valuesShape, valuesData_datatype);
    nvcv::Tensor imgIdx(imgIdxShape, imgIdxData_datatype);

    // Call operator
    cvcuda::Erase eraseOp(max_num_erasing_area);
    EXPECT_EQ(
        NVCV_ERROR_INVALID_ARGUMENT,
        nvcv::ProtectCall([&] { eraseOp(nullptr, batchSrc, batchDst, anchor, erasing, values, imgIdx, false, seed); }));
}

TEST(OpErase_Negative, varshape_hasDifferentFormat)
{
    cudaStream_t stream;
    EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    nvcv::ImageFormat fmt            = nvcv::FMT_RGB8;
    const int         numberOfImages = 5;
    unsigned int      seed           = 0;

    int srcWidthBase  = 4;
    int srcHeightBase = 4;

    std::vector<std::tuple<nvcv::ImageFormat, nvcv::ImageFormat>> testSet{
        {nvcv::FMT_RGBA8,             fmt},
        {            fmt, nvcv::FMT_RGBA8}
    };
    for (auto testCase : testSet)
    {
        nvcv::ImageFormat inputFmtExtra  = std::get<0>(testCase);
        nvcv::ImageFormat outputFmtExtra = std::get<1>(testCase);

        // Create input and output
        std::default_random_engine         randEng;
        std::uniform_int_distribution<int> rndSrcWidth(srcWidthBase * 0.8, srcWidthBase * 1.1);
        std::uniform_int_distribution<int> rndSrcHeight(srcHeightBase * 0.8, srcHeightBase * 1.1);

        int          num_erasing_area = 2;
        nvcv::Tensor anchor({{num_erasing_area}, "N"}, nvcv::TYPE_2S32);
        nvcv::Tensor erasing({{num_erasing_area}, "N"}, nvcv::TYPE_3S32);
        nvcv::Tensor values({{num_erasing_area}, "N"}, nvcv::TYPE_F32);
        nvcv::Tensor imgIdx({{num_erasing_area}, "N"}, nvcv::TYPE_S32);

        std::vector<nvcv::Image> imgSrc, imgDst;

        for (int i = 0; i < numberOfImages - 1; ++i)
        {
            int tmpWidth  = i == 0 ? srcWidthBase : rndSrcWidth(randEng);
            int tmpHeight = i == 0 ? srcHeightBase : rndSrcHeight(randEng);

            imgSrc.emplace_back(nvcv::Size2D{tmpWidth, tmpHeight}, fmt);
            imgDst.emplace_back(nvcv::Size2D{tmpWidth, tmpHeight}, fmt);
        }
        imgSrc.emplace_back(imgSrc[0].size(), inputFmtExtra);
        imgDst.emplace_back(imgSrc.back().size(), outputFmtExtra);

        nvcv::ImageBatchVarShape batchSrc(numberOfImages);
        batchSrc.pushBack(imgSrc.begin(), imgSrc.end());

        nvcv::ImageBatchVarShape batchDst(numberOfImages);
        batchDst.pushBack(imgDst.begin(), imgDst.end());

        // Generate test result
        cvcuda::Erase eraseOp(num_erasing_area);
        EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
                  nvcv::ProtectCall(
                      [&] { eraseOp(nullptr, batchSrc, batchDst, anchor, erasing, values, imgIdx, false, seed); }));
    }

    // Get test data back
    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}
