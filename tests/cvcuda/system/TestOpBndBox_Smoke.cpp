/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

// Smoke tests for OpBndBox - basic functionality tests without libcuosd dependency
// These tests verify the operator works correctly on GCC-10 and other compilers
// For pixel-perfect validation tests, see TestOpBndBox.cpp (requires GCC-11+)

#include "Definitions.hpp"

#include <common/TensorDataUtils.hpp>
#include <cvcuda/OpBndBox.hpp>
#include <cvcuda/priv/Types.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorDataAccess.hpp>

namespace test = nvcv::test;
using namespace cvcuda::priv;

TEST(OpBndBox_Smoke, operator_creation)
{
    // Verify operator can be created and destroyed
    EXPECT_NO_THROW(cvcuda::BndBox op);
}

TEST(OpBndBox_Smoke, basic_functionality_rgb8)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    // Create input/output tensors
    const int         N      = 2;
    const int         W      = 224;
    const int         H      = 224;
    nvcv::ImageFormat format = nvcv::FMT_RGB8;

    nvcv::Tensor imgIn  = nvcv::util::CreateTensor(N, W, H, format);
    nvcv::Tensor imgOut = nvcv::util::CreateTensor(N, W, H, format);

    auto input  = imgIn.exportData<nvcv::TensorDataStridedCuda>();
    auto output = imgOut.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_NE(input, nullptr);
    ASSERT_NE(output, nullptr);

    // Initialize input with a pattern
    auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*input);
    ASSERT_TRUE(inAccess);
    long sampleStride = inAccess->numRows() * inAccess->rowStride();
    EXPECT_EQ(cudaSuccess, cudaMemset(input->basePtr(), 128, sampleStride * N));
    EXPECT_EQ(cudaSuccess, cudaMemset(output->basePtr(), 0, sampleStride * N));

    // Create bounding boxes
    std::vector<std::vector<NVCVBndBoxI>> bndBoxVec;
    for (int n = 0; n < N; n++)
    {
        std::vector<NVCVBndBoxI> boxes;
        NVCVBndBoxI              box;
        box.box.x       = 50;
        box.box.y       = 50;
        box.box.width   = 100;
        box.box.height  = 80;
        box.thickness   = 2;
        box.fillColor   = {255, 0, 0, 255};
        box.borderColor = {0, 255, 0, 255};
        boxes.push_back(box);
        bndBoxVec.push_back(boxes);
    }

    auto bndBoxes = std::make_shared<NVCVBndBoxesImpl>(bndBoxVec);

    // Run operator
    cvcuda::BndBox op;
    EXPECT_NO_THROW(op(stream, imgIn, imgOut, (NVCVBndBoxesI)bndBoxes.get()));

    // Verify operation completed
    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    // Validation strategy: For smoke tests, we verify bounding boxes are drawn by checking that:
    // 1) Output contains the gray background color (128,128,128) from input
    // 2) Output contains red fill pixels (255,0,0) from box fillColor
    // 3) Output contains green border pixels (0,255,0) from box borderColor
    // This is a basic sanity check, not pixel-perfect validation
    std::vector<uint8_t> outData(sampleStride * N);
    EXPECT_EQ(cudaSuccess, cudaMemcpy(outData.data(), output->basePtr(), sampleStride * N, cudaMemcpyDeviceToHost));

    // Check for specific colors: gray background (128,128,128), red fill (255,0,0), green border (0,255,0)
    bool hasGray  = false;
    bool hasRed   = false;
    bool hasGreen = false;

    for (size_t i = 0; i + 2 < outData.size(); i += 3)
    {
        uint8_t r = outData[i];
        uint8_t g = outData[i + 1];
        uint8_t b = outData[i + 2];

        if (r == 128 && g == 128 && b == 128)
            hasGray = true;
        if (r == 255 && g == 0 && b == 0)
            hasRed = true;
        if (r == 0 && g == 255 && b == 0)
            hasGreen = true;

        if (hasGray && hasRed && hasGreen)
            break;
    }

    EXPECT_TRUE(hasGray) << "Output should contain gray background pixels (128,128,128)";
    EXPECT_TRUE(hasRed) << "Output should contain red fill pixels (255,0,0) from box fillColor";
    EXPECT_TRUE(hasGreen) << "Output should contain green border pixels (0,255,0) from box borderColor";

    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpBndBox_Smoke, basic_functionality_rgba8)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    const int         N      = 1;
    const int         W      = 128;
    const int         H      = 128;
    nvcv::ImageFormat format = nvcv::FMT_RGBA8;

    nvcv::Tensor imgIn  = nvcv::util::CreateTensor(N, W, H, format);
    nvcv::Tensor imgOut = nvcv::util::CreateTensor(N, W, H, format);

    auto input  = imgIn.exportData<nvcv::TensorDataStridedCuda>();
    auto output = imgOut.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_NE(input, nullptr);
    ASSERT_NE(output, nullptr);

    // Initialize input with a gray pattern (64, 64, 64, 255)
    auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*input);
    ASSERT_TRUE(inAccess);
    long                 sampleStride = inAccess->numRows() * inAccess->rowStride();
    std::vector<uint8_t> inData(sampleStride * N);
    for (size_t i = 0; i < inData.size(); i += 4)
    {
        inData[i]     = 64;  // R
        inData[i + 1] = 64;  // G
        inData[i + 2] = 64;  // B
        inData[i + 3] = 255; // A
    }
    EXPECT_EQ(cudaSuccess, cudaMemcpy(input->basePtr(), inData.data(), sampleStride * N, cudaMemcpyHostToDevice));
    EXPECT_EQ(cudaSuccess, cudaMemset(output->basePtr(), 0, sampleStride * N));

    std::vector<std::vector<NVCVBndBoxI>> bndBoxVec;
    std::vector<NVCVBndBoxI>              boxes;
    NVCVBndBoxI                           box;
    box.box.x       = 10;
    box.box.y       = 10;
    box.box.width   = 50;
    box.box.height  = 50;
    box.thickness   = 3;
    box.fillColor   = {0, 0, 255, 255}; // Blue with alpha=255 (fully opaque)
    box.borderColor = {255, 255, 0, 0}; // Yellow with alpha=0 (fully transparent)
    boxes.push_back(box);
    bndBoxVec.push_back(boxes);

    auto bndBoxes = std::make_shared<NVCVBndBoxesImpl>(bndBoxVec);

    cvcuda::BndBox op;
    EXPECT_NO_THROW(op(stream, imgIn, imgOut, (NVCVBndBoxesI)bndBoxes.get()));

    // Verify operation completed successfully
    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    // Validation strategy: Verify RGBA format with proper alpha handling
    // - Gray background (64,64,64,255) from input should be visible in areas without drawing
    // - Blue fill (with alpha=255, fully opaque) should completely replace background in fill area
    // - Yellow border has alpha=0 (fully transparent), so should NOT appear anywhere
    std::vector<uint8_t> outData(sampleStride * N);
    EXPECT_EQ(cudaSuccess, cudaMemcpy(outData.data(), output->basePtr(), sampleStride * N, cudaMemcpyDeviceToHost));

    bool hasGray   = false;
    bool hasBlue   = false;
    bool hasYellow = false;

    for (size_t i = 0; i + 3 < outData.size(); i += 4)
    {
        uint8_t r = outData[i];
        uint8_t g = outData[i + 1];
        uint8_t b = outData[i + 2];

        // Gray background with full alpha (should be present)
        if (uint8_t a = outData[i + 3]; r == 64 && g == 64 && b == 64 && a == 255)
            hasGray = true;

        // Blue component should be present from fill (pure blue or blended)
        if (b > 200) // Strong blue presence
            hasBlue = true;

        // Yellow should NOT be present (borderColor has alpha=0, so transparent)
        if (r > 200 && g > 200 && b < 50) // Yellow is high R, high G, low B
            hasYellow = true;
    }

    EXPECT_TRUE(hasGray) << "Output should contain gray background pixels (64,64,64,255)";
    EXPECT_TRUE(hasBlue) << "Output should contain blue pixels from opaque fill (alpha=255)";
    EXPECT_FALSE(hasYellow) << "Output should NOT contain yellow - borderColor has alpha=0 (fully transparent)";

    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpBndBox_Smoke, multiple_boxes)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    nvcv::Tensor imgIn  = nvcv::util::CreateTensor(1, 640, 480, nvcv::FMT_RGB8);
    nvcv::Tensor imgOut = nvcv::util::CreateTensor(1, 640, 480, nvcv::FMT_RGB8);

    auto input  = imgIn.exportData<nvcv::TensorDataStridedCuda>();
    auto output = imgOut.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_NE(input, nullptr);
    ASSERT_NE(output, nullptr);

    // Initialize input with black background
    auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*input);
    ASSERT_TRUE(inAccess);
    long sampleStride = inAccess->numRows() * inAccess->rowStride();
    EXPECT_EQ(cudaSuccess, cudaMemset(input->basePtr(), 0, sampleStride));
    EXPECT_EQ(cudaSuccess, cudaMemset(output->basePtr(), 0, sampleStride));

    // Create multiple bounding boxes
    std::vector<std::vector<NVCVBndBoxI>> bndBoxVec;
    std::vector<NVCVBndBoxI>              boxes;

    for (int i = 0; i < 10; i++)
    {
        NVCVBndBoxI box;
        box.box.x       = i * 50;
        box.box.y       = i * 40;
        box.box.width   = 60;
        box.box.height  = 50;
        box.thickness   = 2;
        box.fillColor   = {static_cast<unsigned char>(i * 25), 100, 200, 100};
        box.borderColor = {255, static_cast<unsigned char>(255 - i * 25), 0, 255};
        boxes.push_back(box);
    }
    bndBoxVec.push_back(boxes);

    auto bndBoxes = std::make_shared<NVCVBndBoxesImpl>(bndBoxVec);

    cvcuda::BndBox op;
    EXPECT_NO_THROW(op(stream, imgIn, imgOut, (NVCVBndBoxesI)bndBoxes.get()));
    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpBndBox_Smoke, memory_management)
{
    // Run operator multiple times to verify no memory leaks
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    cvcuda::BndBox op;

    for (int iter = 0; iter < 5; iter++) // NOSONAR
    {
        nvcv::Tensor imgIn  = nvcv::util::CreateTensor(1, 224, 224, nvcv::FMT_RGB8);
        nvcv::Tensor imgOut = nvcv::util::CreateTensor(1, 224, 224, nvcv::FMT_RGB8);

        auto input  = imgIn.exportData<nvcv::TensorDataStridedCuda>();
        auto output = imgOut.exportData<nvcv::TensorDataStridedCuda>();
        ASSERT_NE(input, nullptr);
        ASSERT_NE(output, nullptr);

        // Initialize input with known background
        auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*input);
        ASSERT_TRUE(inAccess);
        long sampleStride = inAccess->numRows() * inAccess->rowStride();
        EXPECT_EQ(cudaSuccess, cudaMemset(input->basePtr(), 100, sampleStride));
        EXPECT_EQ(cudaSuccess, cudaMemset(output->basePtr(), 0, sampleStride));

        std::vector<std::vector<NVCVBndBoxI>> bndBoxVec;
        std::vector<NVCVBndBoxI>              boxes;
        NVCVBndBoxI                           box;
        box.box.x       = 20 + iter * 10;
        box.box.y       = 20 + iter * 10;
        box.box.width   = 80;
        box.box.height  = 60;
        box.thickness   = 2;
        box.fillColor   = {255, 0, 0, 128};
        box.borderColor = {0, 255, 0, 255};
        boxes.push_back(box);
        bndBoxVec.push_back(boxes);

        auto bndBoxes = std::make_shared<NVCVBndBoxesImpl>(bndBoxVec);

        EXPECT_NO_THROW(op(stream, imgIn, imgOut, (NVCVBndBoxesI)bndBoxes.get()));
        EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    }

    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpBndBox_Smoke, edge_cases)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    nvcv::Tensor imgIn  = nvcv::util::CreateTensor(1, 100, 100, nvcv::FMT_RGB8);
    nvcv::Tensor imgOut = nvcv::util::CreateTensor(1, 100, 100, nvcv::FMT_RGB8);

    auto input  = imgIn.exportData<nvcv::TensorDataStridedCuda>();
    auto output = imgOut.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_NE(input, nullptr);
    ASSERT_NE(output, nullptr);

    // Initialize input with known background
    auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*input);
    ASSERT_TRUE(inAccess);
    long sampleStride = inAccess->numRows() * inAccess->rowStride();
    EXPECT_EQ(cudaSuccess, cudaMemset(input->basePtr(), 50, sampleStride));

    cvcuda::BndBox op;

    // Empty bounding box list - output should match input
    {
        EXPECT_EQ(cudaSuccess, cudaMemset(output->basePtr(), 0, sampleStride));
        std::vector<std::vector<NVCVBndBoxI>> bndBoxVec;
        std::vector<NVCVBndBoxI>              boxes; // Empty
        bndBoxVec.push_back(boxes);
        auto bndBoxes = std::make_shared<NVCVBndBoxesImpl>(bndBoxVec);
        EXPECT_NO_THROW(op(stream, imgIn, imgOut, (NVCVBndBoxesI)bndBoxes.get()));
        EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

        // Empty box list: with the operator bug fixed, output should be uninitialized
        // For a smoke test, we just verify the operation doesn't crash
        // A more comprehensive test would verify output remains unchanged or matches input
    }

    // Box at image boundary - should draw border without crashing
    {
        EXPECT_EQ(cudaSuccess, cudaMemset(output->basePtr(), 0, sampleStride));
        std::vector<std::vector<NVCVBndBoxI>> bndBoxVec;
        std::vector<NVCVBndBoxI>              boxes;
        NVCVBndBoxI                           box;
        box.box.x       = 0;
        box.box.y       = 0;
        box.box.width   = 100;
        box.box.height  = 100;
        box.thickness   = 1;
        box.fillColor   = {0, 0, 0, 0};
        box.borderColor = {255, 255, 255, 255};
        boxes.push_back(box);
        bndBoxVec.push_back(boxes);
        auto bndBoxes = std::make_shared<NVCVBndBoxesImpl>(bndBoxVec);
        EXPECT_NO_THROW(op(stream, imgIn, imgOut, (NVCVBndBoxesI)bndBoxes.get()));
        EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

        // Verify white border was drawn at boundary
        std::vector<uint8_t> outData(sampleStride);
        EXPECT_EQ(cudaSuccess, cudaMemcpy(outData.data(), output->basePtr(), sampleStride, cudaMemcpyDeviceToHost));
        bool hasWhite = false;
        for (size_t i = 0; i + 2 < outData.size(); i += 3)
        {
            if (outData[i] == 255 && outData[i + 1] == 255 && outData[i + 2] == 255)
            {
                hasWhite = true;
                break;
            }
        }
        EXPECT_TRUE(hasWhite) << "Output should contain white border pixels at image boundary";
    }

    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}
