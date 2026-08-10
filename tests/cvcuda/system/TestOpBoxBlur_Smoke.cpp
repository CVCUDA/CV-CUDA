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

// Smoke tests for OpBoxBlur - basic functionality tests without libcuosd dependency
// These tests verify the operator works correctly on GCC-10 and other compilers
// For pixel-perfect validation tests, see TestOpBoxBlur.cpp (requires GCC-11+)

#include "Definitions.hpp"

#include <common/TensorDataUtils.hpp>
#include <cvcuda/OpBoxBlur.hpp>
#include <cvcuda/priv/Types.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorDataAccess.hpp>

namespace test = nvcv::test;
using namespace cvcuda::priv;

TEST(OpBoxBlur_Smoke, operator_creation)
{
    // Verify operator can be created and destroyed
    EXPECT_NO_THROW(cvcuda::BoxBlur op);
}

TEST(OpBoxBlur_Smoke, basic_functionality_rgb8)
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

    // Initialize input with a pattern: white square (255) on black background (0)
    // This creates sharp edges that will be smoothed by blur
    auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*input);
    ASSERT_TRUE(inAccess);
    const auto sampleStride = static_cast<size_t>(inAccess->numRows()) * static_cast<size_t>(inAccess->rowStride());
    std::vector<uint8_t> inData(sampleStride * N, 0);

    // Create white squares in the center for each batch
    auto rowStride = static_cast<int>(inAccess->rowStride());
    for (int n = 0; n < N; n++)
    {
        for (int y = 80; y < 150; y++)
        {
            for (int x = 80; x < 150; x++)
            {
                auto offset        = static_cast<size_t>(n) * sampleStride + y * rowStride + x * 3;
                inData[offset]     = 255; // R
                inData[offset + 1] = 255; // G
                inData[offset + 2] = 255; // B
            }
        }
    }
    EXPECT_EQ(cudaSuccess, cudaMemcpy(input->basePtr(), inData.data(), sampleStride * N, cudaMemcpyHostToDevice));
    EXPECT_EQ(cudaSuccess, cudaMemset(output->basePtr(), 0, sampleStride * N));

    // Create blur boxes that cover the edge of the white square
    std::vector<std::vector<NVCVBlurBoxI>> blurBoxVec;
    for (int n = 0; n < N; n++)
    {
        std::vector<NVCVBlurBoxI> boxes;
        NVCVBlurBoxI              box;
        box.box.x      = 50;
        box.box.y      = 50;
        box.box.width  = 100;
        box.box.height = 80;
        box.kernelSize = 7;
        boxes.push_back(box);
        blurBoxVec.push_back(boxes);
    }

    auto blurBoxes = std::make_shared<NVCVBlurBoxesImpl>(blurBoxVec);

    // Run operator
    cvcuda::BoxBlur op;
    EXPECT_NO_THROW(op(stream, imgIn, imgOut, (NVCVBlurBoxesI)blurBoxes.get()));

    // Verify operation completed
    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    // Validation strategy: For smoke tests, we verify blur occurred by checking that:
    // 1) Pixels in the blur region changed from the input pattern
    // 2) Edge pixels show averaging (neither all black nor all white)
    // This is a basic sanity check, not pixel-perfect validation
    std::vector<uint8_t> outData(sampleStride * N);
    EXPECT_EQ(cudaSuccess, cudaMemcpy(outData.data(), output->basePtr(), sampleStride * N, cudaMemcpyDeviceToHost));

    // Check a pixel on the edge of the white square inside the blur box (should be smoothed)
    // At position (85, 85) we expect blurred values (not pure black=0 or pure white=255)
    int     edgeOffset = 85 * rowStride + 85 * 3;
    uint8_t r          = outData[edgeOffset];
    uint8_t g          = outData[edgeOffset + 1];
    uint8_t b          = outData[edgeOffset + 2];

    EXPECT_GT(r, 0) << "Blur should smooth edge pixels (not pure black)";
    EXPECT_LT(r, 255) << "Blur should smooth edge pixels (not pure white)";
    EXPECT_GT(g, 0);
    EXPECT_LT(g, 255);
    EXPECT_GT(b, 0);
    EXPECT_LT(b, 255);

    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpBoxBlur_Smoke, basic_functionality_rgba8)
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

    // Initialize input with a pattern: red square on blue background
    // This creates sharp color transitions that will be smoothed by blur
    auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*input);
    ASSERT_TRUE(inAccess);
    const auto sampleStride = static_cast<size_t>(inAccess->numRows()) * static_cast<size_t>(inAccess->rowStride());
    std::vector<uint8_t> inData(sampleStride * N);
    auto                 rowStride = static_cast<int>(inAccess->rowStride());

    // Fill with blue background (0, 0, 255, 255)
    for (size_t i = 0; i < inData.size(); i += 4)
    {
        inData[i]     = 0;   // R
        inData[i + 1] = 0;   // G
        inData[i + 2] = 255; // B
        inData[i + 3] = 255; // A
    }

    // Create red square in the center (255, 0, 0, 255)
    for (int y = 40; y < 90; y++)
    {
        for (int x = 40; x < 90; x++)
        {
            int offset         = y * rowStride + x * 4;
            inData[offset]     = 255; // R
            inData[offset + 1] = 0;   // G
            inData[offset + 2] = 0;   // B
            inData[offset + 3] = 255; // A
        }
    }

    EXPECT_EQ(cudaSuccess, cudaMemcpy(input->basePtr(), inData.data(), sampleStride * N, cudaMemcpyHostToDevice));
    EXPECT_EQ(cudaSuccess, cudaMemset(output->basePtr(), 0, sampleStride * N));

    // Create blur box that covers the edge between red and blue
    std::vector<std::vector<NVCVBlurBoxI>> blurBoxVec;
    std::vector<NVCVBlurBoxI>              boxes;
    NVCVBlurBoxI                           box;
    box.box.x      = 10;
    box.box.y      = 10;
    box.box.width  = 50;
    box.box.height = 50;
    box.kernelSize = 5;
    boxes.push_back(box);
    blurBoxVec.push_back(boxes);

    auto blurBoxes = std::make_shared<NVCVBlurBoxesImpl>(blurBoxVec);

    cvcuda::BoxBlur op;
    EXPECT_NO_THROW(op(stream, imgIn, imgOut, (NVCVBlurBoxesI)blurBoxes.get()));
    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    // Validation strategy: Verify blur occurred by checking edge pixels in blur region
    // Edge pixels should show color blending (neither pure red nor pure blue)
    std::vector<uint8_t> outData(sampleStride * N);
    EXPECT_EQ(cudaSuccess, cudaMemcpy(outData.data(), output->basePtr(), sampleStride * N, cudaMemcpyDeviceToHost));

    // Check a pixel on the edge of the red square inside the blur box (should be blended)
    // At position (42, 42) we expect blurred RGBA values
    int     edgeOffset = 42 * rowStride + 42 * 4;
    uint8_t r          = outData[edgeOffset];
    uint8_t b          = outData[edgeOffset + 2];

    EXPECT_GT(r, 0) << "Blur should blend colors (red component should be present)";
    EXPECT_LT(r, 255) << "Blur should blend colors (not pure red)";
    EXPECT_GT(b, 0) << "Blur should blend colors (blue component should be present)";
    EXPECT_LT(b, 255) << "Blur should blend colors (not pure blue)";

    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpBoxBlur_Smoke, multiple_boxes)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    nvcv::Tensor imgIn  = nvcv::util::CreateTensor(1, 640, 480, nvcv::FMT_RGB8);
    nvcv::Tensor imgOut = nvcv::util::CreateTensor(1, 640, 480, nvcv::FMT_RGB8);

    // Create multiple blur boxes with different kernel sizes
    std::vector<std::vector<NVCVBlurBoxI>> blurBoxVec;
    std::vector<NVCVBlurBoxI>              boxes;

    for (int i = 0; i < 8; i++)
    {
        NVCVBlurBoxI box;
        box.box.x      = i * 70;
        box.box.y      = i * 50;
        box.box.width  = 64;
        box.box.height = 64;
        box.kernelSize = 3 + i * 2; // Varying kernel sizes: 3, 5, 7, 9, ...
        boxes.push_back(box);
    }
    blurBoxVec.push_back(boxes);

    auto blurBoxes = std::make_shared<NVCVBlurBoxesImpl>(blurBoxVec);

    cvcuda::BoxBlur op;
    EXPECT_NO_THROW(op(stream, imgIn, imgOut, (NVCVBlurBoxesI)blurBoxes.get()));
    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpBoxBlur_Smoke, various_kernel_sizes)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    cvcuda::BoxBlur op;

    // Test different kernel sizes
    std::vector<int> kernelSizes = {1, 3, 5, 7, 11, 15, 21};

    for (int ks : kernelSizes) // NOSONAR
    {
        nvcv::Tensor imgIn  = nvcv::util::CreateTensor(1, 224, 224, nvcv::FMT_RGB8);
        nvcv::Tensor imgOut = nvcv::util::CreateTensor(1, 224, 224, nvcv::FMT_RGB8);

        std::vector<std::vector<NVCVBlurBoxI>> blurBoxVec;
        std::vector<NVCVBlurBoxI>              boxes;
        NVCVBlurBoxI                           box;
        box.box.x      = 50;
        box.box.y      = 50;
        box.box.width  = 120;
        box.box.height = 120;
        box.kernelSize = ks;
        boxes.push_back(box);
        blurBoxVec.push_back(boxes);

        auto blurBoxes = std::make_shared<NVCVBlurBoxesImpl>(blurBoxVec);

        EXPECT_NO_THROW(op(stream, imgIn, imgOut, (NVCVBlurBoxesI)blurBoxes.get())) << "Failed with kernel size " << ks;
        EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    }

    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpBoxBlur_Smoke, memory_management)
{
    // Run operator multiple times to verify no memory leaks
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    cvcuda::BoxBlur op;

    for (int iter = 0; iter < 5; iter++) // NOSONAR
    {
        nvcv::Tensor imgIn  = nvcv::util::CreateTensor(1, 320, 240, nvcv::FMT_RGB8);
        nvcv::Tensor imgOut = nvcv::util::CreateTensor(1, 320, 240, nvcv::FMT_RGB8);

        std::vector<std::vector<NVCVBlurBoxI>> blurBoxVec;
        std::vector<NVCVBlurBoxI>              boxes;
        NVCVBlurBoxI                           box;
        box.box.x      = 30 + iter * 20;
        box.box.y      = 30 + iter * 15;
        box.box.width  = 100;
        box.box.height = 80;
        box.kernelSize = 7;
        boxes.push_back(box);
        blurBoxVec.push_back(boxes);

        auto blurBoxes = std::make_shared<NVCVBlurBoxesImpl>(blurBoxVec);

        EXPECT_NO_THROW(op(stream, imgIn, imgOut, (NVCVBlurBoxesI)blurBoxes.get()));
        EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    }

    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpBoxBlur_Smoke, edge_cases)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    nvcv::Tensor imgIn  = nvcv::util::CreateTensor(1, 100, 100, nvcv::FMT_RGB8);
    nvcv::Tensor imgOut = nvcv::util::CreateTensor(1, 100, 100, nvcv::FMT_RGB8);

    cvcuda::BoxBlur op;

    // Empty blur box list
    {
        std::vector<std::vector<NVCVBlurBoxI>> blurBoxVec;
        std::vector<NVCVBlurBoxI>              boxes; // Empty
        blurBoxVec.push_back(boxes);
        auto blurBoxes = std::make_shared<NVCVBlurBoxesImpl>(blurBoxVec);
        EXPECT_NO_THROW(op(stream, imgIn, imgOut, (NVCVBlurBoxesI)blurBoxes.get()));
    }

    // Box at image boundary
    {
        std::vector<std::vector<NVCVBlurBoxI>> blurBoxVec;
        std::vector<NVCVBlurBoxI>              boxes;
        NVCVBlurBoxI                           box;
        box.box.x      = 0;
        box.box.y      = 0;
        box.box.width  = 100;
        box.box.height = 100;
        box.kernelSize = 5;
        boxes.push_back(box);
        blurBoxVec.push_back(boxes);
        auto blurBoxes = std::make_shared<NVCVBlurBoxesImpl>(blurBoxVec);
        EXPECT_NO_THROW(op(stream, imgIn, imgOut, (NVCVBlurBoxesI)blurBoxes.get()));
    }

    // Very small box
    {
        std::vector<std::vector<NVCVBlurBoxI>> blurBoxVec;
        std::vector<NVCVBlurBoxI>              boxes;
        NVCVBlurBoxI                           box;
        box.box.x      = 45;
        box.box.y      = 45;
        box.box.width  = 10;
        box.box.height = 10;
        box.kernelSize = 3;
        boxes.push_back(box);
        blurBoxVec.push_back(boxes);
        auto blurBoxes = std::make_shared<NVCVBlurBoxesImpl>(blurBoxVec);
        EXPECT_NO_THROW(op(stream, imgIn, imgOut, (NVCVBlurBoxesI)blurBoxes.get()));
    }

    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpBoxBlur_Smoke, batch_processing)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    const int    batchSize = 4;
    nvcv::Tensor imgIn     = nvcv::util::CreateTensor(batchSize, 256, 256, nvcv::FMT_RGB8);
    nvcv::Tensor imgOut    = nvcv::util::CreateTensor(batchSize, 256, 256, nvcv::FMT_RGB8);

    // Create blur boxes for each batch item
    std::vector<std::vector<NVCVBlurBoxI>> blurBoxVec;
    for (int b = 0; b < batchSize; b++)
    {
        std::vector<NVCVBlurBoxI> boxes;
        NVCVBlurBoxI              box;
        box.box.x      = 50;
        box.box.y      = 50;
        box.box.width  = 150;
        box.box.height = 150;
        box.kernelSize = 9;
        boxes.push_back(box);
        blurBoxVec.push_back(boxes);
    }

    auto blurBoxes = std::make_shared<NVCVBlurBoxesImpl>(blurBoxVec);

    cvcuda::BoxBlur op;
    EXPECT_NO_THROW(op(stream, imgIn, imgOut, (NVCVBlurBoxesI)blurBoxes.get()));
    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}
