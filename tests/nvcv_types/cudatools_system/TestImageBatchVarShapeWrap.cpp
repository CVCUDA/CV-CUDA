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

#include "DeviceImageBatchVarShapeWrap.hpp" // to test in the device

#include <common/MixTypedTests.hpp>             // for NVCV_MIXTYPED_TEST_SUITE_P, etc.
#include <nvcv/ImageBatch.hpp>                  // for ImageBatchVarShape, etc.
#include <nvcv/cuda/ImageBatchVarShapeWrap.hpp> // for ImageBatchVarShapeWrap, etc.
#include <nvcv/cuda/MathOps.hpp>                // for operator == to allow EXPECT_EQ

#include <list>
#include <random>
#include <vector>

namespace cuda  = nvcv::cuda;
namespace ttype = nvcv::test::type;

// --------------------- Testing ImageBatchVarShapeWrap ------------------------

#define NVCV_TEST_ROW(WIDTH, HEIGHT, VARSIZE, SAMPLES, FORMAT, TYPE)                                      \
    ttype::Types<ttype::Value<WIDTH>, ttype::Value<HEIGHT>, ttype::Value<VARSIZE>, ttype::Value<SAMPLES>, \
                 ttype::Value<FORMAT>, TYPE>

NVCV_MIXTYPED_TEST_SUITE(ImageBatchVarShapeWrapTest,
                         ttype::Types<NVCV_TEST_ROW(55, 12, 0, 1, NVCV_IMAGE_FORMAT_RGBA8, uchar4),
                                      NVCV_TEST_ROW(66, 23, 0, 2, NVCV_IMAGE_FORMAT_RGBA8, uchar4),
                                      NVCV_TEST_ROW(13, 19, 9, 3, NVCV_IMAGE_FORMAT_RGB8, uchar3),
                                      NVCV_TEST_ROW(513, 233, 88, 65, NVCV_IMAGE_FORMAT_RGB8, uchar3),
                                      NVCV_TEST_ROW(33, 109, 7, 4, NVCV_IMAGE_FORMAT_RGB8p, uchar1),
                                      NVCV_TEST_ROW(133, 36, 13, 5, NVCV_IMAGE_FORMAT_RGBA8p, uchar1),
                                      NVCV_TEST_ROW(222, 129, 29, 6, NVCV_IMAGE_FORMAT_U8, uchar1),
                                      NVCV_TEST_ROW(2, 2, 1, 5, NVCV_IMAGE_FORMAT_S8, char1),
                                      NVCV_TEST_ROW(333, 44, 22, 4, NVCV_IMAGE_FORMAT_2S16, short2),
                                      NVCV_TEST_ROW(111, 22, 8, 3, NVCV_IMAGE_FORMAT_S32, int1),
                                      NVCV_TEST_ROW(123, 212, 33, 2, NVCV_IMAGE_FORMAT_RGBf32, float3),
                                      NVCV_TEST_ROW(12, 12, 11, 123, NVCV_IMAGE_FORMAT_RGBf32, float3),
                                      NVCV_TEST_ROW(321, 112, 13, 3, NVCV_IMAGE_FORMAT_RGBAf32, float4),
                                      NVCV_TEST_ROW(14, 14, 13, 199, NVCV_IMAGE_FORMAT_RGBAf32, float4),
                                      NVCV_TEST_ROW(321, 112, 13, 3, NVCV_IMAGE_FORMAT_RGBAf32p, float1),
                                      NVCV_TEST_ROW(12, 21, 3, 2, NVCV_IMAGE_FORMAT_RGBf32p, float1)>);

#undef NVCV_TEST_ROW

NVCV_MIXTYPED_TEST(ImageBatchVarShapeWrapTest, correct_content)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    int width   = GetValue<0>();
    int height  = GetValue<1>();
    int varSize = GetValue<2>();
    int samples = GetValue<3>();

    nvcv::ImageFormat format{GetValue<4>()};

    using T = GetType<5>;

    nvcv::ImageBatchVarShape imageBatch(samples);

    std::default_random_engine         randEng{0};
    std::uniform_int_distribution<int> rand{-varSize, varSize};

    std::list<nvcv::Image> imageList;

    for (int i = 0; i < imageBatch.capacity(); ++i)
    {
        imageList.emplace_back(nvcv::Size2D{width + rand(randEng), height + rand(randEng)}, format);
    }

    imageBatch.pushBack(imageList.begin(), imageList.end());

    auto imageBatchData = imageBatch.exportData<nvcv::ImageBatchVarShapeDataStridedCuda>(stream);
    ASSERT_NE(imageBatchData, nullptr);

    int3 maxSize{imageBatchData->maxSize().w, imageBatchData->maxSize().h, imageBatchData->numImages()};

    EXPECT_EQ(cuda::ImageBatchVarShapeWrap<T>::kNumDimensions, 4);
    EXPECT_EQ(cuda::ImageBatchVarShapeWrap<T>::kVariableStrides, 3);
    EXPECT_EQ(cuda::ImageBatchVarShapeWrap<T>::kConstantStrides, 1);

    cuda::ImageBatchVarShapeWrap<T> wrap(*imageBatchData);

    DeviceSetTwos(wrap, maxSize, stream);

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    std::vector<NVCVImageBufferStrided> testPlanes(imageBatchData->numImages());
    ASSERT_EQ(cudaSuccess, cudaMemcpy(testPlanes.data(), imageBatchData->imageList(),
                                      sizeof(testPlanes[0]) * testPlanes.size(), cudaMemcpyDeviceToHost));

    for (int s = 0; s < samples; s++)
    {
        void *testBuffer = testPlanes[s].planes[0].basePtr;

        int width  = testPlanes[s].planes[0].width;
        int height = testPlanes[s].planes[0].height;

        int rowStride = testPlanes[s].planes[0].rowStride;
        int sizeBytes = rowStride * height;

        std::vector<uint8_t> test(sizeBytes);
        std::vector<uint8_t> gold(sizeBytes);

        ASSERT_EQ(cudaSuccess, cudaMemcpy(test.data(), testBuffer, sizeBytes, cudaMemcpyDeviceToHost));

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                *reinterpret_cast<T *>(&gold[y * rowStride + x * sizeof(T)]) = cuda::SetAll<T>(2);
            }
        }

        EXPECT_EQ(test, gold);
    }
}

// ------------------- Testing ImageBatchVarShapeWrapNHWC ----------------------

#define NVCV_TEST_ROW(WIDTH, HEIGHT, VARSIZE, SAMPLES, FORMAT, TYPE, NUMCHANNELS)                         \
    ttype::Types<ttype::Value<WIDTH>, ttype::Value<HEIGHT>, ttype::Value<VARSIZE>, ttype::Value<SAMPLES>, \
                 ttype::Value<FORMAT>, TYPE, ttype::Value<NUMCHANNELS>>

NVCV_MIXTYPED_TEST_SUITE(ImageBatchVarShapeWrapNHWCTest,
                         ttype::Types<NVCV_TEST_ROW(111, 223, 21, 4, NVCV_IMAGE_FORMAT_RGB8, uchar1, 3),
                                      NVCV_TEST_ROW(3, 3, 1, 6, NVCV_IMAGE_FORMAT_RGBA8, uchar1, 4),
                                      NVCV_TEST_ROW(3, 3, 1, 7, NVCV_IMAGE_FORMAT_U8, uchar1, 1),
                                      NVCV_TEST_ROW(123, 212, 13, 3, NVCV_IMAGE_FORMAT_F32, float1, 1),
                                      NVCV_TEST_ROW(123, 212, 13, 3, NVCV_IMAGE_FORMAT_RGBf32, float1, 3),
                                      NVCV_TEST_ROW(21, 12, 3, 2, NVCV_IMAGE_FORMAT_RGBAf32, float1, 4)>);

#undef NVCV_TEST_ROW

NVCV_MIXTYPED_TEST(ImageBatchVarShapeWrapNHWCTest, correct_content)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    int width   = GetValue<0>();
    int height  = GetValue<1>();
    int varSize = GetValue<2>();
    int samples = GetValue<3>();

    nvcv::ImageFormat format{GetValue<4>()};

    using T = GetType<5>;

    int numChannels = GetValue<6>();

    nvcv::ImageBatchVarShape imageBatch(samples);

    std::default_random_engine         randEng{0};
    std::uniform_int_distribution<int> rand{-varSize, varSize};

    std::list<nvcv::Image> imageList;

    for (int i = 0; i < imageBatch.capacity(); ++i)
    {
        imageList.emplace_back(nvcv::Size2D{width + rand(randEng), height + rand(randEng)}, format);
    }

    imageBatch.pushBack(imageList.begin(), imageList.end());

    auto imageBatchData = imageBatch.exportData<nvcv::ImageBatchVarShapeDataStridedCuda>(stream);
    ASSERT_NE(imageBatchData, nullptr);

    int3 maxSize{imageBatchData->maxSize().w, imageBatchData->maxSize().h, imageBatchData->numImages()};

    EXPECT_EQ(cuda::ImageBatchVarShapeWrapNHWC<T>::kNumDimensions, 4);
    EXPECT_EQ(cuda::ImageBatchVarShapeWrapNHWC<T>::kVariableStrides, 3);
    EXPECT_EQ(cuda::ImageBatchVarShapeWrapNHWC<T>::kConstantStrides, 1);

    cuda::ImageBatchVarShapeWrapNHWC<T> wrap(*imageBatchData, numChannels);

    EXPECT_EQ(wrap.numChannels(), numChannels);

    DeviceSetTwos(wrap, maxSize, stream);

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    std::vector<NVCVImageBufferStrided> testPlanes(imageBatchData->numImages());
    ASSERT_EQ(cudaSuccess, cudaMemcpy(testPlanes.data(), imageBatchData->imageList(),
                                      sizeof(testPlanes[0]) * testPlanes.size(), cudaMemcpyDeviceToHost));

    for (int s = 0; s < samples; s++)
    {
        void *testBuffer = testPlanes[s].planes[0].basePtr;

        int width  = testPlanes[s].planes[0].width;
        int height = testPlanes[s].planes[0].height;

        int rowStride = testPlanes[s].planes[0].rowStride;
        int sizeBytes = rowStride * height;

        std::vector<std::byte> test(sizeBytes);
        std::vector<std::byte> gold(sizeBytes);

        ASSERT_EQ(cudaSuccess, cudaMemcpy(test.data(), testBuffer, sizeBytes, cudaMemcpyDeviceToHost));

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                for (int k = 0; k < numChannels; k++)
                {
                    *reinterpret_cast<T *>(&gold[y * rowStride + x * sizeof(T) * numChannels + k * sizeof(T)])
                        = cuda::SetAll<T>(2);
                }
            }
        }

        EXPECT_EQ(test, gold);
    }
}
