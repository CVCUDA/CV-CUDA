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
#include <nvcv/ImageBatch.hpp>

#include <list>
#include <random>

#include <nvcv/Fwd.hpp>

namespace t    = ::testing;
namespace test = nvcv::test;

static bool operator==(const NVCVImagePlaneStrided &a, const NVCVImagePlaneStrided &b)
{
    return a.width == b.width && a.height == b.height && a.rowStride == b.rowStride && a.basePtr == b.basePtr;
}

static bool operator==(const NVCVImageBufferStrided &a, const NVCVImageBufferStrided &b)
{
    if (a.numPlanes != b.numPlanes)
    {
        return false;
    }
    else
    {
        for (int i = 0; i != a.numPlanes; ++i)
        {
            if (a.planes[i] != b.planes[i])
            {
                return false;
            }
        }
    }
    return true;
}

static std::ostream &operator<<(std::ostream &out, const NVCVImagePlaneStrided &a)
{
    return out << a.width << 'x' << a.height << '@' << a.rowStride << ':' << a.basePtr;
}

static std::ostream &operator<<(std::ostream &out, const NVCVImageBufferStrided &a)
{
    out << "{";
    for (int i = 0; i < a.numPlanes; ++i)
    {
        if (i > 0)
        {
            out << ',';
        }
        out << a.planes[i];
    }
    return out;
}

TEST(ImageBatchVarShape, smoke_create)
{
    nvcv::ImageBatchVarShape batch(100);

    EXPECT_EQ(100, batch.capacity());
    EXPECT_EQ(0, batch.numImages());
    ASSERT_NE(nullptr, batch.handle());

    NVCVTypeImageBatch type;
    ASSERT_EQ(NVCV_SUCCESS, nvcvImageBatchGetType(batch.handle(), &type));
    EXPECT_EQ(NVCV_TYPE_IMAGEBATCH_VARSHAPE, type);

    // empty data
    {
        nvcv::ImageBatchData data = batch.exportData(0);

        auto devdata = data.cast<nvcv::ImageBatchVarShapeDataStridedCuda>();
        ASSERT_NE(nvcv::NullOpt, devdata);

        ASSERT_EQ(0, devdata->numImages());
        EXPECT_NE(nullptr, devdata->imageList());
        EXPECT_NE(nullptr, devdata->formatList());

        EXPECT_EQ(nvcv::Size2D(0, 0), devdata->maxSize());
        EXPECT_EQ(devdata->maxSize(), batch.maxSize());
        ASSERT_EQ(nvcv::FMT_NONE, devdata->uniqueFormat());
        ASSERT_EQ(devdata->uniqueFormat(), batch.uniqueFormat());
    }

    std::vector<NVCVImageBufferStrided> goldImages;
    std::vector<NVCVImageFormat>        goldFormats;
    std::vector<NVCVImageHandle>        goldHandles;

    auto addToGold = [&goldImages, &goldFormats, &goldHandles](const nvcv::Image &img)
    {
        auto imgdata = img.exportData<nvcv::ImageDataStridedCuda>();
        EXPECT_NE(nvcv::NullOpt, imgdata);
        if (imgdata)
        {
            goldImages.push_back(imgdata->cdata().buffer.strided);
            goldFormats.push_back(imgdata->cdata().format);
            goldHandles.push_back(img.handle());
        }
    };

    auto calcMaxSize = [&goldImages]()
    {
        nvcv::Size2D maxSize = {0, 0};
        for (size_t i = 0; i < goldImages.size(); ++i)
        {
            maxSize.w = std::max(maxSize.w, goldImages[i].planes[0].width);
            maxSize.h = std::max(maxSize.h, goldImages[i].planes[0].height);
        }
        return maxSize;
    };

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    // Add a bunch of images, using different methods

    nvcv::Image img0{
        {320, 128},
        nvcv::FMT_NV12
    };
    batch.pushBack(img0);
    addToGold(img0);

    nvcv::Image img1{
        {320, 128},
        nvcv::FMT_NV12
    };
    batch.pushBack(&img1, &img1 + 1);
    addToGold(img1);

    std::list<nvcv::Image> vec0;
    for (int i = 0; i < 10; ++i)
    {
        vec0.emplace_back(nvcv::Size2D{328 + i * 2, 130 - i * 2}, nvcv::FMT_NV12);
        addToGold(vec0.back());
    }
    batch.pushBack(vec0.begin(), vec0.end());

    std::vector<nvcv::Image> vec1;
    for (int i = 0; i < 10; ++i)
    {
        vec1.emplace_back(nvcv::Image(nvcv::Size2D{328 + i * 2, 130 - i * 2}, nvcv::FMT_NV12));
        addToGold(vec1.back());
    }
    batch.pushBack(vec1.begin(), vec1.end());

    // To synchronize buffers
    nvcv::ImageBatchVarShapeData vsdata  = batch.exportData(stream); // test output type
    auto                         devdata = vsdata.cast<nvcv::ImageBatchVarShapeDataStridedCuda>();
    ASSERT_NE(nvcv::NullOpt, devdata);
    EXPECT_EQ(calcMaxSize(), devdata->maxSize());
    EXPECT_EQ(devdata->maxSize(), batch.maxSize());

    std::vector<nvcv::Image> vec2;
    for (int i = 0; i < 10; ++i)
    {
        vec2.emplace_back(nvcv::Image(nvcv::Size2D{328 + i * 2, 130 - i * 2}, nvcv::FMT_NV12));
        addToGold(vec2.back());
    }
    batch.pushBack(vec2.begin(), vec2.end());

    std::vector<std::reference_wrapper<nvcv::Image>> vec3;
    for (nvcv::Image &img : vec0)
    {
        vec3.emplace_back(img);
        addToGold(vec3.back().get());
    }
    batch.pushBack(vec3.begin(), vec3.end());

    // Remove some
    batch.popBack(5);
    goldImages.erase(goldImages.end() - 5, goldImages.end());
    goldFormats.erase(goldFormats.end() - 5, goldFormats.end());
    goldHandles.erase(goldHandles.end() - 5, goldHandles.end());

    // To synchronize buffers
    devdata = batch.exportData<nvcv::ImageBatchVarShapeDataStridedCuda>(stream);
    ASSERT_NE(nullptr, devdata);
    EXPECT_EQ(calcMaxSize(), devdata->maxSize());
    EXPECT_EQ(devdata->maxSize(), batch.maxSize());

    // So far we have all images with the same format
    ASSERT_EQ(nvcv::FMT_NV12, devdata->uniqueFormat());
    ASSERT_EQ(devdata->uniqueFormat(), batch.uniqueFormat());
    EXPECT_THAT(std::make_tuple(devdata->hostFormatList(), devdata->numImages()),
                t::Each(static_cast<NVCVImageFormat>(devdata->uniqueFormat())));

    // Add one image with a different format
    nvcv::Image imgRGBA8({42, 59}, nvcv::FMT_RGBA8);
    addToGold(imgRGBA8);
    batch.pushBack(imgRGBA8);

    // use callback
    batch.pushBack(
        [&]() -> nvcv::Image
        {
            int i = batch.numImages();
            if (i < 5)
            {
                nvcv::Image img(nvcv::Size2D{320 + i * 2, 122 - i * 2}, nvcv::FMT_NV12);
                addToGold(img);
                return img;
            }
            else
            {
                return {};
            }
        });

    // not-empty data
    {
        auto data = batch.exportData(stream);

        auto devdata = data.cast<nvcv::ImageBatchVarShapeDataStridedCuda>();
        ASSERT_NE(nvcv::NullOpt, devdata);

        ASSERT_EQ(devdata->uniqueFormat(), batch.uniqueFormat());

        ASSERT_EQ(goldHandles.size(), devdata->numImages());
        EXPECT_NE(nullptr, devdata->imageList());
        EXPECT_NE(nullptr, devdata->formatList());
        EXPECT_NE(nullptr, devdata->hostFormatList());

        EXPECT_EQ(calcMaxSize(), devdata->maxSize());
        EXPECT_EQ(devdata->maxSize(), batch.maxSize());

        std::vector<NVCVImageBufferStrided> images(devdata->numImages());
        ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(images.data(), devdata->imageList(), sizeof(images[0]) * images.size(),
                                               cudaMemcpyDeviceToHost, stream));

        std::vector<NVCVImageFormat> formats(devdata->numImages());
        ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(formats.data(), devdata->formatList(),
                                               sizeof(formats[0]) * formats.size(), cudaMemcpyDeviceToHost, stream));

        ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

        EXPECT_THAT(images, t::ElementsAreArray(goldImages));
        EXPECT_THAT(formats, t::ElementsAreArray(goldFormats));
        EXPECT_THAT(std::make_tuple(devdata->hostFormatList(), devdata->numImages()), t::ElementsAreArray(goldFormats));

        int cur = 0;
        for (auto it = batch.begin(); it != batch.end(); ++it, ++cur)
        {
            ASSERT_LT(cur, (int)goldHandles.size());
            EXPECT_EQ(goldHandles[cur], it->handle()) << "Image #" << cur;
        }

        for (int i = 0; i < batch.numImages(); ++i)
        {
            EXPECT_EQ(goldHandles[i], batch[i].handle()) << "Image #" << i;
        }
    }

    {
        nvcv::ImageBatchVarShapeWrapHandle wrap(batch.handle());
        const nvcv::ImageBatchVarShape    &wrapref = wrap;
        EXPECT_EQ(batch.capacity(), wrapref.capacity());
        ASSERT_EQ(batch.numImages(), wrapref.numImages());
        EXPECT_EQ(batch.handle(), wrapref.handle());

        int  cur    = 0;
        auto itwrap = wrapref.begin();
        for (auto itgold = batch.begin(); itgold != batch.end(); ++itgold, ++itwrap, ++cur)
        {
            EXPECT_EQ(itgold->handle(), itwrap->handle()) << "Image #" << cur;
        }
    }

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(ImageBatchVarShape, smoke_sync)
{
    std::vector<NVCVImageBufferStrided> goldImages;
    std::vector<NVCVImageFormat>        goldFormats;
    std::vector<NVCVImageHandle>        goldHandles;

    auto addToGold = [&goldImages, &goldFormats, &goldHandles](const nvcv::Image &img)
    {
        auto imgdata = img.exportData<nvcv::ImageDataStridedCuda>();
        EXPECT_NE(nvcv::NullOpt, imgdata);
        if (imgdata)
        {
            goldImages.push_back(imgdata->cdata().buffer.strided);
            goldFormats.push_back(imgdata->format());
            goldHandles.push_back(img.handle());
        }
    };

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    nvcv::ImageBatchVarShape batch(10000);

    std::mt19937                  rng(123);
    std::uniform_int_distribution rnd(1, 4);

    std::list<nvcv::Image> vec0;
    for (int i = 0; i < batch.capacity(); ++i)
    {
        vec0.emplace_back(nvcv::Size2D{rnd(rng) * 2, rnd(rng) * 2}, nvcv::FMT_NV12);
        addToGold(vec0.back());
    }

    std::list<nvcv::Image>       vec1;
    std::vector<NVCVImageHandle> vec1Handles;
    for (int i = 0; i < batch.capacity(); ++i)
    {
        vec1.emplace_back(nvcv::Size2D{rnd(rng) * 2, rnd(rng) * 2}, nvcv::FMT_NV12);
        vec1Handles.push_back(vec1.back().handle());
    }

    batch.pushBack(vec0.begin(), vec0.end());

    // trigger host->dev async copy
    auto devdata = batch.exportData<nvcv::ImageBatchVarShapeDataStridedCuda>(stream);

    // Re-write batch contents in host-side, must have waited
    // until async copy finishes
    batch.clear();
    ASSERT_EQ(NVCV_SUCCESS, nvcvImageBatchVarShapePushImages(batch.handle(), vec1Handles.data(), vec1Handles.size()));

    // test device buffer against gold, new data from vec1 must not
    // show up
    {
        ASSERT_EQ(goldHandles.size(), devdata->numImages());
        EXPECT_NE(nullptr, devdata->imageList());
        EXPECT_NE(nullptr, devdata->formatList());

        std::vector<NVCVImageBufferStrided> images(devdata->numImages());
        ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(images.data(), devdata->imageList(), sizeof(images[0]) * images.size(),
                                               cudaMemcpyDeviceToHost, stream));
        ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

        EXPECT_THAT(images, t::ElementsAreArray(goldImages));
    }

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(ImageBatch, smoke_user_pointer)
{
    nvcv::ImageBatchVarShape batch(3);
    EXPECT_EQ(nullptr, batch.userPointer());

    batch.setUserPointer((void *)0x123);
    EXPECT_EQ((void *)0x123, batch.userPointer());

    batch.setUserPointer(nullptr);
    EXPECT_EQ(nullptr, batch.userPointer());
}

TEST(ImageBatch, smoke_cast)
{
    // Now when we create the object via C API

    NVCVImageBatchHandle               handle;
    NVCVImageBatchVarShapeRequirements reqs;
    ASSERT_EQ(NVCV_SUCCESS, nvcvImageBatchVarShapeCalcRequirements(5, &reqs));
    ASSERT_EQ(NVCV_SUCCESS, nvcvImageBatchVarShapeConstruct(&reqs, nullptr, &handle));
    int ref;
    ASSERT_EQ(NVCV_SUCCESS, nvcvImageBatchRefCount(handle, &ref));
    EXPECT_EQ(ref, 1);

    auto             h = handle;
    nvcv::ImageBatch img(std::move(handle));
    EXPECT_EQ(h, img.handle());
    EXPECT_EQ(5, img.capacity());

    auto h2 = h;
    ASSERT_EQ(NVCV_SUCCESS, nvcvImageBatchIncRef(h2, &ref));
    EXPECT_EQ(ref, 2);
    EXPECT_NO_THROW(nvcv::ImageBatchVarShape{std::move(h2)});

    ref = img.reset();
    EXPECT_EQ(ref, 0);
}
