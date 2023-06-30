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

#include <nvcv/Image.hpp>
#include <nvcv/alloc/Allocator.hpp>

#include <nvcv/Fwd.hpp>

TEST(Image, smoke_create)
{
    nvcv::Image img({163, 117}, nvcv::FMT_RGBA8);

    EXPECT_EQ(nvcv::Size2D(163, 117), img.size());
    EXPECT_EQ(nvcv::FMT_RGBA8, img.format());
    ASSERT_NE(nullptr, img.handle());

    NVCVTypeImage type;
    ASSERT_EQ(NVCV_SUCCESS, nvcvImageGetType(img.handle(), &type));
    EXPECT_EQ(NVCV_TYPE_IMAGE, type);

    auto data    = img.exportData();
    auto devdata = data.cast<nvcv::ImageDataStridedCuda>();
    ASSERT_NE(nvcv::NullOpt, devdata);

    ASSERT_EQ(1, devdata->numPlanes());
    EXPECT_EQ(img.format(), devdata->format());
    EXPECT_EQ(img.size(), devdata->size());
    EXPECT_EQ(img.size().w, devdata->plane(0).width);
    EXPECT_EQ(img.size().h, devdata->plane(0).height);
    EXPECT_LE(163 * 4, devdata->plane(0).rowStride);
    EXPECT_NE(nullptr, devdata->plane(0).basePtr);

    const nvcv::ImagePlaneStrided &plane = devdata->plane(0);

    EXPECT_EQ(cudaSuccess, cudaMemset2D(plane.basePtr, plane.rowStride, 123, plane.width * 4, plane.height));
}

TEST(Image, smoke_cast)
{
    NVCVImageHandle       handle;
    NVCVImageRequirements reqs;
    ASSERT_EQ(NVCV_SUCCESS, nvcvImageCalcRequirements(163, 117, NVCV_IMAGE_FORMAT_RGBA8, 0, 0, &reqs));
    ASSERT_EQ(NVCV_SUCCESS, nvcvImageConstruct(&reqs, nullptr, &handle));
    int ref;
    ASSERT_EQ(NVCV_SUCCESS, nvcvImageRefCount(handle, &ref));
    EXPECT_EQ(ref, 1);

    auto        h = handle;
    nvcv::Image img(std::move(handle));
    EXPECT_EQ(h, img.handle());
    EXPECT_EQ(163, img.size().w);
    EXPECT_EQ(117, img.size().h);
    EXPECT_EQ(nvcv::FMT_RGBA8, img.format());

    ref = img.reset();
    EXPECT_EQ(ref, 0);
}

TEST(Image, smoke_user_pointer)
{
    nvcv::Image img({163, 117}, nvcv::FMT_RGBA8);
    EXPECT_EQ(nullptr, img.userPointer());

    void *userPtr;
    ASSERT_EQ(NVCV_SUCCESS, nvcvImageGetUserPointer(img.handle(), &userPtr));
    EXPECT_EQ(nullptr, userPtr);

    img.setUserPointer((void *)0x123);
    EXPECT_EQ((void *)0x123, img.userPointer());

    ASSERT_EQ(NVCV_SUCCESS, nvcvImageGetUserPointer(img.handle(), &userPtr));
    EXPECT_EQ((void *)0x123, userPtr);

    img.setUserPointer(nullptr);
    EXPECT_EQ(nullptr, img.userPointer());

    ASSERT_EQ(NVCV_SUCCESS, nvcvImageGetUserPointer(img.handle(), &userPtr));
    EXPECT_EQ(nullptr, userPtr);
}

TEST(Image, smoke_create_managed)
{
    ;

    int64_t setBufLen   = 0;
    int32_t setBufAlign = 0;

    // clang-format off
    nvcv::CustomAllocator managedAlloc
    {
        nvcv::CustomCudaMemAllocator
        {
            [&setBufLen, &setBufAlign](int64_t size, int32_t bufAlign)
            {
                setBufLen = size;
                setBufAlign = bufAlign;

                 void *ptr = nullptr;
                 cudaMallocManaged(&ptr, size);
                 return ptr;
            },
            [](void *ptr, int64_t bufLen, int32_t bufAlign)
            {
                cudaFree(ptr);
            }
        }
    };
    // clang-format on

    nvcv::Image img({163, 117}, nvcv::FMT_RGBA8, managedAlloc,
                    nvcv::MemAlignment{}.rowAddr(1).baseAddr(32)); // packed rows
    EXPECT_EQ(32, setBufAlign);

    nvcv::Optional<nvcv::ImageData> data = img.exportData();
    ASSERT_NE(nvcv::NullOpt, data);

    auto devdata = data->cast<nvcv::ImageDataStridedCuda>();
    ASSERT_NE(nvcv::NullOpt, devdata);

    ASSERT_EQ(1, devdata->numPlanes());
    EXPECT_LE(163 * 4, devdata->plane(0).rowStride);
    EXPECT_NE(nullptr, devdata->plane(0).basePtr);

    const nvcv::ImagePlaneStrided &plane = devdata->plane(0);

    EXPECT_EQ(cudaSuccess, cudaMemset2D(plane.basePtr, plane.rowStride, 123, plane.width * 4, plane.height));

    for (int i = 0; i < plane.height; ++i)
    {
        std::byte *beg = reinterpret_cast<std::byte *>(plane.basePtr) + plane.rowStride * i;
        std::byte *end = beg + plane.width * 4;

        ASSERT_EQ(end, std::find_if(beg, end, [](std::byte b) { return b != std::byte{123}; }))
            << "All bytes in the image must be 123";
    }
}

TEST(ImageWrapData, smoke_create)
{
    nvcv::ImageDataStridedCuda::Buffer buf;
    buf.numPlanes           = 1;
    buf.planes[0].width     = 173;
    buf.planes[0].height    = 79;
    buf.planes[0].rowStride = 190;
    buf.planes[0].basePtr   = reinterpret_cast<NVCVByte *>(678);

    auto img = nvcv::ImageWrapData(nvcv::ImageDataStridedCuda{nvcv::FMT_U8, buf});

    EXPECT_EQ(nvcv::Size2D(173, 79), img.size());
    EXPECT_EQ(nvcv::FMT_U8, img.format());
    ASSERT_NE(nullptr, img.handle());

    NVCVTypeImage type;
    ASSERT_EQ(NVCV_SUCCESS, nvcvImageGetType(img.handle(), &type));
    EXPECT_EQ(NVCV_TYPE_IMAGE_WRAPDATA, type);

    nvcv::Optional<nvcv::ImageData> data = img.exportData();
    ASSERT_NE(nvcv::NullOpt, data);

    auto devdata = data->cast<nvcv::ImageDataStridedCuda>();
    ASSERT_NE(nvcv::NullOpt, devdata);

    ASSERT_EQ(1, devdata->numPlanes());
    EXPECT_EQ(img.format(), devdata->format());
    EXPECT_EQ(img.size(), devdata->size());
    EXPECT_EQ(img.size().w, devdata->plane(0).width);
    EXPECT_EQ(img.size().h, devdata->plane(0).height);
    EXPECT_LE(190, devdata->plane(0).rowStride);
    EXPECT_EQ(buf.planes[0].basePtr, devdata->plane(0).basePtr);
}

TEST(ImageWrapData, smoke_user_pointer)
{
    nvcv::ImageDataStridedCuda::Buffer buf;
    buf.numPlanes           = 1;
    buf.planes[0].width     = 173;
    buf.planes[0].height    = 79;
    buf.planes[0].rowStride = 190;
    buf.planes[0].basePtr   = reinterpret_cast<NVCVByte *>(678);

    auto img = nvcv::ImageWrapData(nvcv::ImageDataStridedCuda{nvcv::FMT_U8, buf});

    EXPECT_EQ(nullptr, img.userPointer());

    img.setUserPointer((void *)0x123);
    EXPECT_EQ((void *)0x123, img.userPointer());

    img.setUserPointer(nullptr);
    EXPECT_EQ(nullptr, img.userPointer());
}

TEST(Image, smoke_operator)
{
    ;

    nvcv::Image in{
        {512, 256},
        nvcv::FMT_RGBA8
    };
    nvcv::Image out{
        {512, 256},
        nvcv::FMT_RGBA8
    };

    auto inData  = in.exportData<nvcv::ImageDataStridedCuda>();
    auto outData = out.exportData<nvcv::ImageDataStridedCuda>();

    if (!inData || !outData)
    {
        throw std::runtime_error("Input and output images must have cuda-accessible pitch-linear memory");
    }
    if (inData->format() != outData->format())
    {
        throw std::runtime_error("Input and output images must have same format");
    }
    if (inData->size() != outData->size())
    {
        throw std::runtime_error("Input and output images must have same size");
    }

    assert(inData->numPlanes() == outData->numPlanes());

    for (int p = 0; p < inData->numPlanes(); ++p)
    {
        const nvcv::ImagePlaneStrided &inPlane  = inData->plane(p);
        const nvcv::ImagePlaneStrided &outPlane = outData->plane(p);

        cudaMemcpy2D(outPlane.basePtr, outPlane.rowStride, inPlane.basePtr, inPlane.rowStride,
                     (inData->format().planeBitsPerPixel(p) + 7) / 8 * inPlane.width, inPlane.height,
                     cudaMemcpyDeviceToDevice);
    }
}

TEST(ImageWrapData, smoke_cleanup)
{
    nvcv::ImageDataStridedCuda::Buffer buf;
    buf.numPlanes           = 1;
    buf.planes[0].width     = 173;
    buf.planes[0].height    = 79;
    buf.planes[0].rowStride = 190;
    buf.planes[0].basePtr   = reinterpret_cast<NVCVByte *>(678);

    int  cleanupCalled = 0;
    auto cleanup       = [&cleanupCalled](const nvcv::ImageData &data)
    {
        ++cleanupCalled;
    };

    {
        auto img = nvcv::ImageWrapData(nvcv::ImageDataStridedCuda{nvcv::FMT_U8, buf}, cleanup);
        EXPECT_EQ(0, cleanupCalled);
    }
    EXPECT_EQ(1, cleanupCalled) << "Cleanup must have been called when img got destroyed";
}

TEST(ImageWrapData, smoke_mem_reqs)
{
    nvcv::Image::Requirements reqs = nvcv::Image::CalcRequirements({512, 256}, nvcv::FMT_NV12);

    nvcv::Image img(reqs);

    EXPECT_EQ(512, img.size().w);
    EXPECT_EQ(256, img.size().h);
    EXPECT_EQ(nvcv::FMT_NV12, img.format());

    auto data = img.exportData<nvcv::ImageDataStridedCuda>();

    ASSERT_NE(nvcv::NullOpt, data);
    ASSERT_EQ(2, data->numPlanes());
    EXPECT_EQ(512, data->plane(0).width);
    EXPECT_EQ(256, data->plane(0).height);

    EXPECT_EQ(256, data->plane(1).width);
    EXPECT_EQ(128, data->plane(1).height);

    EXPECT_EQ(data->plane(1).basePtr, data->plane(0).basePtr + data->plane(0).rowStride * 256);

    for (int p = 0; p < 2; ++p)
    {
        EXPECT_EQ(cudaSuccess,
                  cudaMemset2D(data->plane(p).basePtr, data->plane(p).rowStride, 123,
                               data->plane(p).width * img.format().planePixelStrideBytes(p), data->plane(p).height))
            << "Plane " << p;
    }
}

// Future API ideas
#if 0
TEST(Image, smoke_image_managed_memory)
{
    ;

    nvcv::CustomAllocator managedAlloc
    {
        nvcv::CustomCudaMemAllocator
        {
            [](int64_t size, int32_t)
            {
                void *ptr = nullptr;
                cudaMallocManaged(&ptr, size);
                return ptr;
            },
            [](void *ptr, int64_t, int32_t)
            {
                cudaFree(ptr);
            }
        }
    };

    nvcv::Image img({512, 256}, nvcv::FMT_RGBA8, &managedAlloc);

    EXPECT_EQ(nvcv::Size2D(512,256), img.size());
    EXPECT_EQ(nvcv::FMT_RGBA8, img.format());

    {
        nvcv::LockImageData lkData = img.lock(nvcv::READ);
        if(auto data = lkData->data<nvcv::ImageDataCudaMem>())
        {
            nvcv::GpuMat ocvGPU{data->size.h, data->size.w,
                              data->plane(0).buffer,
                              data->plane(0).rowStride};
            // ...

        }
        else if(auto data = lkData->data<nvcv::ImageDataCudaArray>())
        {
            cudaArray_t carray = data->plane(0);
            // ...
        }
    }

    if(nvcv::LockImageData lkData = img.lock<nvcv::ImageDataCudaMem>(nvcv::READ))
    {
        nvcv::GpuMat ocvGPU{lkData->size.h, lkData->size.w,
                          lkData->plane(0).buffer,
                          lkData->plane(0).rowStride};
        // ...
    }

    // alternative?
    if(nvcv::LockImageData lkData = img.lockCudaMem(nvcv::READ))
    {
        // If we know image holds managed memory, we can do this:
        nvcv::Mat ocvCPU{lkData->size.h, lkData->size.w,
                       lkData->plane(0).buffer,
                       lkData->plane(0).rowStride};
        // ...
    }

    class ProcessImageVisitor
        : public nvcv::IVisitorImageData
    {
    public:
        ProcessImageVisitor(cudaStream_t stream)
            : m_stream(stream) {}

        bool visit(ImageDataCudaMem &data) override
        {
            // pitch-linear processing
            nvcv::GpuMat ocvGPU{data->size.h, data->size.w,
                              data->plane(0).buffer,
                              data->plane(0).rowStride};
            // process image in m_stream
            return true;
        }

        bool visit(ImageDataCudaArray &data) override
        {
            // block-linear processing
            cudaArray_t carray = data->plane(0);
            // process image in m_stream
            return true;
        }

    }

    // Works for both pitch-linear and block-linear
    img.lock(nvcv::READ).accept(ProcessImageVisitor(stream));
}

TEST(Image, smoke_wrap_opencv_read)
{
    ;

    // create opencv mat and wrap it
    nvcv::Mat mat(256,512,CV_8UC3);
    auto img = nvcv::ImageWrapData(mat, nvcv::FMT_BGR8);

    // ... op write to img ...

    // write result to disk
    {
        nvcv::LockedImage lk = img.lock(nvcv::LOCK_READ);
        nvcv::imwrite("output.png",mat);
    }
}

TEST(Image, smoke_wrap_opencv_write)
{
    ;

    // create opencv mat and wrap it
    nvcv::Mat mat(256,512,CV_8UC3);
    auto img = nvcv::ImageWrapData(mat, nvcv::FMT_BGR8);

    {
        nvcv::LockedImage lk = img.lock(nvcv::LOCK_WRITE);
        // write to mat
    }

    // ... op read from img ...
}

TEST(Image, smoke_img_opencv_read)
{
    ;

    nvcv::Image img({512, 256}, nvcv::FMT_BGR8, nvcv::HOST);

    // ... op write to img ...

    // write result to disk
    {
        nvcv::LockedImage lk = img.lockOpenCV(nvcv::LOCK_READ); // - dev->host copy
        nvcv::imwrite("output.png",*lk);
    }
}

TEST(Image, smoke_memcpy_opencv_read)
{
    ;

    auto img = nvcv::ImageWrapData({512,256}, nvcv::FMT_BGR8);

    // ... op write to img ...

    // write result to disk
    {
        nvcv::LockedImage lk = img.lockDevice(nvcv::LOCK_READ);

        nvcv::Mat mat(256,512,CV_8UC3);
        memcpy(mat, *lk);

        nvcv::imwrite("output.png",mat);
    }
}


TEST(Image, smoke_memcpy_opencv_write)
{
    ;

    auto img = nvcv::ImageWrapData({512,256}, nvcv::FMT_BGR8);

    // read result from disk
    {
        nvcv::LockedImage lk = img.lockDevice(nvcv::LOCK_READ);

        nvcv::Mat mat = nvcv::imread("input.png");

        memcpy(*lk, mat);
    }

    // ... op write to img ...
}

#endif
