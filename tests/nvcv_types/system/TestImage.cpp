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

#include "Definitions.hpp"

#include <nvcv/Casts.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/alloc/CustomAllocator.hpp>
#include <nvcv/alloc/CustomResourceAllocator.hpp>

#include <nvcv/Fwd.hpp>

TEST(Image, wip_create)
{
    nvcv::Image img({163, 117}, nvcv::FMT_RGBA8);

    EXPECT_EQ(nvcv::Size2D(163, 117), img.size());
    EXPECT_EQ(nvcv::FMT_RGBA8, img.format());
    ASSERT_NE(nullptr, img.handle());

    NVCVTypeImage type;
    ASSERT_EQ(NVCV_SUCCESS, nvcvImageGetType(img.handle(), &type));
    EXPECT_EQ(NVCV_TYPE_IMAGE, type);

    const nvcv::IImageData *data = img.exportData();
    ASSERT_NE(nullptr, data);

    auto *devdata = dynamic_cast<const nvcv::IImageDataStridedCuda *>(data);
    ASSERT_NE(nullptr, devdata);

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

TEST(Image, wip_cast)
{
    nvcv::Image img({163, 117}, nvcv::FMT_RGBA8);

    EXPECT_EQ(&img, nvcv::StaticCast<nvcv::Image *>(img.handle()));
    EXPECT_EQ(&img, &nvcv::StaticCast<nvcv::Image>(img.handle()));

    EXPECT_EQ(&img, nvcv::StaticCast<nvcv::IImage *>(img.handle()));
    EXPECT_EQ(&img, &nvcv::StaticCast<nvcv::IImage>(img.handle()));

    EXPECT_EQ(&img, nvcv::DynamicCast<nvcv::Image *>(img.handle()));
    EXPECT_EQ(&img, &nvcv::DynamicCast<nvcv::Image>(img.handle()));

    EXPECT_EQ(&img, nvcv::DynamicCast<nvcv::IImage *>(img.handle()));
    EXPECT_EQ(&img, &nvcv::DynamicCast<nvcv::IImage>(img.handle()));

    EXPECT_EQ(nullptr, nvcv::DynamicCast<nvcv::ImageWrapData *>(img.handle()));

    EXPECT_EQ(nullptr, nvcv::StaticCast<nvcv::IImage *>(nullptr));
    EXPECT_EQ(nullptr, nvcv::DynamicCast<nvcv::IImage *>(nullptr));
    EXPECT_THROW(nvcv::DynamicCast<nvcv::IImage>(nullptr), std::bad_cast);

    // Now when we create the object via C API

    NVCVImageHandle       handle;
    NVCVImageRequirements reqs;
    ASSERT_EQ(NVCV_SUCCESS, nvcvImageCalcRequirements(163, 117, NVCV_IMAGE_FORMAT_RGBA8, 0, 0, &reqs));
    ASSERT_EQ(NVCV_SUCCESS, nvcvImageConstruct(&reqs, nullptr, &handle));

    // Size of the internal buffer used to store the WrapHandle object
    // we might have to create for containers allocated via C API.
    // This value must never decrease, or else it'll break ABI compatibility.
    uintptr_t max = 512;

    EXPECT_GE(max, sizeof(nvcv::detail::WrapHandle<nvcv::IImage>)) << "Must be big enough for the WrapHandle";

    void *cxxPtr = &max;
    ASSERT_EQ(NVCV_SUCCESS, nvcvImageGetUserPointer((NVCVImageHandle)(((uintptr_t)handle) | 1), &cxxPtr));
    ASSERT_NE(&max, cxxPtr) << "Pointer must have been changed";

    // Buffer too big, bail.
    max    = 513;
    cxxPtr = &max;
    ASSERT_EQ(NVCV_ERROR_INTERNAL, nvcvImageGetUserPointer((NVCVImageHandle)(((uintptr_t)handle) | 1), &cxxPtr))
        << "Required WrapHandle buffer storage should have been too big";

    nvcv::IImage *pimg = nvcv::StaticCast<nvcv::IImage *>(handle);
    ASSERT_NE(nullptr, pimg);
    EXPECT_EQ(handle, pimg->handle());
    EXPECT_EQ(163, pimg->size().w);
    EXPECT_EQ(117, pimg->size().h);
    EXPECT_EQ(nvcv::FMT_RGBA8, pimg->format());

    EXPECT_EQ(pimg, nvcv::DynamicCast<nvcv::IImage *>(handle));
    EXPECT_EQ(nullptr, nvcv::DynamicCast<nvcv::Image *>(handle));

    nvcvImageDestroy(handle);
}

TEST(Image, wip_user_pointer)
{
    nvcv::Image img({163, 117}, nvcv::FMT_RGBA8);
    EXPECT_EQ(nullptr, img.userPointer());

    void *cxxPtr;
    ASSERT_EQ(NVCV_SUCCESS, nvcvImageGetUserPointer((NVCVImageHandle)(((uintptr_t)img.handle()) | 1), &cxxPtr));
    ASSERT_EQ(&img, cxxPtr) << "cxx object pointer must always be associated with the corresponding handle";

    img.setUserPointer((void *)0x123);
    EXPECT_EQ((void *)0x123, img.userPointer());

    ASSERT_EQ(NVCV_SUCCESS, nvcvImageGetUserPointer((NVCVImageHandle)(((uintptr_t)img.handle()) | 1), &cxxPtr));
    ASSERT_EQ(&img, cxxPtr) << "cxx object pointer must always be associated with the corresponding handle";

    img.setUserPointer(nullptr);
    EXPECT_EQ(nullptr, img.userPointer());

    ASSERT_EQ(NVCV_SUCCESS, nvcvImageGetUserPointer((NVCVImageHandle)(((uintptr_t)img.handle()) | 1), &cxxPtr));
    ASSERT_EQ(&img, cxxPtr) << "cxx object pointer must always be associated with the corresponding handle";
}

TEST(Image, wip_create_managed)
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

    nvcv::Image img({163, 117}, nvcv::FMT_RGBA8, &managedAlloc,
                    nvcv::MemAlignment{}.rowAddr(1).baseAddr(32)); // packed rows
    EXPECT_EQ(32, setBufAlign);

    const nvcv::IImageData *data = img.exportData();
    ASSERT_NE(nullptr, data);

    auto *devdata = dynamic_cast<const nvcv::IImageDataStridedCuda *>(data);
    ASSERT_NE(nullptr, devdata);

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

TEST(ImageWrapData, wip_create)
{
    nvcv::ImageDataStridedCuda::Buffer buf;
    buf.numPlanes           = 1;
    buf.planes[0].width     = 173;
    buf.planes[0].height    = 79;
    buf.planes[0].rowStride = 190;
    buf.planes[0].basePtr   = reinterpret_cast<NVCVByte *>(678);

    nvcv::ImageWrapData img{
        nvcv::ImageDataStridedCuda{nvcv::FMT_U8, buf}
    };

    EXPECT_EQ(nvcv::Size2D(173, 79), img.size());
    EXPECT_EQ(nvcv::FMT_U8, img.format());
    ASSERT_NE(nullptr, img.handle());

    NVCVTypeImage type;
    ASSERT_EQ(NVCV_SUCCESS, nvcvImageGetType(img.handle(), &type));
    EXPECT_EQ(NVCV_TYPE_IMAGE_WRAPDATA, type);

    const nvcv::IImageData *data = img.exportData();
    ASSERT_NE(nullptr, data);

    auto *devdata = dynamic_cast<const nvcv::IImageDataStridedCuda *>(data);
    ASSERT_NE(nullptr, devdata);

    ASSERT_EQ(1, devdata->numPlanes());
    EXPECT_EQ(img.format(), devdata->format());
    EXPECT_EQ(img.size(), devdata->size());
    EXPECT_EQ(img.size().w, devdata->plane(0).width);
    EXPECT_EQ(img.size().h, devdata->plane(0).height);
    EXPECT_LE(190, devdata->plane(0).rowStride);
    EXPECT_EQ(buf.planes[0].basePtr, devdata->plane(0).basePtr);
}

TEST(ImageWrapData, wip_user_pointer)
{
    nvcv::ImageDataStridedCuda::Buffer buf;
    buf.numPlanes           = 1;
    buf.planes[0].width     = 173;
    buf.planes[0].height    = 79;
    buf.planes[0].rowStride = 190;
    buf.planes[0].basePtr   = reinterpret_cast<NVCVByte *>(678);

    nvcv::ImageWrapData img{
        nvcv::ImageDataStridedCuda{nvcv::FMT_U8, buf}
    };

    EXPECT_EQ(nullptr, img.userPointer());

    img.setUserPointer((void *)0x123);
    EXPECT_EQ((void *)0x123, img.userPointer());

    img.setUserPointer(nullptr);
    EXPECT_EQ(nullptr, img.userPointer());
}

TEST(Image, wip_operator)
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

    auto *inData  = dynamic_cast<const nvcv::IImageDataStridedCuda *>(in.exportData());
    auto *outData = dynamic_cast<const nvcv::IImageDataStridedCuda *>(out.exportData());

    if (inData == nullptr || outData == nullptr)
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

TEST(ImageWrapData, wip_cleanup)
{
    nvcv::ImageDataStridedCuda::Buffer buf;
    buf.numPlanes           = 1;
    buf.planes[0].width     = 173;
    buf.planes[0].height    = 79;
    buf.planes[0].rowStride = 190;
    buf.planes[0].basePtr   = reinterpret_cast<NVCVByte *>(678);

    int  cleanupCalled = 0;
    auto cleanup       = [&cleanupCalled](const nvcv::IImageData &data)
    {
        ++cleanupCalled;
    };

    {
        nvcv::ImageWrapData img(nvcv::ImageDataStridedCuda{nvcv::FMT_U8, buf}, cleanup);
        EXPECT_EQ(0, cleanupCalled);
    }
    EXPECT_EQ(1, cleanupCalled) << "Cleanup must have been called when img got destroyed";
}

TEST(ImageWrapData, wip_mem_reqs)
{
    nvcv::Image::Requirements reqs = nvcv::Image::CalcRequirements({512, 256}, nvcv::FMT_NV12);

    nvcv::Image img(reqs);

    EXPECT_EQ(512, img.size().w);
    EXPECT_EQ(256, img.size().h);
    EXPECT_EQ(nvcv::FMT_NV12, img.format());

    const auto *data = dynamic_cast<const nvcv::IImageDataStridedCuda *>(img.exportData());

    ASSERT_NE(nullptr, data);
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
TEST(Image, wip_image_managed_memory)
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

    EXPECT_EQ(nvcv::Size2D{512,256}, img.size());
    EXPECT_EQ(nvcv::FMT_RGBA8, img.format());

    {
        nvcv::LockImageData lkData = img.lock(nvcv::READ);
        if(auto *data = dynamic_cast<const nvcv::IImageDataCudaMem *>(lkData->data()))
        {
            nvcv::GpuMat ocvGPU{data->size.h, data->size.w,
                              data->plane(0).buffer,
                              data->plane(0).rowStride};
            // ...

        }
        else if(auto *data = dynamic_cast<const nvcv::IImageDataCudaArray *>(lkData->data()))
        {
            cudaArray_t carray = data->plane(0);
            // ...
        }
    }

    if(nvcv::LockImageData lkData = img.lock<nvcv::IImageDataCudaMem>(nvcv::READ))
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

        bool visit(IImageDataCudaMem &data) override
        {
            // pitch-linear processing
            nvcv::GpuMat ocvGPU{data->size.h, data->size.w,
                              data->plane(0).buffer,
                              data->plane(0).rowStride};
            // process image in m_stream
            return true;
        }

        bool visit(IImageDataCudaArray &data) override
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

TEST(Image, wip_wrap_opencv_read)
{
    ;

    // create opencv mat and wrap it
    nvcv::Mat mat(256,512,CV_8UC3);
    nvcv::ImageWrapData img(mat, nvcv::FMT_BGR8);

    // ... op write to img ...

    // write result to disk
    {
        nvcv::LockedImage lk = img.lock(nvcv::LOCK_READ);
        nvcv::imwrite("output.png",mat);
    }
}

TEST(Image, wip_wrap_opencv_write)
{
    ;

    // create opencv mat and wrap it
    nvcv::Mat mat(256,512,CV_8UC3);
    nvcv::ImageWrapData img(mat, nvcv::FMT_BGR8);

    {
        nvcv::LockedImage lk = img.lock(nvcv::LOCK_WRITE);
        // write to mat
    }

    // ... op read from img ...
}

TEST(Image, wip_img_opencv_read)
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

TEST(Image, wip_memcpy_opencv_read)
{
    ;

    nvcv::ImageWrapData img({512,256}, nvcv::FMT_BGR8);

    // ... op write to img ...

    // write result to disk
    {
        nvcv::LockedImage lk = img.lockDevice(nvcv::LOCK_READ);

        nvcv::Mat mat(256,512,CV_8UC3);
        memcpy(mat, *lk);

        nvcv::imwrite("output.png",mat);
    }
}


TEST(Image, wip_memcpy_opencv_write)
{
    ;

    nvcv::ImageWrapData img({512,256}, nvcv::FMT_BGR8);

    // read result from disk
    {
        nvcv::LockedImage lk = img.lockDevice(nvcv::LOCK_READ);

        nvcv::Mat mat = nvcv::imread("input.png");

        memcpy(*lk, mat);
    }

    // ... op write to img ...
}

#endif
