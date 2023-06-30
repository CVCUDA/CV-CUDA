/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <nvcv/ImageFormat.hpp>
#include <nvcv/Size.hpp>
#include <nvcv_types/priv/Context.hpp>
#include <nvcv_types/priv/DefaultAllocator.hpp>
#include <nvcv_types/priv/Exception.hpp>
#include <nvcv_types/priv/Image.hpp>
#include <nvcv_types/priv/ImageManager.hpp>
#include <nvcv_types/priv/SharedCoreObj.hpp>

inline NVCVImageHandle CreateImage()
{
    nvcv::Size2D          size{640, 480};
    nvcv::ImageFormat     fmt = nvcv::FMT_RGBA8;
    NVCVImageRequirements reqs;
    nvcv::detail::CheckThrow(nvcvImageCalcRequirements(size.w, size.h, fmt, 256, 16, &reqs));
    auto &alloc = nvcv::priv::GetDefaultAllocator();

    return nvcv::priv::CreateCoreObject<nvcv::priv::Image>(reqs, alloc);
}

TEST(SharedCoreObjTest, Construct)
{
    NVCVImageHandle h = CreateImage();
    EXPECT_EQ(nvcv::priv::CoreObjectRefCount(h), 1);
    {
        using Ptr = nvcv::priv::SharedCoreObj<nvcv::priv::Image>;
        Ptr s1    = nvcv::priv::ToSharedObj<nvcv::priv::Image>(h);
        EXPECT_EQ(nvcv::priv::CoreObjectRefCount(h), 2);
        Ptr s2 = Ptr::FromPointer(nvcv::priv::ToStaticPtr<nvcv::priv::Image>(h), true);
        EXPECT_EQ(nvcv::priv::CoreObjectRefCount(h), 3);
        Ptr s3(s2.release());
        EXPECT_EQ(nvcv::priv::CoreObjectRefCount(h), 3)
            << "Construction from pointer should not increment the reference count.";

        EXPECT_EQ(s1, s3);
        EXPECT_EQ(s1->handle(), h);
        EXPECT_EQ(s3->handle(), h);

        Ptr s4 = Ptr::FromHandle(h, true);
        EXPECT_EQ(s4->handle(), h);
        EXPECT_EQ(nvcv::priv::CoreObjectRefCount(h), 4);

        Ptr s5 = std::move(s4);
        EXPECT_FALSE((bool)s4);
        EXPECT_EQ(s5->handle(), h);
        EXPECT_EQ(nvcv::priv::CoreObjectRefCount(h), 4);

        Ptr s6 = s5;
        EXPECT_EQ(s5->handle(), h);
        EXPECT_EQ(s6->handle(), h);
        EXPECT_EQ(nvcv::priv::CoreObjectRefCount(h), 5);

        Ptr s7 = Ptr::FromHandle(h, false);
        EXPECT_EQ(nvcv::priv::CoreObjectRefCount(h), 5);
        EXPECT_EQ(s7.release(), s6.get());
    }
    EXPECT_EQ(nvcv::priv::CoreObjectRefCount(h), 1);
    EXPECT_EQ(nvcv::priv::CoreObjectDecRef(h), 0);
}

TEST(SharedCoreObjTest, Comparison)
{
    using Ptr = nvcv::priv::SharedCoreObj<nvcv::priv::Image>;

    auto h1 = CreateImage();
    auto h2 = CreateImage();

    Ptr s1 = Ptr::FromHandle(h1, true);
    EXPECT_EQ(nvcv::priv::CoreObjectDecRef(h1), 1);

    Ptr s2 = Ptr::FromHandle(h2, true);
    EXPECT_EQ(nvcv::priv::CoreObjectDecRef(h2), 1);

    Ptr s3 = nullptr;

    EXPECT_TRUE(s1 == s1);
    EXPECT_FALSE(s1 != s1);

    EXPECT_TRUE(s1 != s2);
    EXPECT_FALSE(s1 == s2);

    EXPECT_TRUE(s1 != s3);
    EXPECT_FALSE(s1 == s3);

    EXPECT_TRUE(s3 == s3);
    EXPECT_FALSE(s3 != s3);

    EXPECT_FALSE(s1 == nullptr);
    EXPECT_FALSE(nullptr == s1);
    EXPECT_TRUE(s1 != nullptr);
    EXPECT_TRUE(nullptr != s1);

    EXPECT_TRUE(s3 == nullptr);
    EXPECT_TRUE(nullptr == s3);
    EXPECT_FALSE(s3 != nullptr);
    EXPECT_FALSE(nullptr != s3);

    EXPECT_TRUE((bool)s1);
    EXPECT_TRUE((bool)s2);
    EXPECT_FALSE((bool)s3);

    EXPECT_EQ((s1.get() < s2.get()), (s1 < s2));
    EXPECT_EQ((s2.get() < s1.get()), (s2 < s1));
}

TEST(SharedCoreObjTest, AssignCopyMove)
{
    auto h = CreateImage();
    EXPECT_EQ(nvcv::priv::CoreObjectRefCount(h), 1);
    {
        using Ptr = nvcv::priv::SharedCoreObj<nvcv::priv::Image>;

        Ptr s1 = nvcv::priv::ToSharedObj<nvcv::priv::Image>(h);
        EXPECT_EQ(nvcv::priv::CoreObjectRefCount(h), 2) << "Ref count not raised.";

        Ptr s2 = s1;
        EXPECT_EQ(nvcv::priv::CoreObjectRefCount(h), 3) << "Ref count not raised during copy construction.";

        Ptr s3 = std::move(s2);
        EXPECT_EQ(nvcv::priv::CoreObjectRefCount(h), 3) << "Ref count changed during move construction.";
        EXPECT_EQ(s2, nullptr) << "Not moved out properly";

        s2 = s3;
        EXPECT_EQ(nvcv::priv::CoreObjectRefCount(h), 4) << "Ref count not raised after copy.";
        s3 = nullptr;
        EXPECT_EQ(s3.get(), nullptr) << "Null assignment failed.";
        EXPECT_EQ(nvcv::priv::CoreObjectRefCount(h), 3) << "Ref count not dropped after null assignment.";
        s3 = std::move(s2);
        EXPECT_EQ(nvcv::priv::CoreObjectRefCount(h), 3) << "Ref count changed during move into a null shared pointer.";
        EXPECT_EQ(s3, s1);
        EXPECT_EQ(s2, nullptr) << "Not moved out properly";
        s3 = s2;
        EXPECT_EQ(nvcv::priv::CoreObjectRefCount(h), 2) << "Ref count not dropped after copy-from-null assignment.";
    }
    EXPECT_EQ(nvcv::priv::CoreObjectRefCount(h), 1);
    EXPECT_EQ(nvcv::priv::CoreObjectDecRef(h), 0);
}
