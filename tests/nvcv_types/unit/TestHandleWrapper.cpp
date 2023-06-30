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

#include <nvcv/HandleWrapper.hpp>
#include <nvcv_types/priv/Exception.hpp>
#include <nvcv_types/priv/HandleManager.hpp>
#include <nvcv_types/priv/HandleManagerImpl.hpp>

namespace {

class alignas(nvcv::priv::kResourceAlignment) IObject
{
public:
    virtual int value() const = 0;
};

constexpr int kThrowAtConstruction = 0xBADF00D;

class DummyResource : public IObject
{
public:
    explicit DummyResource(int val)
        : m_value(val)
    {
        if (val == kThrowAtConstruction)
        {
            throw std::runtime_error("Forced failure");
        }
    }

    virtual int value() const override
    {
        return m_value;
    }

private:
    int m_value;
};

} // namespace

namespace nvcv::priv {
template<>
struct ResourceStorage<IObject>
{
    using type = CompatibleStorage<DummyResource>;
};
} // namespace nvcv::priv

namespace {

typedef struct NVCVDummy *NVCVDummyHandle;

auto &ManagerInst()
{
    static nvcv::priv::HandleManager<IObject> g_manager("Dummy");
    return g_manager;
}

NVCVStatus nvcvDummyCreate(int k, NVCVDummyHandle *out)
{
    return nvcv::priv::ProtectCall([&]() { *out = (NVCVDummyHandle)ManagerInst().create<DummyResource>(k).first; });
}

NVCVStatus nvcvDummyIncRef(NVCVDummyHandle handle, int *ref)
{
    return nvcv::priv::ProtectCall(
        [&]()
        {
            int r = ManagerInst().incRef(handle);
            if (ref)
                *ref = r;
        });
}

NVCVStatus nvcvDummyDecRef(NVCVDummyHandle handle, int *ref)
{
    return nvcv::priv::ProtectCall(
        [&]()
        {
            int r = ManagerInst().decRef(handle);
            if (ref)
                *ref = r;
        });
}

NVCVStatus nvcvDummyDestroy(NVCVDummyHandle handle)
{
    return nvcvDummyDecRef(handle, nullptr);
}

NVCVStatus nvcvDummyRefCount(NVCVDummyHandle handle, int *ref)
{
    return nvcv::ProtectCall(
        [&]()
        {
            int r = ManagerInst().refCount(handle);
            *ref  = r;
        });
}

} // namespace

namespace nvcv {
// Generate the handle ops
NVCV_IMPL_SHARED_HANDLE(Dummy);
NVCV_IMPL_DESTROY(Dummy);
} // namespace nvcv

TEST(HandleWrapperTest, TestHandleOps)
{
    NVCVDummyHandle handle;
    ASSERT_EQ(NVCV_SUCCESS, nvcvDummyCreate(0, &handle));
    ASSERT_EQ(ManagerInst().refCount(handle), 1);

    nvcv::detail::SharedHandleOps<NVCVDummyHandle> shared_ops;
    static_assert(nvcv::detail::SharedHandleOps<NVCVDummyHandle>::IsNull(nullptr));
    static_assert(nvcv::detail::SharedHandleOps<NVCVDummyHandle>::Null() == nullptr);

    EXPECT_NE(handle, shared_ops.Null());
    EXPECT_FALSE(shared_ops.IsNull(handle));

    EXPECT_EQ(shared_ops.RefCount(handle), 1);
    EXPECT_EQ(shared_ops.IncRef(handle), 2);
    EXPECT_EQ(shared_ops.IncRef(handle), 3);
    EXPECT_EQ(shared_ops.RefCount(handle), 3);
    EXPECT_EQ(shared_ops.DecRef(handle), 2);
    nvcv::detail::UniqueHandleOps<NVCVDummyHandle> unique_ops;
    unique_ops.Destroy(handle);
    EXPECT_EQ(shared_ops.RefCount(handle), 1); // destroy should call DecRef (that's how this Dummy works)

    EXPECT_EQ(shared_ops.DecRef(handle), 0);           // object destroyed
    EXPECT_NE(NVCV_SUCCESS, nvcvDummyDestroy(handle)); // already destroyed
}

TEST(UniqueHandleTest, CreateDestroy)
{
    NVCVDummyHandle handle;
    ASSERT_EQ(NVCV_SUCCESS, nvcvDummyCreate(0, &handle));

    auto backup = handle;
    {
        nvcv::UniqueHandle<NVCVDummyHandle> uh(std::move(handle));
        EXPECT_EQ(handle, nullptr);
        EXPECT_EQ(backup, uh.get());
        EXPECT_NE(nullptr, ManagerInst().validate(backup));
    }
    EXPECT_EQ(nullptr, ManagerInst().validate(backup));
}

TEST(UniqueHandleTest, ResetRelease)
{
    NVCVDummyHandle handle;
    ASSERT_EQ(NVCV_SUCCESS, nvcvDummyCreate(0, &handle));

    auto                                backup = handle;
    nvcv::UniqueHandle<NVCVDummyHandle> uh1(std::move(handle));
    EXPECT_EQ(handle, nullptr);
    EXPECT_EQ(backup, uh1.get());

    nvcv::UniqueHandle<NVCVDummyHandle> uh2(uh1.release());
    EXPECT_EQ(uh1.get(), nullptr);
    EXPECT_EQ(uh2.get(), backup);
    EXPECT_NE(nullptr, ManagerInst().validate(backup));
    uh2.reset();
    EXPECT_EQ(nullptr, ManagerInst().validate(backup));
}

TEST(UniqueHandleTest, Overwrite)
{
    NVCVDummyHandle h1, h2;
    ASSERT_EQ(NVCV_SUCCESS, nvcvDummyCreate(0, &h1));
    ASSERT_EQ(NVCV_SUCCESS, nvcvDummyCreate(0, &h2));

    auto                                backup1 = h1;
    auto                                backup2 = h2;
    nvcv::UniqueHandle<NVCVDummyHandle> uh1(std::move(h1));
    nvcv::UniqueHandle<NVCVDummyHandle> uh2(std::move(h2));
    ASSERT_EQ(uh1.get(), backup1);
    ASSERT_EQ(uh2.get(), backup2);

    EXPECT_NE(nullptr, ManagerInst().validate(backup1));
    EXPECT_NE(nullptr, ManagerInst().validate(backup2));

    uh1 = std::move(uh2);
    EXPECT_EQ(uh2.get(), nullptr);
    EXPECT_EQ(uh1.get(), backup2);

    EXPECT_NE(nullptr, ManagerInst().validate(backup2));
    EXPECT_EQ(nullptr, ManagerInst().validate(backup1));
}

TEST(SharedHandleTest, CreateDestroy)
{
    NVCVDummyHandle handle;
    ASSERT_EQ(NVCV_SUCCESS, nvcvDummyCreate(0, &handle));

    auto backup = handle;
    {
        nvcv::SharedHandle<NVCVDummyHandle> uh(std::move(handle));
        EXPECT_EQ(handle, nullptr);
        EXPECT_EQ(backup, uh.get());
        EXPECT_EQ(1, ManagerInst().refCount(backup));
    }
    EXPECT_EQ(nullptr, ManagerInst().validate(backup));
}

TEST(SharedHandleTest, ResetRelease)
{
    NVCVDummyHandle handle;
    ASSERT_EQ(NVCV_SUCCESS, nvcvDummyCreate(0, &handle));

    auto                                backup = handle;
    nvcv::SharedHandle<NVCVDummyHandle> uh1(std::move(handle));
    EXPECT_EQ(handle, nullptr);
    EXPECT_EQ(backup, uh1.get());

    nvcv::SharedHandle<NVCVDummyHandle> uh2(uh1.release());
    EXPECT_EQ(uh1.get(), nullptr);
    EXPECT_EQ(uh2.get(), backup);
    EXPECT_EQ(1, ManagerInst().refCount(backup));
    uh2.reset();
    EXPECT_EQ(nullptr, ManagerInst().validate(backup));
}

TEST(SharedHandleTest, CopyMove)
{
    NVCVDummyHandle h1, h2;
    ASSERT_EQ(NVCV_SUCCESS, nvcvDummyCreate(0, &h1));
    ASSERT_EQ(NVCV_SUCCESS, nvcvDummyCreate(0, &h2));

    auto                                backup1 = h1;
    auto                                backup2 = h2;
    nvcv::SharedHandle<NVCVDummyHandle> uh1(std::move(h1));
    nvcv::SharedHandle<NVCVDummyHandle> uh2(std::move(h2));

    EXPECT_EQ(uh1.get(), backup1);
    EXPECT_EQ(uh1.refCount(), 1);

    EXPECT_EQ(uh2.get(), backup2);
    EXPECT_EQ(uh2.refCount(), 1);

    nvcv::SharedHandle<NVCVDummyHandle> uh3 = uh1;

    EXPECT_EQ(uh3.get(), backup1);
    EXPECT_EQ(uh3.refCount(), 2);
    EXPECT_EQ(uh1.refCount(), 2);

    EXPECT_NE(uh1, uh2); // h1, h2
    EXPECT_EQ(uh1, uh3); // h1, h1 <- these two should now match
    EXPECT_NE(uh2, uh3); // h2, h1

    uh3 = std::move(uh1);

    EXPECT_NE(uh1, uh2); // null, h2
    EXPECT_NE(uh1, uh3); // null, h1
    EXPECT_NE(uh2, uh3); // h2, h1

    EXPECT_FALSE(uh1);
    EXPECT_EQ(uh1.refCount(), 0);

    EXPECT_EQ(uh3.refCount(), 1);

    uh1.reset();
    uh2.reset();
    uh3.reset();

    EXPECT_EQ(uh1, uh2); // all null - all equal
    EXPECT_EQ(uh1, uh3);
    EXPECT_EQ(uh2, uh3);

    EXPECT_EQ(nullptr, ManagerInst().validate(backup1));
    EXPECT_EQ(nullptr, ManagerInst().validate(backup2));
}
