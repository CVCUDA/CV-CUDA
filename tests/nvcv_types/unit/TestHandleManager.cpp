/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <nvcv/src/priv/Exception.hpp>
#include <nvcv/src/priv/HandleManager.hpp>
#include <nvcv/src/priv/HandleManagerImpl.hpp>

#include <cstdlib>
#include <stdexcept>
#include <unordered_set>

namespace priv = nvcv::priv;

constexpr int FORCE_FAILURE = 0xDEADBEEF;

namespace {
struct IObjectHandleRec
{
};

using IObjectHandle = IObjectHandleRec *;

class HandleManagerTestError : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

class alignas(priv::kResourceAlignment) IObject
{
public:
    using HandleType = IObjectHandle;

    virtual ~IObject() = default;

    virtual int value() const = 0;
};

class Object : public IObject
{
public:
    explicit Object(int val, int *destructionCount = nullptr)
        : m_value(val)
        , m_destructionCount(destructionCount)
    {
        if (val == FORCE_FAILURE)
        {
            throw HandleManagerTestError("Forced failure");
        }
    }

    Object(const Object &)            = delete;
    Object(Object &&)                 = delete;
    Object &operator=(const Object &) = delete;
    Object &operator=(Object &&)      = delete;

    ~Object() override
    {
        if (m_destructionCount != nullptr)
        {
            ++*m_destructionCount;
        }
    }

    int value() const override
    {
        return m_value;
    }

private:
    int  m_value;
    int *m_destructionCount;
};

class DependentObject final : public IObject
{
public:
    DependentObject(priv::HandleManager<IObject> &manager, IObjectHandle parent, int &destructionCount)
        : m_manager(manager)
        , m_parent(parent)
        , m_destructionCount(destructionCount)
    {
    }

    DependentObject(const DependentObject &)            = delete;
    DependentObject(DependentObject &&)                 = delete;
    DependentObject &operator=(const DependentObject &) = delete;
    DependentObject &operator=(DependentObject &&)      = delete;

    ~DependentObject() override
    {
        EXPECT_NO_THROW(m_manager.decRef(m_parent));
        ++m_destructionCount;
    }

    int value() const override
    {
        return 0;
    }

private:
    priv::HandleManager<IObject> &m_manager;
    IObjectHandle                 m_parent;
    int                          &m_destructionCount;
};
} // namespace

namespace nvcv::priv {
template<>
struct ResourceStorage<IObject>
{
    using type = CompatibleStorage<Object, DependentObject>;
};
} // namespace nvcv::priv

TEST(HandleManager, smoke_handle_generation_wraps_around)
{
    priv::HandleManager<IObject> mgr("Object");

    mgr.setFixedSize(1);

    std::unordered_set<IObjectHandle> usedHandles;

    IObjectHandle h;
    Object       *obj;
    std::tie(h, obj) = mgr.create<Object>(0);
    ASSERT_EQ(0, obj->value());
    ASSERT_EQ(obj, mgr.validate(h));
    usedHandles.insert(h);

    IObjectHandle origh = h;

    // Maximum of 16 generations
    for (int i = 1; i < 16; ++i)
    {
        IObject *currentObj = mgr.validate(h);
        ASSERT_EQ(i - 1, currentObj->value());

        mgr.decRef(h);
        IObjectHandle newh = mgr.create<Object>(i).first;
        ASSERT_FALSE(usedHandles.contains(newh)) << "Handle generation must be different";
        usedHandles.insert(newh);

        IObject *newobj = mgr.validate(newh);
        ASSERT_EQ(currentObj, newobj);
        ASSERT_EQ(i, newobj->value());

        h = newh;
    }

    mgr.decRef(h);
    std::tie(h, obj) = mgr.create<Object>(16);
    ASSERT_EQ(origh, h) << "Handle generation must wrapped around";
    ASSERT_TRUE(usedHandles.contains(h)) << "Handle must have been reused";
    IObject *iobj = mgr.validate(h);
    ASSERT_EQ(obj, iobj);
    ASSERT_EQ(16, iobj->value());

    mgr.decRef(h);
}

TEST(HandleManager, smoke_destroy_already_destroyed)
{
    priv::HandleManager<IObject> mgr("Object");

    IObjectHandle h = mgr.create<Object>(0).first;
    ASSERT_EQ(0, mgr.decRef(h));
    ASSERT_THROW(mgr.decRef(h), nvcv::priv::Exception);
}

TEST(HandleManager, smoke_ref_unref)
{
    priv::HandleManager<IObject> mgr("Object");

    IObjectHandle h = mgr.create<Object>(0).first;
    ASSERT_EQ(2, mgr.incRef(h));
    ASSERT_EQ(1, mgr.decRef(h));
    ASSERT_EQ(2, mgr.incRef(h));
    ASSERT_EQ(1, mgr.decRef(h));
    ASSERT_EQ(0, mgr.decRef(h));

    EXPECT_THROW(mgr.incRef(h), nvcv::priv::Exception); // invalid handle
    EXPECT_THROW(mgr.decRef(h), nvcv::priv::Exception); // invalid handle
}

TEST(HandleManager, smoke_dec_ref_invalid)
{
    priv::HandleManager<IObject> mgr("Object");

    IObjectHandle    h = mgr.create<Object>(0).first;
    IObjectHandleRec invalidHandle;
    EXPECT_THROW(mgr.decRef(&invalidHandle), nvcv::priv::Exception);
    EXPECT_EQ(0, mgr.decRef(h));
}

TEST(HandleManager, smoke_validate_already_destroyed)
{
    priv::HandleManager<IObject> mgr("Object");

    IObjectHandle h = mgr.create<Object>(0).first;
    ASSERT_NE(nullptr, mgr.validate(h));

    ASSERT_EQ(0, mgr.decRef(h));
    ASSERT_EQ(nullptr, mgr.validate(h));
}

TEST(HandleManager, smoke_validate_invalid)
{
    priv::HandleManager<IObject> mgr("Object");

    IObjectHandle h = mgr.create<Object>(0).first;
    ASSERT_NE(nullptr, mgr.validate(h)); // just to have something being managed already

    IObjectHandleRec invalidHandle;
    ASSERT_EQ(nullptr, mgr.validate(&invalidHandle));

    ASSERT_EQ(0, mgr.decRef(h));
}

TEST(HandleManager, smoke_handle_count_overflow)
{
    priv::HandleManager<IObject> mgr("Object");
    mgr.setFixedSize(1);

    IObjectHandle h = nullptr;
    ASSERT_NO_THROW(h = mgr.create<Object>(0).first);
    NVCV_ASSERT_STATUS(NVCV_ERROR_OUT_OF_MEMORY, mgr.create<Object>(1));

    mgr.decRef(h);
}

TEST(HandleManager, smoke_no_handle_leak_if_object_creation_throws)
{
    priv::HandleManager<IObject> mgr("Object");
    mgr.setFixedSize(1);

    ASSERT_THROW(mgr.create<Object>(FORCE_FAILURE), std::runtime_error);

    IObjectHandle h = nullptr;
    ASSERT_NO_THROW(h = mgr.create<Object>(1).first);
    mgr.decRef(h);
}

TEST(HandleManager, clear_destroys_objects_with_dependent_handles)
{
    ::testing::FLAGS_gtest_death_test_style = "threadsafe";
    ASSERT_EXIT(
        {
            // Let forced cleanup run instead of stopping at the test runner's leak-abort policy.
            if (::setenv("NVCV_LEAK_DETECTION", "warn", 1) != 0)
            {
                std::_Exit(2);
            }

            int parentDestructions    = 0;
            int dependentDestructions = 0;
            {
                priv::HandleManager<IObject> mgr("Object");
                IObjectHandle                parent = mgr.create<Object>(0, &parentDestructions).first;
                if (mgr.incRef(parent) != 2)
                {
                    std::_Exit(3);
                }
                (void)mgr.create<DependentObject>(mgr, parent, dependentDestructions);
            }

            if (parentDestructions != 1)
            {
                std::_Exit(4);
            }
            if (dependentDestructions != 1)
            {
                std::_Exit(5);
            }

            std::_Exit(::testing::Test::HasFailure() ? 6 : 0);
        },
        ::testing::ExitedWithCode(0), "Object leak detection: 2 handles still in use");
}
