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

#include <nvcv_types/priv/Exception.hpp>
#include <nvcv_types/priv/HandleManager.hpp>
#include <nvcv_types/priv/HandleManagerImpl.hpp>

#include <unordered_set>

namespace priv = nvcv::priv;

constexpr int FORCE_FAILURE = 0xDEADBEEF;

namespace {
class alignas(priv::kResourceAlignment) IObject
{
public:
    virtual int value() const = 0;
};

class Object : public IObject
{
public:
    explicit Object(int val)
        : m_value(val)
    {
        if (val == FORCE_FAILURE)
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
    using type = CompatibleStorage<Object>;
};
} // namespace nvcv::priv

TEST(HandleManager, smoke_handle_generation_wraps_around)
{
    priv::HandleManager<IObject> mgr("Object");

    mgr.setFixedSize(1);

    std::unordered_set<void *> usedHandles;

    void   *h;
    Object *obj;
    std::tie(h, obj) = mgr.create<Object>(0);
    ASSERT_EQ(0, obj->value());
    ASSERT_EQ(obj, mgr.validate(h));
    usedHandles.insert(h);

    void *origh = h;

    // Maximum of 16 generations
    for (int i = 1; i < 16; ++i)
    {
        IObject *obj = mgr.validate(h);
        ASSERT_EQ(i - 1, obj->value());

        mgr.decRef(h);
        void *newh = mgr.create<Object>(i).first;
        ASSERT_FALSE(usedHandles.contains(newh)) << "Handle generation must be different";
        usedHandles.insert(newh);

        IObject *newobj = mgr.validate(newh);
        ASSERT_EQ(obj, newobj);
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

    void *h = mgr.create<Object>(0).first;
    ASSERT_EQ(0, mgr.decRef(h));
    ASSERT_THROW(mgr.decRef(h), nvcv::priv::Exception);
}

TEST(HandleManager, smoke_ref_unref)
{
    priv::HandleManager<IObject> mgr("Object");

    void *h = mgr.create<Object>(0).first;
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

    void *h = mgr.create<Object>(0).first;
    EXPECT_THROW(mgr.decRef((void *)0x666), nvcv::priv::Exception);
    EXPECT_EQ(0, mgr.decRef(h));
}

TEST(HandleManager, smoke_validate_already_destroyed)
{
    priv::HandleManager<IObject> mgr("Object");

    void *h = mgr.create<Object>(0).first;
    ASSERT_NE(nullptr, mgr.validate(h));

    ASSERT_EQ(0, mgr.decRef(h));
    ASSERT_EQ(nullptr, mgr.validate(h));
}

TEST(HandleManager, smoke_validate_invalid)
{
    priv::HandleManager<IObject> mgr("Object");

    void *h = mgr.create<Object>(0).first;
    ASSERT_NE(nullptr, mgr.validate(h)); // just to have something being managed already

    ASSERT_EQ(nullptr, mgr.validate((void *)0x666));

    ASSERT_EQ(0, mgr.decRef(h));
}

TEST(HandleManager, smoke_handle_count_overflow)
{
    priv::HandleManager<IObject> mgr("Object");
    mgr.setFixedSize(1);

    void *h = nullptr;
    ASSERT_NO_THROW(h = mgr.create<Object>(0).first);
    NVCV_ASSERT_STATUS(NVCV_ERROR_OUT_OF_MEMORY, mgr.create<Object>(1));

    mgr.decRef(h);
}

TEST(HandleManager, smoke_no_handle_leak_if_object_creation_throws)
{
    priv::HandleManager<IObject> mgr("Object");
    mgr.setFixedSize(1);

    ASSERT_THROW(mgr.create<Object>(FORCE_FAILURE), std::runtime_error);

    void *h = nullptr;
    ASSERT_NO_THROW(h = mgr.create<Object>(1).first);
    mgr.decRef(h);
}
