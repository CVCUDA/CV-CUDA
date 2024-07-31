/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cvcuda/util/SimpleCache.hpp>

namespace {
struct Payload
{
    int  data      = 0;
    bool destroyed = false;
    bool movedOut  = false;

    explicit Payload(int data)
        : data(data)
    {
    }

    ~Payload()
    {
        data      = -2;
        destroyed = true;
    }

    Payload(Payload &&p)
    {
        *this = std::move(p);
    }

    Payload &operator=(Payload &&p)
    {
        data       = p.data;
        destroyed  = p.destroyed;
        movedOut   = p.movedOut;
        p.data     = -1;
        p.movedOut = true;
        return *this;
    }
};

} // namespace

TEST(SimpleCacheTest, PutGet)
{
    nvcv::util::SimpleCache<Payload> cache;
    EXPECT_FALSE(cache.get().has_value());
    Payload p = cache.getOrCreate([]() { return Payload(42); });
    EXPECT_EQ(p.data, 42);
    EXPECT_FALSE(p.destroyed);
    EXPECT_FALSE(p.movedOut);
    cache.put(std::move(p));
    cache.put(Payload(1234));
    cache.emplace(4321);
    EXPECT_TRUE(p.movedOut);
    EXPECT_FALSE(p.destroyed);

    std::optional<Payload> o = cache.get();
    ASSERT_TRUE(o.has_value());
    EXPECT_EQ(o->data, 4321);
    EXPECT_FALSE(o->destroyed);
    EXPECT_FALSE(o->movedOut);

    o = cache.get();
    ASSERT_TRUE(o.has_value());
    EXPECT_EQ(o->data, 1234);
    EXPECT_FALSE(o->destroyed);
    EXPECT_FALSE(o->movedOut);

    o = cache.get();
    ASSERT_TRUE(o.has_value());
    EXPECT_EQ(o->data, 42);
    EXPECT_FALSE(o->destroyed);
    EXPECT_FALSE(o->movedOut);

    o = cache.get();
    EXPECT_FALSE(o.has_value());
}
