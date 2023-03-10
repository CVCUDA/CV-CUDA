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

#include <nvcv/detail/Callback.hpp>

#include <atomic>
#include <functional>
#include <string>
#include <type_traits>

namespace {

float foo(int x, float y)
{
    return x + y;
}

float bar(int x, float y)
{
    return x - y;
}

struct EmptyCallable
{
    const char *operator()(const char *x)
    {
        return x + 1;
    };
};

struct SimpleCallable
{
    int offset = 2;

    const char *operator()(const char *x)
    {
        return x + offset;
    };
};

template<typename T>
struct InstanceCounter
{
    InstanceCounter()
    {
        num_instances++;
    }

    InstanceCounter(const InstanceCounter &)
    {
        num_instances++;
    }

    InstanceCounter(InstanceCounter &&)
    {
        num_instances++;
    }

    ~InstanceCounter()
    {
        num_instances--;
    }

    static std::atomic_int num_instances;
};

template<typename T>
std::atomic_int InstanceCounter<T>::num_instances{0};

struct ComplexCallable : InstanceCounter<ComplexCallable>
{
    std::string prefix;

    std::string operator()(const char *x)
    {
        return prefix + x;
    }
};

} // namespace

using nvcv::Callback;

static_assert(std::is_constructible_v<Callback<int(int)>, int (*)(int)>);
static_assert(std::is_constructible_v<Callback<int(float)>, int (*)(int)>);
static_assert(!std::is_constructible_v<Callback<int(float &)>, int (*)(int &)>);

static_assert(!std::is_constructible_v<Callback<int(int)>, SimpleCallable>);
static_assert(std::is_constructible_v<Callback<const char *(const char *)>, EmptyCallable>);
static_assert(std::is_constructible_v<Callback<const char *(const char *)>, SimpleCallable>);
static_assert(std::is_constructible_v<Callback<std::string(const char *)>, ComplexCallable>);

// return type conversion
static_assert(std::is_constructible_v<Callback<std::string(const char *)>, SimpleCallable>);

static_assert(!Callback<int(int)>::requiresCleanup<int (*)(int)>());
static_assert(!Callback<const char *(const char *)>::requiresCleanup<EmptyCallable>());
static_assert(!Callback<const char *(const char *)>::requiresCleanup<SimpleCallable>());
static_assert(Callback<const std::string(const char *)>::requiresCleanup<ComplexCallable>());

TEST(CallbackTest, FromFunction)
{
    // test construction
    Callback<float(int, float)> cb = foo;
    EXPECT_EQ(cb(42, 1.5f), 43.5f);
    EXPECT_FALSE(cb.requiresCleanup());
    EXPECT_EQ(cb.targetHandle(), reinterpret_cast<void *>(&foo));
    EXPECT_EQ(cb.cleanupFunc(), nullptr);

    // test assignment
    cb = bar;

    EXPECT_EQ(cb(42, 1.5f), 40.5f);
    EXPECT_FALSE(cb.requiresCleanup());
    EXPECT_EQ(cb.targetHandle(), reinterpret_cast<void *>(&bar));
    EXPECT_EQ(cb.cleanupFunc(), nullptr);
}

TEST(CallbackTest, FromEmptyCallable)
{
    // test construction
    Callback<const char *(const char *)> cb    = EmptyCallable();
    const char                          *input = "test";
    EXPECT_EQ(cb(input), input + 1);
    EXPECT_FALSE(cb.requiresCleanup());
    EXPECT_EQ(cb.targetHandle(), nullptr) << "Empty callable shouldn't require any context.";
    EXPECT_EQ(cb.cleanupFunc(), nullptr) << "Empty callable shouldn't require cleanup";
}

TEST(CallbackTest, FromSimpleCallable)
{
    // test construction
    SimpleCallable sc;
    sc.offset = 3;

    Callback<const char *(const char *)> cb    = sc;
    const char                          *input = "test";

    EXPECT_EQ(cb(input), input + 3);
    EXPECT_FALSE(cb.requiresCleanup());
    EXPECT_NE(cb.targetHandle(), nullptr);
    EXPECT_EQ(cb.cleanupFunc(), nullptr);
}

TEST(CallbackTest, ComplexObjectLifecycle)
{
    // test construction
    {
        ComplexCallable cc;
        cc.prefix = "pre";

        EXPECT_EQ(cc.num_instances, 1);
        Callback<std::string(const char *)> cb = cc;
        EXPECT_EQ(cc.num_instances, 2);
        const char *input = "text";
        EXPECT_EQ(cb(input), "pretext");
        EXPECT_TRUE(cb.requiresCleanup());
        EXPECT_NE(cb.targetHandle(), nullptr);
        EXPECT_NE(cb.cleanupFunc(), nullptr);

        void *old_target = cb.targetHandle();

        Callback<std::string(const char *)> cb2 = std::move(cb);
        // moved out - check it
        EXPECT_FALSE(cb.requiresCleanup());
        EXPECT_EQ(cb.targetFunc(), nullptr);
        EXPECT_EQ(cb.targetHandle(), nullptr);
        EXPECT_EQ(cb.cleanupFunc(), nullptr);
        EXPECT_EQ(cc.num_instances, 2);

        EXPECT_TRUE(cb2.requiresCleanup());
        EXPECT_EQ(cb2.targetHandle(), old_target);
        EXPECT_EQ(cb2("amble"), "preamble");
        cb2.reset();
        EXPECT_EQ(cb2.targetFunc(), nullptr);
        EXPECT_EQ(cb2.targetHandle(), nullptr);
        EXPECT_EQ(cb2.cleanupFunc(), nullptr);
        EXPECT_EQ(cc.num_instances, 1);

        cb2 = cc;
        cb2 = cc; // force overwrite
        EXPECT_EQ(cc.num_instances, 2);

        EXPECT_EQ(cb2("fix"), "prefix");
    }
    EXPECT_EQ(ComplexCallable::num_instances, 0);
}

TEST(CallbackTest, FromStdFunction)
{
    // compatible, but not exact - the return type differs
    Callback<int(int, float)> cb = std::function<float(int, float)>(&foo);
    EXPECT_FALSE(cb.requiresCleanup());
    EXPECT_EQ(cb(1, 2), 3);

    // yet another conversion
    cb = std::function<int(int, int)>([](int x, int y) { return x * y; });
    // The Callback's constructor cannot see a lambda directly (it's wrapped in std::function)
    // and it cannot resolve it to a simple function pointer - therefore, it has to store
    // the enclosing `std::function` object and that requires cleanup.
    EXPECT_TRUE(cb.requiresCleanup());
    EXPECT_EQ(cb(2, 3), 6);
}

TEST(CallbackTest, FromLambda)
{
    Callback<float(float, float)> cb = [](int a, int b)
    {
        return a + b;
    };
    // the lambda has no closure and therefore is empty - no cleanup required

    EXPECT_FALSE(cb.requiresCleanup());
    EXPECT_EQ(cb(1, 2), 3);

    int z = 42;

    cb = [=](float x, float y)
    {
        return x * y + z;
    };
    // The lambda capture fits inside one pointer - no cleanup required
    EXPECT_FALSE(cb.requiresCleanup());
    EXPECT_EQ(cb(2, 3), 48);

    int zzz[7] = {1, 2, 3, 4, 5, 6, 7};

    cb = [=](int x, int y)
    {
        return x * y + zzz[6];
    };
    // The lambda capture is too big - cleanup required
    EXPECT_TRUE(cb.requiresCleanup());
    EXPECT_EQ(cb(2, 3), 13);
}

///////////////////////////////////////////////////////////////////////////////
// In this section, we're battle-testing the Callback with a C++ callback that
// uses a wrapper class around a C structure. The callback marshals the
// C++ object as a corresponding C struct. When the callback is invoked, the C
// structure is wrapped into a temporary object.

namespace {

typedef struct SomeCStruct
{
    void *mem;
    int   len;
} CStruct;

struct StructWrapper
{
    StructWrapper() = default;

    explicit StructWrapper(const CStruct &s)
        : m_data(s)
    {
    }

    CStruct m_data;
};

using CCleanup   = int(void *ctx, CStruct *data);
using CppCleanup = void(const StructWrapper &wrapper);

struct TranslateCleanup
{
    template<typename CppCallable>
    int operator()(CppCallable &&fun, CStruct *data)
    {
        try
        {
            fun(StructWrapper(*data));
            return 0;
        }
        catch (...)
        {
            return -1;
        }
    }
};

struct CContainer
{
    CStruct data;

    struct
    {
        CCleanup *func;
        void     *ctx;
    } cleanup;
};

CContainer *CreateCContainer(const StructWrapper &wrapper, Callback<CppCleanup, CCleanup, TranslateCleanup> &&cb)
{
    auto c  = std::make_unique<CContainer>(CContainer{});
    c->data = wrapper.m_data;
    if (cb.requiresCleanup())
        throw std::invalid_argument("The cleanup function must not require extra cleanup.");
    c->cleanup.func = cb.targetFunc();
    c->cleanup.ctx  = cb.targetHandle();
    return c.release();
}

void DestroyCContainer(CContainer *c)
{
    if (!c)
        return;
    if (c->cleanup.func)
        c->cleanup.func(c->cleanup.ctx, &c->data);
    delete c;
}

} // namespace

TEST(CallbackTest, TestCCleanup)
{
    static const char payload[] = "test";

    StructWrapper wrapper;
    wrapper.m_data.mem = const_cast<char *>(payload);
    wrapper.m_data.len = sizeof(payload);

    bool cleanup_called = false;

    auto good_cleanup = [&](const StructWrapper &w)
    {
        EXPECT_EQ(w.m_data.mem, payload);
        EXPECT_EQ(w.m_data.len, sizeof(payload));
        cleanup_called = true;
    };

    CContainer *cc = CreateCContainer(wrapper, good_cleanup);

    DestroyCContainer(cc);

    EXPECT_TRUE(cleanup_called);

    std::function<CppCleanup> bad_cleanup = good_cleanup;
    EXPECT_THROW(CreateCContainer(wrapper, bad_cleanup), std::invalid_argument);
}

//////////////////////////////////////////////////////////////////////////////
// CleanupCallback - testing self-cleanup capabilities.

TEST(CleanupCallbackTest, TestSelfCleanup)
{
    ComplexCallable cc;
    cc.prefix = "pre";

    EXPECT_EQ(cc.num_instances, 1);
    nvcv::CleanupCallback<std::string(const char *)> cb = cc;
    EXPECT_EQ(cc.num_instances, 2);
    const char *input = "text";
    EXPECT_TRUE(cb.requiresCleanup());
    EXPECT_NE(cb.targetHandle(), nullptr);
    EXPECT_NE(cb.cleanupFunc(), nullptr);
    EXPECT_EQ(cb(input), "pretext");
    EXPECT_EQ(cc.num_instances, 1) << "Self-cleanup did not occur";
    EXPECT_THROW(cb(input), std::bad_function_call);
    EXPECT_EQ(cb.targetHandle(), nullptr) << "Target handle wasn't reset";
    EXPECT_EQ(cb.cleanupFunc(), nullptr) << "Cleanup function wasn't reset";
    EXPECT_EQ(cb.targetFunc(), nullptr) << "Target function wasn't reset";
}

TEST(CleanupCallbackTest, LambdaWithDestructorCalledFromC)
{
    {
        ComplexCallable cc;
        cc.prefix = "pre";

        auto lambda = [=](const std::string &arg) mutable
        {
            return cc(arg.c_str());
        };

        EXPECT_EQ(cc.num_instances, 2);
        nvcv::CleanupCallback<std::string(const char *)> cb = lambda;
        EXPECT_EQ(cc.num_instances, 3);
        const char *input = "text";
        EXPECT_TRUE(cb.requiresCleanup());
        EXPECT_NE(cb.targetHandle(), nullptr);
        EXPECT_NE(cb.cleanupFunc(), nullptr);
        auto [func, target, cleanup] = cb.release();
        EXPECT_EQ(cb.targetFunc(), nullptr);
        EXPECT_EQ(cb.targetHandle(), nullptr);
        EXPECT_EQ(cb.cleanupFunc(), nullptr);
        EXPECT_EQ(func(target, input), "pretext");
        EXPECT_EQ(cc.num_instances, 2);
    }
    EXPECT_EQ(ComplexCallable::num_instances, 0);
}

TEST(CleanupCallbackTest, LambdaWithDestructorNotCalled)
{
    ComplexCallable cc;
    cc.prefix = "unused";

    EXPECT_EQ(ComplexCallable::num_instances, 1);

    auto lambda = [=](const std::string &arg) mutable -> std::string
    {
        return cc.prefix + arg;
    };

    EXPECT_EQ(ComplexCallable::num_instances, 2);

    {
        nvcv::CleanupCallback<std::string(const char *)> cb = std::move(lambda);
        EXPECT_EQ(ComplexCallable::num_instances, 3);
    }
    EXPECT_EQ(ComplexCallable::num_instances, 2) << "Callback context leaked";
}

TEST(CleanupCallbackTest, ThrowInC)
{
    {
        ComplexCallable cc;
        cc.prefix = "defeat short string optimization";

        auto lambda = [=](const std::string &arg) mutable -> std::string
        {
            throw std::runtime_error(cc.prefix);
        };

        EXPECT_EQ(cc.num_instances, 2);
        nvcv::CleanupCallback<std::string(const char *)> cb = lambda;
        EXPECT_EQ(cc.num_instances, 3);
        const char *input = "text";
        EXPECT_TRUE(cb.requiresCleanup());
        EXPECT_NE(cb.targetHandle(), nullptr);
        EXPECT_NE(cb.cleanupFunc(), nullptr);
        auto [func, target, cleanup] = cb.release();
        EXPECT_EQ(cb.targetFunc(), nullptr);
        EXPECT_EQ(cb.targetHandle(), nullptr);
        EXPECT_EQ(cb.cleanupFunc(), nullptr);
        EXPECT_THROW(
            {
                try
                {
                    func(target, input);
                }
                catch (const std::runtime_error &e)
                {
                    EXPECT_EQ(e.what(), cc.prefix);
                    throw;
                }
            },
            std::runtime_error);

        EXPECT_EQ(cc.num_instances, 2) << "Cleanup was unsuccessful";
    }
    EXPECT_EQ(ComplexCallable::num_instances, 0) << "Callback context leaked";
}
