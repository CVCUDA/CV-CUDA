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

#include <dlfcn.h>
#include <pthread.h>
#include <stdlib.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <cassert>
#include <memory>
#include <vector>

// ================================================
//__cxa_thread_atexit_impl

// We need to support glibc-2.17, which doesn't export
// __cxa_thread_atexit_impl, need by libstdc++ >= 8. This
// allows us to support old distros like CentOS 7.

namespace {

struct ThreadDestructorArg;
struct DsoHandle;

using ThreadDestructor     = void(ThreadDestructorArg *);
using PthreadDestructorArg = void;

// Encapsulates the original thread_atexit implementation
// by retrieving the function from libc.so.6, if it exists.
class OrigImpl
{
public:
    OrigImpl()
    {
        m_libc = static_cast<DsoHandle *>(dlopen("libc.so.6", RTLD_LOCAL | RTLD_LAZY));
        if (m_libc)
        {
            m_fn = reinterpret_cast<OrigImplFn>(dlvsym(m_libc, "__cxa_thread_atexit_impl", "GLIBC_2.18"));
        }
    }

    OrigImpl(const OrigImpl &)            = delete;
    OrigImpl &operator=(const OrigImpl &) = delete;
    OrigImpl(OrigImpl &&)                 = delete;
    OrigImpl &operator=(OrigImpl &&)      = delete;

    ~OrigImpl()
    {
        if (m_libc)
        {
            dlclose(m_libc);
        }
    }

    bool valid() const
    {
        return m_fn != nullptr;
    }

    int operator()(ThreadDestructor *func, ThreadDestructorArg *arg, DsoHandle *d) const
    {
        assert(m_fn);
        return m_fn(func, arg, d);
    }

private:
    DsoHandle *m_libc = nullptr;

    using OrigImplFn = int (*)(ThreadDestructor *func, ThreadDestructorArg *arg, DsoHandle *d);
    OrigImplFn m_fn  = nullptr;
};

struct DestructorInfo
{
    ThreadDestructor    *func;
    ThreadDestructorArg *arg;
};

// Called upon thread destruction (not main thread!)
void my_thread_atexit_cleanup(ThreadDestructorArg *arg)
{
    std::unique_ptr<std::vector<DestructorInfo>> list(
        static_cast<std::vector<DestructorInfo> *>(static_cast<void *>(arg)));
    if (!list)
    {
        return;
    }

    // Call all destructors
    for (DestructorInfo &info : *list)
    {
        info.func(info.arg);
    }
}

void pthread_thread_atexit_cleanup(PthreadDestructorArg *arg)
{
    my_thread_atexit_cleanup(static_cast<ThreadDestructorArg *>(arg));
}

pthread_key_t &ThreadKey()
{
    static pthread_key_t key;
    return key;
}

pthread_once_t &ThreadKeyOnce()
{
    static pthread_once_t keyOnce = PTHREAD_ONCE_INIT;
    return keyOnce;
}

bool IsMainThread()
{
    // Works on Linux.
    // Ref: https://stackoverflow.com/questions/4867839/how-can-i-tell-if-pthread-self-is-the-main-first-thread-in-the-process
    return syscall(SYS_gettid) == getpid();
}

std::vector<DestructorInfo> *&MainThreadDestructorList()
{
    // Destructor list to be used for objects in main thread.
    static std::vector<DestructorInfo> *list = nullptr;
    return list;
}

std::unique_ptr<std::vector<DestructorInfo>> CreateDestructorList() noexcept
{
    try
    {
        return std::make_unique<std::vector<DestructorInfo>>();
    }
    catch (...)
    {
        return nullptr;
    }
}

int my_thread_atexit_impl(ThreadDestructor *func, ThreadDestructorArg *arg, [[maybe_unused]] DsoHandle *d)
{
    std::vector<DestructorInfo> *list = nullptr;

    if (IsMainThread())
    {
        auto &mainThreadList = MainThreadDestructorList();

        // List not created yet?
        if (mainThreadList == nullptr)
        {
            // Create it!
            auto newList = CreateDestructorList();
            if (!newList)
            {
                return -1;
            }

            list           = newList.get();
            mainThreadList = newList.release();

            // Make sure it's cleaned up when main thread exits.
            atexit(
                [] {
                    my_thread_atexit_cleanup(
                        static_cast<ThreadDestructorArg *>(static_cast<void *>(MainThreadDestructorList())));
                });
        }
        else
        {
            list = mainThreadList;
        }
    }
    else
    {
        // For other threads, we have to use pthread's TLS functionality. We can't use
        // C++'s because it'll lead to infinite recursion, as it'll end up calling the current function
        // to set up the destructor.

        // Make sure we create the key only once
        pthread_once(&ThreadKeyOnce(),
                     []
                     {
                         // At every thread destruction (not main thread!), it'll call the cleanup function, passing
                         // the list as parameter.
                         int ret = pthread_key_create(&ThreadKey(), &pthread_thread_atexit_cleanup);
                         (void)ret;
                         assert(ret == 0);
                     });

        // TLS list not created yet?
        list = static_cast<std::vector<DestructorInfo> *>(pthread_getspecific(ThreadKey()));
        if (list == nullptr)
        {
            // Create it!
            auto newList = CreateDestructorList();
            if (!newList)
            {
                return -1;
            }
            list = newList.get();

            // Assign it to current thread!
            int ret = pthread_setspecific(ThreadKey(), list);
            (void)ret;
            assert(ret == 0);
            if (ret != 0)
            {
                return -1;
            }
            newList.release();
        }
    }

    // Add the destructor info to the list
    DestructorInfo info;
    info.func = func;
    info.arg  = arg;

    assert(list);
    list->push_back(info);
    return 0;
}

} // namespace

extern "C"
{
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-declarations"

__attribute__((weak)) int __cxa_thread_atexit_impl(ThreadDestructor *func, ThreadDestructorArg *arg, DsoHandle *d)
{
    static OrigImpl origImpl;

    // Do we have the original glibc implementation available?
    if (origImpl.valid())
    {
        // Call it!
        return origImpl(func, arg, d);
    }
    else
    {
        // Use our own.
        return my_thread_atexit_impl(func, arg, d);
    }
}

#pragma GCC diagnostic pop
}
