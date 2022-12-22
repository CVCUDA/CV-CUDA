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

#include <dlfcn.h>
#include <pthread.h>
#include <stdlib.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <cassert>
#include <vector>

// ================================================
//__cxa_thread_atexit_impl

// We need to support glibc-2.17, which doesn't export
// __cxa_thread_atexit_impl, need by libstdc++ >= 8. This
// allows us to support old distros like CentOS 7.

namespace {

// Encapsulates the original thread_atexit implementation
// by retrieving the function from libc.so.6, if it exists.
class OrigImpl
{
public:
    OrigImpl()
    {
        m_libc = dlopen("libc.so.6", RTLD_LOCAL | RTLD_LAZY);
        if (m_libc)
        {
            m_fn = (OrigImplFn)dlvsym(m_libc, "__cxa_thread_atexit_impl", "GLIBC_2.18");
        }
    }

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

    int operator()(void (*func)(void *), void *arg, void *d) const
    {
        assert(m_fn);
        return m_fn(func, arg, d);
    }

private:
    void *m_libc = nullptr;

    using OrigImplFn = int (*)(void (*func)(void *), void *arg, void *d);
    OrigImplFn m_fn  = nullptr;
};

struct DestructorInfo
{
    void (*func)(void *);
    void *arg;
};

// Called upon thread destruction (not main thread!)
void my_thread_atexit_cleanup(void *arg)
{
    auto *list = reinterpret_cast<std::vector<DestructorInfo> *>(arg);

    // Call all destructors
    for (DestructorInfo &info : *list)
    {
        info.func(info.arg);
    }

    delete list;
}

static pthread_key_t  g_key;
static pthread_once_t g_keyOnce = PTHREAD_ONCE_INIT;

bool IsMainThread()
{
    // Works on Linux.
    // Ref: https://stackoverflow.com/questions/4867839/how-can-i-tell-if-pthread-self-is-the-main-first-thread-in-the-process
    return syscall(SYS_gettid) == getpid();
}

// Destructor list to be used for objects in main thread.
static std::vector<DestructorInfo> *g_ListMainThread = nullptr;

int my_thread_atexit_impl(void (*func)(void *), void *arg, void *d)
{
    std::vector<DestructorInfo> *list;

    if (IsMainThread())
    {
        // List not created yet?
        if (g_ListMainThread == nullptr)
        {
            // Create it!
            list = new (std::nothrow) std::vector<DestructorInfo>();
            if (list == nullptr)
            {
                return -1;
            }

            g_ListMainThread = list;

            // Make sure it's cleaned up when main thread exits.
            atexit([] { my_thread_atexit_cleanup(g_ListMainThread); });
        }
    }
    else
    {
        // For other threads, we have to use pthread's TLS functionality. We can't use
        // C++'s because it'll lead to infinite recursion, as it'll end up calling the current function
        // to set up the destructor.

        // Make sure we create the key only once
        pthread_once(&g_keyOnce,
                     []
                     {
                         // At every thread destruction (not main thread!), it'll call the cleanup function, passing
                         // the list as parameter.
                         int ret = pthread_key_create(&g_key, &my_thread_atexit_cleanup);
                         (void)ret;
                         assert(ret == 0);
                     });

        // TLS list not created yet?
        list = reinterpret_cast<std::vector<DestructorInfo> *>(pthread_getspecific(g_key));
        if (list == nullptr)
        {
            // Create it!
            list = new (std::nothrow) std::vector<DestructorInfo>();
            if (list == nullptr)
            {
                return -1;
            }

            // Assign it to current thread!
            int ret = pthread_setspecific(g_key, list);
            (void)ret;
            assert(ret == 0);
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

__attribute__((weak)) int __cxa_thread_atexit_impl(void (*func)(void *), void *arg, void *d)
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
