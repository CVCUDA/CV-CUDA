/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NVCV_PYTHON_PRIV_BATCHLOCK_HPP
#define NVCV_PYTHON_PRIV_BATCHLOCK_HPP

#include <common/Binding.hpp>
#include <common/PyUtil.hpp>

#include <algorithm>
#include <array>
#include <mutex>

namespace nvcvpy::priv {

class BatchMutex
{
public:
    void lock()
    {
#ifdef NB_FREE_THREADED
        if (!m_mutex.try_lock())
        {
            util::ScopedThreadDetach release;
            m_mutex.lock();
        }
#endif
    }

    void unlock()
    {
#ifdef NB_FREE_THREADED
        m_mutex.unlock();
#endif
    }

#ifdef NB_FREE_THREADED
private:
    std::recursive_mutex m_mutex;
#endif
};

#ifdef NB_FREE_THREADED
// The binding holds batch structure from metadata inspection through enqueue.
// ResourceGuard releases it before retirement can run Python finalizers or wait
// for GPU completion. A call that never submits releases it on ordinary unwind.
class BatchLockScope
{
public:
    BatchLockScope(BatchMutex **mutexes, size_t count)
        : m_mutexes(mutexes)
        , m_previous(s_current)
    {
        if (count != 0)
        {
            std::sort(mutexes, mutexes + count, std::less<BatchMutex *>{});
            count = std::unique(mutexes, mutexes + count) - mutexes;
        }
        try
        {
            for (size_t i = 0; i < count; ++i)
            {
                mutexes[i]->lock();
                ++m_locked;
            }
        }
        catch (...)
        {
            unlock();
            throw;
        }
        s_current = this;
    }

    ~BatchLockScope()
    {
        unlock();
        s_current = m_previous;
    }

    BatchLockScope(const BatchLockScope &)            = delete;
    BatchLockScope &operator=(const BatchLockScope &) = delete;

    static void Release()
    {
        for (auto *scope = s_current; scope; scope = scope->m_previous) scope->unlock();
    }

private:
    void unlock()
    {
        while (m_locked) m_mutexes[--m_locked]->unlock();
    }

    BatchMutex                               **m_mutexes;
    size_t                                     m_locked = 0;
    BatchLockScope                            *m_previous;
    inline static thread_local BatchLockScope *s_current = nullptr;
};

template<class T>
auto GetBatchMutex(const T &value, int) -> decltype(&value.batchMutex())
{
    return &value.batchMutex();
}

template<class T>
BatchMutex *GetBatchMutex(const T &, long)
{
    return nullptr;
}

template<class T>
auto GetBatchMutex(T *value, int) -> decltype(GetBatchMutex(*value, 0))
{
    return value ? GetBatchMutex(*value, 0) : nullptr;
}

template<class T>
auto GetBatchMutex(const std::optional<T> &value, int) -> decltype(GetBatchMutex(*value, 0))
{
    return value ? GetBatchMutex(*value, 0) : nullptr;
}

template<class... Args>
auto BatchMutexes(const Args &...args)
{
    return std::array<BatchMutex *, sizeof...(Args)>{GetBatchMutex(args, 0)...};
}
#endif
} // namespace nvcvpy::priv
#endif
