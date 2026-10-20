/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NVCV_PYTHON_PRIV_SUBMISSION_HPP
#define NVCV_PYTHON_PRIV_SUBMISSION_HPP

#include "Stream.hpp"

#include <common/PyUtil.hpp>
#include <nvcv/python/LockMode.hpp>

#ifdef NB_FREE_THREADED
#    include <algorithm>
#    include <atomic>
#    include <functional>
#    include <mutex>

namespace nvcvpy::priv {

// One allocation can have many Python wrappers. Both the host enqueue lease
// and the CUDA dependency history belong to the allocation, not its wrappers.
class SubmissionState
{
public:
    struct Access
    {
        cudaStream_t                             stream = nullptr;
        int                                      device = -1;
        std::shared_ptr<const StreamHandleOwner> owner;
    };

    ~SubmissionState();
    // prepare requires a reservation; exports require READWRITE. seed acquires its own.
    // The reservation also keeps reader-cache owners alive and excludes writes.
    void   prepare(Stream &stream, int device, LockMode mode, StreamOwnerBag &retiredOwners);
    void   seed(cudaStream_t stream, int device);
    Access exportAccess();
    void   exportTo(const Access &consumer);

    // Requires a READWRITE reservation, like write preparation below.
    bool isOrderedOn(const Stream &stream, int device) const;

private:
    void                                   wait(const Access &previous, const Access &next);
    nanobind::ft_mutex                     m_mutex;
    std::atomic<const StreamHandleOwner *> m_lastReader{nullptr};
    std::optional<Access>                  m_writer;
    std::vector<Access>                    m_readers;
    std::optional<Access>                  m_handoff;
    std::unordered_map<int, cudaEvent_t>   m_events;
};

using SubmissionStates = std::vector<std::shared_ptr<SubmissionState>>;
std::shared_ptr<SubmissionState> AllocationSubmissionState(const void *pointer, int device = -1);
SubmissionState::Access          ExportSubmissionAccess(const SubmissionStates &states);

class SubmissionLocks
{
public:
    ~SubmissionLocks()
    {
        unlock();
    }

    SubmissionLocks()                                   = default;
    SubmissionLocks(const SubmissionLocks &)            = delete;
    SubmissionLocks &operator=(const SubmissionLocks &) = delete;

    void add(const SubmissionStates &states, LockMode mode)
    {
        if (states.size() == 1)
            add(*states.front(), mode);
        else if (!states.empty())
            addMultiple(states, mode);
    }

    void lock();
    bool try_lock();
    void unlock();

    void prepare(Stream &stream, int device)
    {
        for (auto *entry = entriesBegin(), *last = entriesEnd(); entry != last; ++entry)
            entry->state->prepare(stream, device, entry->mode, m_retiredOwners);
    }

    void exportTo(const SubmissionState::Access &consumer)
    {
        for (auto *entry = entriesBegin(), *last = entriesEnd(); entry != last; ++entry)
            entry->state->exportTo(consumer);
    }

    void add(SubmissionState &state, LockMode mode)
    {
        Entry entry{&state, mode};
        if (m_entries.empty())
        {
            if (!m_single.state || m_single.state == &state)
            {
                m_single = {&state, static_cast<LockMode>(m_single.mode | mode)};
                return;
            }
            m_entries.reserve(2);
            m_entries.push_back(m_single);
        }
        if (m_entries.back() < entry)
            m_entries.push_back(entry);
        else
        {
            auto position = std::lower_bound(m_entries.begin(), m_entries.end(), entry);
            if (position != m_entries.end() && position->state == &state)
                position->mode = static_cast<LockMode>(position->mode | mode);
            else
                m_entries.insert(position, entry);
        }
    }

private:
    void addMultiple(const SubmissionStates &states, LockMode mode);

    struct Entry
    {
        SubmissionState *state;
        LockMode         mode;

        bool operator<(const Entry &other) const
        {
            return std::less<SubmissionState *>{}(state, other.state);
        }
    };

    const Entry *entriesBegin() const
    {
        return m_entries.empty() ? &m_single : m_entries.data();
    }

    const Entry *entriesEnd() const
    {
        return m_entries.empty() ? &m_single + (m_single.state != nullptr) : m_entries.data() + m_entries.size();
    }

    // Most exports reserve one allocation; larger sets keep the sorted vector.
    Entry              m_single{nullptr, LOCK_MODE_NONE};
    std::vector<Entry> m_entries;
    StreamOwnerBag     m_retiredOwners;
    struct Registry;
    static Registry &registry();
    bool             tryRegister(Registry &registry);
    bool             conflicts(const SubmissionLocks &other) const;

    SubmissionLocks *m_next       = nullptr;
    bool             m_registered = false;
};

} // namespace nvcvpy::priv
#endif
#endif
