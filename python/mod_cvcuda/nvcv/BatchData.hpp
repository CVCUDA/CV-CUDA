/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NVCV_PYTHON_PRIV_BATCHDATA_HPP
#define NVCV_PYTHON_PRIV_BATCHDATA_HPP

#include "BatchLock.hpp"
#include "Resource.hpp"

#include <limits>
#include <vector>

namespace nvcvpy::priv {

template<class Native, class Child>
struct BatchData
#ifdef NB_FREE_THREADED
    : Resource
#endif
{
    explicit BatchData(int capacity)
        : impl(capacity)
    {
        list.reserve(capacity);
    }

#ifdef NB_FREE_THREADED
    BatchData(const BatchData &other)
        : impl(other.impl.capacity())
        , list(other.list)
        , size(other.size)
    {
        for (const auto &child : list) impl.pushBack(child->impl());
    }

    RetainedMemory retainedMemory() const override
    {
        RetainedMemory result{size, 0};
        for (const auto &child : list)
        {
            auto memory = child->retainedMemory();
            result.knownBytes += std::min(memory.knownBytes, std::numeric_limits<uint64_t>::max() - result.knownBytes);
            result.unknownResources
                += std::min(memory.unknownResources, std::numeric_limits<size_t>::max() - result.unknownResources);
        }
        return result;
    }

    Resource &submissionChildren()
    {
        // A snapshot retains the children and their allocation identities.
        // Flatten their access set once, until the batch is modified again.
        if (childrenDirty)
        {
            SubmissionStates states;
            bool             readOnly = false;
            for (const auto &child : list)
            {
                const auto &childStates = child->submissionStates();
                states.insert(states.end(), childStates.begin(), childStates.end());
                readOnly |= child->readOnly();
            }
            children.setSubmissionStates(std::move(states));
            children.setReadOnly(readOnly);
            childrenDirty = false;
        }
        return children;
    }

    struct ChildAccess : Resource
    {
        ChildAccess()
            : Resource(false)
        {
        }
    } children;

    bool childrenDirty = true;
#endif
    Native                              impl;
    std::vector<std::shared_ptr<Child>> list;
#ifdef NB_FREE_THREADED
    uint64_t size = 0;
#endif
};

template<class Native, class Child>
class BatchStorage
{
    using Data = BatchData<Native, Child>;

public:
    explicit BatchStorage(int capacity)
#ifdef NB_FREE_THREADED
        : m_data(std::make_shared<Data>(capacity))
#else
        : m_data(capacity)
#endif
    {
    }

    Data *operator->() const
    {
#ifdef NB_FREE_THREADED
        return m_data.get();
#else
        return &m_data;
#endif
    }

    void writable()
    {
#ifdef NB_FREE_THREADED
        if (!m_data.unique())
            m_data = std::make_shared<Data>(*m_data);
        m_data->childrenDirty = true;
#endif
    }
#ifdef NB_FREE_THREADED
    std::shared_ptr<Data> snapshot() const
    {
        return m_data;
    }
#endif
private:
#ifdef NB_FREE_THREADED
    std::shared_ptr<Data> m_data;
#else
    mutable Data m_data;
#endif
};

template<class Batch, class Child>
class BatchIterator
{
public:
    explicit BatchIterator(std::shared_ptr<const Batch> batch)
        : m_batch(std::move(batch))
        , m_version(m_batch->iterationVersion())
    {
    }

    std::shared_ptr<Child> next()
    {
        return m_batch->iterationAt(m_index, m_version);
    }

private:
    std::shared_ptr<const Batch> m_batch;
    uint64_t                     m_version;
    size_t                       m_index = 0;
};
} // namespace nvcvpy::priv
#endif
