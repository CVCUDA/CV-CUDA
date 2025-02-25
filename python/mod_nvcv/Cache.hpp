/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef NVCV_PYTHON_PRIV_CACHE_HPP
#define NVCV_PYTHON_PRIV_CACHE_HPP

#include "Object.hpp"

#include <common/Assert.hpp>
#include <common/Hash.hpp>
#include <nvcv/python/Cache.hpp>
#include <pybind11/pybind11.h>

#include <unordered_set>
#include <vector>

namespace nvcvpy::priv {

namespace py = pybind11;

class PYBIND11_EXPORT CacheItem : public virtual Object
{
public:
    uint64_t id() const;

    virtual const IKey &key() const = 0;

    std::shared_ptr<CacheItem>       shared_from_this();
    std::shared_ptr<const CacheItem> shared_from_this() const;

    virtual int64_t GetSizeInBytes() const = 0;

    bool isInUse() const;

protected:
    CacheItem();

private:
    uint64_t m_id;
};

class ExternalCacheItem : public CacheItem
{
public:
    ExternalCacheItem(std::shared_ptr<nvcvpy::ICacheItem> obj_)
        : obj(obj_)
        , m_size_inbytes(doComputeSizeInBytes())
    {
    }

    int64_t GetSizeInBytes() const override
    {
        // m_size_inbytes == -1 indicates failure case and value has not been computed yet
        NVCV_ASSERT(m_size_inbytes != -1
                    && "ExternalCacheItem has m_size_inbytes == -1, ie m_size_inbytes has not been correctly set");
        return m_size_inbytes;
    }

    std::shared_ptr<nvcvpy::ICacheItem> obj;

    const IKey &key() const override
    {
        return obj->key();
    }

private:
    int64_t doComputeSizeInBytes()
    {
        // ExternalCacheItems (CacheItems outside of nvcv, eg. operators from cvcuda) will not pollute the
        // Cache, thus for now we say they've no impact on the Cache
        return 0;
    }

    int64_t m_size_inbytes = -1;
};

class PYBIND11_EXPORT Cache
{
public:
    static void Export(py::module &m);

    static Cache &Instance();
    static void   ClearAll();
    static size_t TotalSize();

    void add(CacheItem &container);
    void removeAllNotInUseMatching(const IKey &key);

    std::vector<std::shared_ptr<CacheItem>> fetch(const IKey &key) const;
    std::shared_ptr<CacheItem>              fetchOne(const IKey &key) const;

#ifndef NDEBUG
    // Make this function available only in Debug builds
    void dbgPrintCacheForKey(const IKey &key, const std::string &prefix = "");
#endif

    template<class T>
    std::vector<std::shared_ptr<T>> fetchAll() const
    {
        std::vector<std::shared_ptr<T>> out;

        doIterateThroughItems(
            [&out](CacheItem &item)
            {
                if (auto titem = std::dynamic_pointer_cast<T>(item.shared_from_this()))
                {
                    out.emplace_back(std::move(titem));
                }
            });
        return out;
    }

    void   clear();
    size_t size() const;

    void    setCacheLimit(int64_t new_cache_limit);
    int64_t getCacheLimit() const;
    int64_t getCurrentSizeInBytes();

private:
    inline static std::unordered_set<Cache *> instances;

    struct Impl;
    std::unique_ptr<Impl> pimpl;

    Cache();
    ~Cache();

    void    doIterateThroughItems(const std::function<void(CacheItem &item)> &fn) const;
    int64_t doGetCurrentSizeInBytes() const;
    int64_t doGetCacheLimit() const;
};

} // namespace nvcvpy::priv

#endif // NVCV_PYTHON_PRIV_CACHE_HPP
