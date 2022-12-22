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

#ifndef NVCV_PYTHON_CACHE_HPP
#define NVCV_PYTHON_CACHE_HPP

#include "CAPI.hpp"

#include <pybind11/pybind11.h>

namespace nvcvpy {

namespace py = ::pybind11;

class IKey
{
public:
    virtual ~IKey() = default;

    size_t hash() const
    {
        size_t h = doGetHash();

        // Make hash dependent on concrete object type
        h ^= typeid(*this).hash_code() << 1;
        return h;
    }

    bool operator==(const IKey &that) const
    {
        if (typeid(*this) == typeid(that))
        {
            return doIsEqual(that);
        }
        else
        {
            return false;
        }
    }

private:
    virtual size_t doGetHash() const                 = 0;
    virtual bool   doIsEqual(const IKey &that) const = 0;
};

class ICacheItem : public std::enable_shared_from_this<ICacheItem>
{
public:
    virtual ~ICacheItem() = default;

    virtual const IKey &key() const       = 0;
    virtual py::object  container() const = 0;
};

class Cache
{
public:
    static void add(ICacheItem &item)
    {
        capi().Cache_Add(&item);
    }

    static std::vector<std::shared_ptr<ICacheItem>> fetch(const IKey &key)
    {
        std::unique_ptr<ICacheItem *[]> list
        {
            capi().Cache_Fetch(&key)
        };

        std::vector<std::shared_ptr<ICacheItem>> out;
        for (int i = 0; list[i]; ++i)
        {
            out.emplace_back(list[i]->shared_from_this());
        }
        return out;
    }
};

} // namespace nvcvpy

#endif // NVCV_PYTHON_CACHE_HPP
