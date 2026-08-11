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

#ifndef NVCV_PYTHON_PRIV_CONTAINER_HPP
#define NVCV_PYTHON_PRIV_CONTAINER_HPP

#include "Cache.hpp"
#include "Resource.hpp"

#include <common/Assert.hpp>
#include <nvcv/python/Container.hpp>

#include <memory>

namespace nvcvpy::priv {
namespace py = pybind11;

class Container
    : public Resource
    , public CacheItem
{
public:
    static void Export(py::module &m);

protected:
    Container() = default;
};

inline std::shared_ptr<Container> SharedContainerFrom(Container &container)
{
    return std::static_pointer_cast<Container>(static_cast<Resource &>(container).shared_from_this());
}

inline std::shared_ptr<const Container> SharedContainerFrom(const Container &container)
{
    return std::static_pointer_cast<const Container>(static_cast<const Resource &>(container).shared_from_this());
}

template<class T>
std::shared_ptr<T> SharedContainerFrom(T &container)
{
    return std::static_pointer_cast<T>(SharedContainerFrom(static_cast<Container &>(container)));
}

template<class T>
std::shared_ptr<const T> SharedContainerFrom(const T &container)
{
    return std::static_pointer_cast<const T>(SharedContainerFrom(static_cast<const Container &>(container)));
}

class ExternalContainer : public Container
{
public:
    explicit ExternalContainer(nvcvpy::Container &extCont)
        : m_extCont(extCont)
        , m_size_inbytes{doComputeSizeInBytes()}
    {
    }

    int64_t GetSizeInBytes() const override
    {
        // m_size_inbytes == -1 indicates failure case and value has not been computed yet
        NVCV_ASSERT(m_size_inbytes != -1
                    && "ExternalContainer has m_size_inbytes == -1, ie m_size_inbytes has not been correctly set");
        return m_size_inbytes;
    }

private:
    nvcvpy::Container &m_extCont;

    int64_t doComputeSizeInBytes() const
    {
        // ExternalCacheItems (CacheItems outside of nvcv, eg. operators from cvcuda) will not pollute the
        // Cache, thus for now we say they've no impact on the Cache
        return 0;
    }

    int64_t m_size_inbytes = -1;

    const IKey &key() const override
    {
        return m_extCont.key();
    }
};

} // namespace nvcvpy::priv

#endif // NVCV_PYTHON_PRIV_CONTAINER_HPP
