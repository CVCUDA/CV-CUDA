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

#ifndef NVCV_PYTHON_PRIV_CONTAINER_HPP
#define NVCV_PYTHON_PRIV_CONTAINER_HPP

#include "Cache.hpp"
#include "Resource.hpp"

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

    std::shared_ptr<Container>       shared_from_this();
    std::shared_ptr<const Container> shared_from_this() const;

protected:
    Container() = default;
};

class ExternalContainer : public Container
{
public:
    explicit ExternalContainer(nvcvpy::Container &extCont)
        : m_extCont(extCont)
    {
    }

private:
    nvcvpy::Container &m_extCont;

    const IKey &key() const override
    {
        return m_extCont.key();
    }
};

} // namespace nvcvpy::priv

#endif // NVCV_PYTHON_PRIV_CONTAINER_HPP
