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

#ifndef NVCV_PYTHON_PRIV_STREAM_HPP
#define NVCV_PYTHON_PRIV_STREAM_HPP

#include "Cache.hpp"
#include "Object.hpp"

#include <cuda_runtime.h>
#include <nvcv/python/LockMode.hpp>

#include <initializer_list>
#include <memory>
#include <unordered_map>
#include <vector>

namespace nvcvpy::priv {

class Resource;

class IExternalStream
{
public:
    virtual cudaStream_t handle() const        = 0;
    virtual py::object   wrappedObject() const = 0;
};

using LockResources = std::unordered_multimap<LockMode, std::shared_ptr<const Resource>>;

class PYBIND11_EXPORT Stream : public CacheItem
{
public:
    static void Export(py::module &m);

    static Stream &Current();

    static std::shared_ptr<Stream> Create();

    ~Stream();

    std::shared_ptr<Stream>       shared_from_this();
    std::shared_ptr<const Stream> shared_from_this() const;

    void activate();
    void deactivate(py::object exc_type, py::object exc_value, py::object exc_tb);

    void holdResources(LockResources usedResources);

    void         sync();
    cudaStream_t handle() const;

    // Returns the cuda handle in python
    intptr_t pyhandle() const;

    Stream(IExternalStream &extStream);

    friend std::ostream &operator<<(std::ostream &out, const Stream &stream);

private:
    Stream(Stream &&) = delete;
    Stream();

    class Key final : public IKey
    {
    private:
        virtual size_t doGetHash() const override;
        virtual bool   doIsEqual(const IKey &that) const override;
    };

    virtual const Key &key() const override
    {
        static Key key;
        return key;
    }

    bool         m_owns;
    cudaStream_t m_handle;
    py::object   m_wrappedObj;
};

} // namespace nvcvpy::priv

#endif // NVCV_PYTHON_PRIV_STREAM_HPP
