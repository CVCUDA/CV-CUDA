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

#ifndef NVCV_PYTHON_PRIV_RESOURCE_HPP
#define NVCV_PYTHON_PRIV_RESOURCE_HPP

#include "Object.hpp"

#include <nvcv/detail/CudaFwd.h>
#include <nvcv/python/LockMode.hpp>
#include <pybind11/pybind11.h>

#include <memory>

// fwd declaration from driver_types.h
typedef struct CUevent_st *cudaEvent_t;

namespace nvcvpy::priv {
namespace py = pybind11;

class Stream;

class PYBIND11_EXPORT Resource : public virtual Object
{
public:
    ~Resource();

    static void Export(py::module &m);

    uint64_t id() const;

    void submitSync(Stream &stream, LockMode mode) const;
    void submitSignal(Stream &stream, LockMode mode) const;

    // Assumes GIL is locked (is in acquired state)
    void sync(LockMode mode) const;

    std::shared_ptr<Resource>       shared_from_this();
    std::shared_ptr<const Resource> shared_from_this() const;

protected:
    Resource();

    void doSubmitSync(Stream &stream, LockMode mode) const;

    // Assumes GIL is not locked (is in released state)
    void doSync(LockMode mode) const;

private:
    // To be overriden by children if they have their own requirements
    virtual void doBeforeSync(LockMode mode) const {};
    virtual void doBeforeSubmitSync(Stream &stream, LockMode mode) const {};
    virtual void doBeforeSubmitSignal(Stream &stream, LockMode mode) const {};

    uint64_t    m_id;
    cudaEvent_t m_readEvent, m_writeEvent;
};

} // namespace nvcvpy::priv

#endif // NVCV_PYTHON_PRIV_RESOURCE_HPP
