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

#include <common/Hash.hpp>
#include <nvcv/python/Cache.hpp>
#include <nvcv/python/Container.hpp>
#include <nvcv/python/ImageFormat.hpp>
#include <nvcv/python/Size.hpp>
#include <pybind11/pybind11.h>

#include <nvcv/python/Fwd.hpp>

namespace nvcvpy::util {
}

namespace cvcudapy {

using nvcvpy::Image;
using nvcvpy::ImageBatchVarShape;
using nvcvpy::LockMode;
using nvcvpy::ResourceGuard;
using nvcvpy::Shape;
using nvcvpy::Stream;
using nvcvpy::Tensor;

namespace util = nvcvpy::util;
namespace py   = ::pybind11;

void ExportOpReformat(py::module &m);
void ExportOpResize(py::module &m);
void ExportOpCustomCrop(py::module &m);
void ExportOpNormalize(py::module &m);
void ExportOpConvertTo(py::module &m);
void ExportOpPadAndStack(py::module &m);
void ExportOpCopyMakeBorder(py::module &m);
void ExportOpRotate(py::module &m);
void ExportOpErase(py::module &m);
void ExportOpGaussian(py::module &m);
void ExportOpMedianBlur(py::module &m);
void ExportOpLaplacian(py::module &m);
void ExportOpAverageBlur(py::module &m);
void ExportOpConv2D(py::module &m);
void ExportOpBilateralFilter(py::module &m);
void ExportOpCenterCrop(py::module &m);
void ExportOpWarpAffine(py::module &m);
void ExportOpWarpPerspective(py::module &m);
void ExportOpChannelReorder(py::module &m);
void ExportOpMorphology(py::module &m);
void ExportOpFlip(py::module &m);
void ExportOpCvtColor(py::module &m);
void ExportOpComposite(py::module &m);
void ExportOpGammaContrast(py::module &m);
void ExportOpPillowResize(py::module &m);

// Helper class that serves as python-side operator class.
// OP: native operator class
// CTOR: ctor signature
template<class OP, class CTOR>
class PyOperator;

template<class OP, class... CTOR_ARGS>
class PyOperator<OP, void(CTOR_ARGS...)> : public nvcvpy::Container
{
public:
    template<class... AA>
    void submit(AA &&...args)
    {
        m_op(std::forward<AA>(args)...);
    }

    py::object container() const override
    {
        return *this;
    }

    const nvcvpy::IKey &key() const override
    {
        return m_key;
    }

    class Key : public nvcvpy::IKey
    {
    public:
        Key(const CTOR_ARGS &...args)
            : m_args{args...}
        {
        }

    private:
        size_t doGetHash() const override
        {
            return apply(
                [](auto... args)
                {
                    using nvcvpy::util::ComputeHash;
                    return ComputeHash(args...);
                },
                m_args);
        }

        bool doIsEqual(const nvcvpy::IKey &that_) const override
        {
            const Key &that = static_cast<const Key &>(that_);
            return m_args == that.m_args;
        }

        std::tuple<std::decay_t<CTOR_ARGS>...> m_args;
    };

private:
    template<class OP2, class... AA2>
    friend std::shared_ptr<PyOperator<OP2, void(AA2...)>> CreateOperator(AA2 &&...args);

    PyOperator(CTOR_ARGS &&...args)
        : m_key(args...)
        , m_op{std::forward<CTOR_ARGS>(args)...}
    {
    }

    // Order is important
    Key m_key;
    OP  m_op;
};

// Returns an operator instance.
// Either gets it from the resource cache or creates one from scratch.
// When creationg, it'll add it to the cache.
template<class OP, class... AA>
std::shared_ptr<PyOperator<OP, void(AA...)>> CreateOperator(AA &&...args)
{
    using PyOP = PyOperator<OP, void(AA...)>;

    // Creates a key out of the operator's ctor parameters
    typename PyOP::Key key(args...);

    // Try to fetch it from cache
    std::vector<std::shared_ptr<nvcvpy::ICacheItem>> vcont = nvcvpy::Cache::fetch(key);

    // None found?
    if (vcont.empty())
    {
        // Creates a new one
        auto op = std::shared_ptr<PyOP>(new PyOP(std::forward<AA>(args)...));

        // Adds to the resource cache
        nvcvpy::Cache::add(*op);
        return op;
    }
    else
    {
        // Get the first one found in cache
        auto op = std::dynamic_pointer_cast<PyOP>(vcont[0]);
        assert(op);
        return op;
    }
}
} // namespace cvcudapy

namespace nvcv {
inline size_t ComputeHash(const ImageFormat fmt)
{
    using nvcvpy::util::ComputeHash;
    return ComputeHash(fmt.cvalue());
}
} // namespace nvcv
