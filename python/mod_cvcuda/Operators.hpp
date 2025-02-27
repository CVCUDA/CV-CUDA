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

#include <common/Hash.hpp>
#include <cvcuda/cuda_tools/MathOps.hpp>
#include <nvcv/python/Array.hpp>
#include <nvcv/python/Cache.hpp>
#include <nvcv/python/Container.hpp>
#include <nvcv/python/ImageFormat.hpp>
#include <nvcv/python/Shape.hpp>
#include <nvcv/python/Size.hpp>
#include <nvcv/python/Tensor.hpp>
#include <nvcv/python/TensorBatch.hpp>
#include <pybind11/pybind11.h>

#include <nvcv/python/Fwd.hpp>

namespace nvcvpy::util {
}

namespace cvcudapy {

using nvcvpy::Array;
using nvcvpy::CreateNVCVTensorShape;
using nvcvpy::CreateShape;
using nvcvpy::Image;
using nvcvpy::ImageBatchVarShape;
using nvcvpy::LockMode;
using nvcvpy::ResourceGuard;
using nvcvpy::Shape;
using nvcvpy::Stream;
using nvcvpy::Tensor;
using nvcvpy::TensorBatch;

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
void ExportOpJointBilateralFilter(py::module &m);
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
void ExportOpThreshold(py::module &m);
void ExportOpBndBox(py::module &m);
void ExportOpBoxBlur(py::module &m);
void ExportOpBrightnessContrast(py::module &m);
void ExportOpColorTwist(py::module &m);
void ExportOpHQResize(py::module &m);
void ExportOpRemap(py::module &m);
void ExportOpCropFlipNormalizeReformat(py::module &m);
void ExportOpAdaptiveThreshold(py::module &m);
void ExportOpNonMaximumSuppression(py::module &m);
void ExportOpOSD(py::module &m);
void ExportOpRandomResizedCrop(py::module &m);
void ExportOpGaussianNoise(py::module &m);
void ExportOpMinMaxLoc(py::module &m);
void ExportOpSIFT(py::module &m);
void ExportOpHistogram(py::module &m);
void ExportOpInpaint(py::module &m);
void ExportOpHistogramEq(py::module &m);
void ExportOpMinAreaRect(py::module &m);
void ExportOpAdvCvtColor(py::module &m);
void ExportOpLabel(py::module &m);
void ExportOpPairwiseMatcher(py::module &m);
void ExportOpStack(py::module &m);
void ExportOpFindHomography(py::module &m);
void ExportOpResizeCropConvertReformat(py::module &m);

// Helper class that serves as generic python-side operator class.
// OP: native operator class
// CTOR: ctor signature
template<class OP, class CTOR>
class PyOperator;

template<class OP, class... CTOR_ARGS>
class PyOperator<OP, void(CTOR_ARGS...)> : public nvcvpy::Container
{
public:
    // This defines a generic cache key class for any cvcuda::OP operator.
    // It allows for reusing cache objects (instead of allocating new ones) only if the hash and key equal match.
    // By default, the hash uses the OP type (in IKey) and OP ctor args, and the equal checks if all args are equal.
    // This may be too restrict for operators with payloads, for them it may be better to specialize PyOperator/Key.
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
            return apply([](auto... items) { return nvcvpy::util::ComputeHash(items...); }, m_args);
        }

        bool doIsCompatible(const nvcvpy::IKey &that_) const override
        {
            const Key &that = static_cast<const Key &>(that_);
            return m_args == that.m_args;
        }

        std::tuple<std::decay_t<CTOR_ARGS>...> m_args;
    };

    PyOperator(CTOR_ARGS &&...args)
        : m_key(args...)
        , m_op{std::forward<CTOR_ARGS>(args)...}
    {
    }

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

    // The static fetch function is used to fetch one object from a sub-set of objects from cache.
    // All objects passed in the cache argument already match the operator object, this second-level fetch
    // allows to choose one of them and further do cache operations on the matched items.
    static std::shared_ptr<nvcvpy::ICacheItem> fetch(std::vector<std::shared_ptr<nvcvpy::ICacheItem>> &cache)
    {
        assert(!cache.empty());
        // This generic operator just returns the first one found in cache.
        return cache[0];
    }

private:
    // Order is important
    Key m_key;
    OP  m_op;
};

// Creates an operator instance.
// Either gets it from the resource cache or creates one from scratch.
// When creating, it'll be added to the cache.
template<class PyOP, class... CTOR_ARGS>
std::shared_ptr<PyOP> CreateOperatorEx(CTOR_ARGS &&...args)
{
    // The Key class is defined by the operator.
    using Key = typename PyOP::Key;

    // Creates a key out of the operator's ctor parameters
    Key key{args...};

    // Try to fetch it from cache
    std::vector<std::shared_ptr<nvcvpy::ICacheItem>> vcont = nvcvpy::Cache::fetch(key);

    // None found?
    if (vcont.empty())
    {
        // Creates a new one
        auto op = std::shared_ptr<PyOP>(new PyOP(std::forward<CTOR_ARGS>(args)...));

        // Adds to the resource cache
        nvcvpy::Cache::add(*op);

        return op;
    }
    else
    {
        auto op = std::dynamic_pointer_cast<PyOP>(PyOP::fetch(vcont));
        assert(op);
        return op;
    }
}

template<class OP, class... CTOR_ARGS>
std::shared_ptr<PyOperator<OP, void(CTOR_ARGS...)>> CreateOperator(CTOR_ARGS &&...args)
{
    return CreateOperatorEx<PyOperator<OP, void(CTOR_ARGS...)>>(std::forward<CTOR_ARGS>(args)...);
}

} // namespace cvcudapy

namespace nvcv {
inline size_t ComputeHash(const ImageFormat fmt)
{
    using nvcvpy::util::ComputeHash;
    return ComputeHash(fmt.cvalue());
}
} // namespace nvcv
