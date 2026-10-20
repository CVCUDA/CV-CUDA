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

#include "../NvtxRange.hpp"
#include "../nvcv/Cache.hpp"
// Select the binding module's numpy-aware caster before the public wrappers
// can select their external-module C API adapter with the same include guard.
#include "../nvcv/DataType.hpp"
#include "../nvcv/Resource.hpp"
#include "NativeResourceGuard.hpp"

#include <common/Assert.hpp>
#include <common/Binding.hpp>
#include <common/Hash.hpp>
#include <cvcuda/cuda_tools/MathOps.hpp>
#include <nvcv/python/Cache.hpp>
#include <nvcv/python/Shape.hpp>
#include <nvcv/python/Size.hpp>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <mutex>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <nvcv/python/Fwd.hpp>

namespace cvcudapy {

using nvcvpy::CreateNVCVTensorShape;
using nvcvpy::CreateShape;
using nvcvpy::LockMode;
using nvcvpy::Shape;

using ResourceGuard = NativeResourceGuard;

namespace util = nvcvpy::util;
namespace py   = ::nanobind;

// Operator state is shared across queued submissions by default. Operators
// with mutable staging or workspace owned by the operator object opt into a
// completion lease by specializing this trait for either the native operator
// type or a custom Python operator-state type.
template<class T, class = void>
struct OperatorCacheUsesCompletionLease : std::false_type
{
};

template<class T>
struct OperatorCacheUsesCompletionLease<T, std::void_t<typename T::NativeOperator>>
    : OperatorCacheUsesCompletionLease<typename T::NativeOperator>
{
};

void ExportOpReformat(py::module_ &m);
void ExportOpResize(py::module_ &m);
void ExportOpCustomCrop(py::module_ &m);
void ExportOpNormalize(py::module_ &m);
void ExportOpConvertTo(py::module_ &m);
void ExportOpPadAndStack(py::module_ &m);
void ExportOpCopyMakeBorder(py::module_ &m);
void ExportOpRotate(py::module_ &m);
void ExportOpErase(py::module_ &m);
void ExportOpGaussian(py::module_ &m);
void ExportOpMedianBlur(py::module_ &m);
void ExportOpLaplacian(py::module_ &m);
void ExportOpAverageBlur(py::module_ &m);
void ExportOpConv2D(py::module_ &m);
void ExportOpBilateralFilter(py::module_ &m);
void ExportOpJointBilateralFilter(py::module_ &m);
void ExportOpCenterCrop(py::module_ &m);
void ExportOpWarpAffine(py::module_ &m);
void ExportOpWarpPerspective(py::module_ &m);
void ExportOpChannelReorder(py::module_ &m);
void ExportOpSplit(py::module_ &m);
void ExportOpMerge(py::module_ &m);
void ExportOpMorphology(py::module_ &m);
void ExportOpFlip(py::module_ &m);
void ExportOpCvtColor(py::module_ &m);
void ExportOpComposite(py::module_ &m);
void ExportOpGammaContrast(py::module_ &m);
void ExportOpPillowResize(py::module_ &m);
void ExportOpThreshold(py::module_ &m);
void ExportOpBndBox(py::module_ &m);
void ExportOpBoxBlur(py::module_ &m);
void ExportOpBrightnessContrast(py::module_ &m);
void ExportOpColorTwist(py::module_ &m);
void ExportOpHQResize(py::module_ &m);
void ExportOpRemap(py::module_ &m);
void ExportOpCropFlipNormalizeReformat(py::module_ &m);
void ExportOpAdaptiveThreshold(py::module_ &m);
void ExportOpNonMaximumSuppression(py::module_ &m);
void ExportOpOSD(py::module_ &m);
void ExportOpRandomResizedCrop(py::module_ &m);
void ExportOpGaussianNoise(py::module_ &m);
void ExportOpMinMaxLoc(py::module_ &m);
void ExportOpSIFT(py::module_ &m);
void ExportOpHistogram(py::module_ &m);
void ExportOpInpaint(py::module_ &m);
void ExportOpHistogramEq(py::module_ &m);
void ExportOpMinAreaRect(py::module_ &m);
void ExportOpAdvCvtColor(py::module_ &m);
void ExportOpLabel(py::module_ &m);
void ExportOpPairwiseMatcher(py::module_ &m);
void ExportOpStack(py::module_ &m);
void ExportOpFindHomography(py::module_ &m);
void ExportOpResizeCropConvertReformat(py::module_ &m);
void ExportOpCLAHE(py::module_ &m);
void ExportOpInvert(py::module_ &m);
void ExportOpSolarize(py::module_ &m);
void ExportOpPosterize(py::module_ &m);
void ExportOpAutoContrast(py::module_ &m);
void ExportOpAdjustHue(py::module_ &m);
void ExportOpAdjustSaturation(py::module_ &m);
void ExportOpAdjustSharpness(py::module_ &m);
void ExportOpAdjustContrast(py::module_ &m);
void ExportOpJpegCompressionDistortion(py::module_ &m);
void ExportOpTvResize(py::module_ &m);

class OperatorResource : public nvcvpy::priv::Resource
{
public:
    std::mutex *submissionMutex() const override
    {
        return &m_submissionMutex;
    }

private:
    mutable std::mutex m_submissionMutex;
};

// Native state for a generic cached operator.
// OP: native operator class
// CTOR: ctor signature
template<class OP, class CTOR>
class OperatorState;

template<class OP, class... CTOR_ARGS>
class OperatorState<OP, void(CTOR_ARGS...)> : public OperatorResource
{
public:
    using NativeOperator = OP;

    // This defines a generic cache key class for any cvcuda::OP operator.
    // It allows for reusing cache objects (instead of allocating new ones) only if the hash and key equal match.
    // By default, the hash uses the OP type (in IKey) and OP ctor args, and the equal checks if all args are equal.
    // This may be too restrictive for operators with payloads; those can use
    // a specialized native state and key.
    class Key : public nvcvpy::IKey
    {
    public:
        explicit Key(const CTOR_ARGS &...args)
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
            const auto &that = static_cast<const Key &>(that_);
            return m_args == that.m_args;
        }

        std::tuple<std::decay_t<CTOR_ARGS>...> m_args;
    };

    explicit OperatorState(CTOR_ARGS &&...args)
        : m_op{static_cast<CTOR_ARGS &&>(args)...}
    {
    }

    template<class... AA>
    void submit(AA &&...args)
    {
        m_op(std::forward<AA>(args)...);
    }

private:
    OP m_op;
};

template<class State, class = void>
struct HasCacheItemRank : std::false_type
{
};

template<class State>
struct HasCacheItemRank<State, std::void_t<decltype(State::cacheItemRank(std::declval<const typename State::Key &>()))>>
    : std::true_type
{
};

// The cache owns this metadata while the global pending-release queue owns only
// native operator state. This keeps lookup availability independent of
// asynchronous resource lifetime. The default policy deliberately keeps one
// shared state for operators audited as safe to reuse across ordered
// submissions; operators with mutable in-flight workspace opt into the pool
// below.
template<class State, bool = OperatorCacheUsesCompletionLease<State>::value>
class OperatorCacheItem;

template<class State>
class OperatorCacheItem<State, false> final : public nvcvpy::priv::CacheItem
{
public:
    using Key = typename State::Key;

    template<class... CTOR_ARGS>
    explicit OperatorCacheItem(Key key, CTOR_ARGS &&...args)
        : m_key(std::move(key))
        , m_state(std::make_shared<State>(std::forward<CTOR_ARGS>(args)...))
    {
    }

    const nvcvpy::IKey &key() const override
    {
        return m_key;
    }

    int64_t GetSizeInBytes() const override
    {
        // Operator allocations are owned by the native state rather than the
        // allocation-oriented NVCV wrapper cache.
        return 0;
    }

    const Key &typedKey() const
    {
        return m_key;
    }

    const std::shared_ptr<State> &state() const
    {
        return m_state;
    }

private:
    Key                    m_key;
    std::shared_ptr<State> m_state;
};

template<class State>
class OperatorCacheItem<State, true> final : public nvcvpy::priv::CacheItem
{
public:
    using Key = typename State::Key;

    explicit OperatorCacheItem(Key key)
        : m_key(std::move(key))
    {
    }

    const nvcvpy::IKey &key() const override
    {
        return m_key;
    }

    int64_t GetSizeInBytes() const override
    {
        return 0;
    }

    const Key &typedKey() const
    {
        return m_key;
    }

    template<class... CTOR_ARGS>
    std::shared_ptr<State> acquire(Key requestKey, CTOR_ARGS &&...args)
    {
        std::lock_guard lock(m_mutex);

        size_t selected = m_entries.size();
        for (size_t i = 0; i < m_entries.size(); ++i)
        {
            const Entry &entry = m_entries[i];
            // The pool is the sole owner only after the global pending-release
            // queue has retired every submission that used this state.
            if (entry.state.use_count() != 1 || !(requestKey == entry.key))
            {
                continue;
            }
            if constexpr (HasCacheItemRank<State>::value)
            {
                if (selected != m_entries.size()
                    && State::cacheItemRank(entry.key) <= State::cacheItemRank(m_entries[selected].key))
                {
                    continue;
                }
            }
            selected = i;
            if constexpr (!HasCacheItemRank<State>::value)
            {
                break;
            }
        }

        if (selected != m_entries.size())
        {
            return m_entries[selected].state;
        }

        auto state = std::make_shared<State>(std::forward<CTOR_ARGS>(args)...);
        m_entries.push_back({std::move(requestKey), state});
        return state;
    }

private:
    struct Entry
    {
        Key                    key;
        std::shared_ptr<State> state;
    };

    Key                m_key;
    std::mutex         m_mutex;
    std::vector<Entry> m_entries;
};

template<class State>
bool PreferOperatorCacheItem(const nvcvpy::priv::CacheItem &candidate, const nvcvpy::priv::CacheItem &selected)
{
    using Item = OperatorCacheItem<State>;
    return State::cacheItemRank(static_cast<const Item &>(candidate).typedKey())
         > State::cacheItemRank(static_cast<const Item &>(selected).typedKey());
}

// Creates an operator instance.
// Either gets it from the resource cache or creates one from scratch.
// When creating, it'll be added to the cache.
template<class PyOP, class... CTOR_ARGS>
std::shared_ptr<PyOP> CreateOperatorEx(CTOR_ARGS &&...args)
{
    using Item = OperatorCacheItem<PyOP>;
    using Key  = typename PyOP::Key;

    // Creates a key out of the operator's ctor parameters
    Key key{args...};

    // Operator metadata is independent of pending allocation releases, so
    // avoid their CUDA polling on this hot path. Generic lookup selects the first
    // match without allocating a vector; specialized states may provide a
    // rank function to preserve their best-fit selection policy.
    std::shared_ptr<nvcvpy::priv::CacheItem> cached;
    if constexpr (HasCacheItemRank<PyOP>::value)
    {
        cached = nvcvpy::priv::Cache::Instance().fetchReusableNoPoll(key, &PreferOperatorCacheItem<PyOP>);
    }
    else
    {
        cached = nvcvpy::priv::Cache::Instance().fetchReusableNoPoll(key);
    }

    if (cached)
    {
        // Each nested Key type belongs to exactly one operator State, which
        // makes its matching cache-item type unique. Keep this invariant: it
        // is what makes the release-path static cast safe and avoids RTTI.
        NVCV_ASSERT(typeid(*cached) == typeid(Item));
        auto item = std::static_pointer_cast<Item>(std::move(cached));
        if constexpr (OperatorCacheUsesCompletionLease<PyOP>::value)
        {
            return item->acquire(std::move(key), std::forward<CTOR_ARGS>(args)...);
        }
        else
        {
            return item->state();
        }
    }

    if constexpr (OperatorCacheUsesCompletionLease<PyOP>::value)
    {
        // The metadata key and first pool-entry key share the device lookup.
        // Copying a key preserves that captured device rather than constructing
        // another IKey and calling cudaGetDevice again.
        auto item  = std::make_shared<Item>(key);
        auto state = item->acquire(std::move(key), std::forward<CTOR_ARGS>(args)...);
        nvcvpy::priv::Cache::Instance().add(*item);
        return state;
    }
    else
    {
        // Adopt the lookup key so a miss performs cudaGetDevice only once.
        auto item = std::make_shared<Item>(std::move(key), std::forward<CTOR_ARGS>(args)...);
        nvcvpy::priv::Cache::Instance().add(*item);
        return item->state();
    }
}

template<class OP, class... CTOR_ARGS>
std::shared_ptr<OperatorState<OP, void(CTOR_ARGS...)>> CreateOperator(CTOR_ARGS &&...args)
{
    return CreateOperatorEx<OperatorState<OP, void(CTOR_ARGS...)>>(std::forward<CTOR_ARGS>(args)...);
}

} // namespace cvcudapy

namespace nvcv {
inline size_t ComputeHash(const ImageFormat fmt)
{
    using nvcvpy::util::ComputeHash;
    return ComputeHash(fmt.cvalue());
}
} // namespace nvcv
