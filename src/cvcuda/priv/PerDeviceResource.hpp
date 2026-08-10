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

#ifndef CVCUDA_PRIV_PER_DEVICE_RESOURCE_HPP
#define CVCUDA_PRIV_PER_DEVICE_RESOURCE_HPP

#include <cuda_runtime.h>
#include <nvcv/Exception.hpp>

#include <functional>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>

namespace cvcuda::priv {

// Lazily creates and caches a resource T per CUDA device.  On the first call
// from a new device the factory runs (with that device already current).
// Subsequent calls from the same device return the cached instance.
//
// Thread-safe: concurrent get() calls from different threads are serialized
// via a shared_mutex (readers) / unique_mutex (writers) pair.
//
// Multi-GPU contract: legacy operators are single-device by design -- they
// allocate GPU memory in their constructor and free it in their destructor,
// with no device-switching logic.  PerDeviceResource is the layer that
// provides multi-GPU support: it maintains one operator instance per device
// and ensures each instance is created (and destroyed) with the correct
// CUDA device active.  Callers simply use get() and always receive the
// instance that belongs to the current device.
template<typename T>
class PerDeviceResource
{
public:
    using Factory = std::function<std::unique_ptr<T>(int deviceId)>;

    explicit PerDeviceResource(Factory factory)
        : m_factory(std::move(factory))
    {
    }

    ~PerDeviceResource()
    {
        int savedDevice = -1;
        cudaGetDevice(&savedDevice);

        // Each resource was allocated on a specific device; set that device
        // before destroying it, then restore the caller's device context.
        for (auto &[dev, ptr] : m_resources)
        {
            cudaSetDevice(dev);
            ptr.reset();
        }
        if (savedDevice >= 0)
        {
            cudaSetDevice(savedDevice);
        }
    }

    PerDeviceResource(const PerDeviceResource &)            = delete;
    PerDeviceResource &operator=(const PerDeviceResource &) = delete;

    T &get()
    {
        int dev;
        {
            cudaError_t err = cudaGetDevice(&dev);
            if (err != cudaSuccess)
                throw nvcv::Exception(nvcv::Status::ERROR_INTERNAL, "cudaGetDevice failed");
        }

        {
            std::shared_lock lock(m_mutex);
            auto             it = m_resources.find(dev);
            if (it != m_resources.end())
                return *it->second;
        }

        std::unique_lock lock(m_mutex);
        auto [it, _] = m_resources.try_emplace(dev, nullptr);
        if (!it->second)
            it->second = m_factory(dev);
        return *it->second;
    }

private:
    Factory                                     m_factory;
    std::unordered_map<int, std::unique_ptr<T>> m_resources;
    mutable std::shared_mutex                   m_mutex;
};

} // namespace cvcuda::priv

#endif // CVCUDA_PRIV_PER_DEVICE_RESOURCE_HPP
