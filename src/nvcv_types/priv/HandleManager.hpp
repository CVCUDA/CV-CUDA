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

#ifndef NVCV_PRIV_CORE_HANDLE_MANAGER_HPP
#define NVCV_PRIV_CORE_HANDLE_MANAGER_HPP

#include <util/Algorithm.hpp>
#include <util/Assert.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <utility>

namespace nvcv::priv {

namespace detail {
template<class, class = void>
struct GetHandleType
{
    using type = void *;
};

template<class T>
struct GetHandleType<T, std::void_t<decltype(sizeof(typename T::HandleType))>>
{
    using type = typename T::HandleType;
};
} // namespace detail

template<class T>
using GetHandleType = typename detail::GetHandleType<T>::type;

// We use the 1st LSB of the Resource address to enable special (private) API
// processing, which dependent on the C function the handle is passed to.
// The next 3 bits are used to store the handle generation.
// For all of this  to work, Resource object address must be aligned to 16 bytes.
// Handle:        RRRR.RRRR.RRRR.GGGP
// Resource addr: RRRR.RRRR.RRRR.0000
// R=resource address, G=generation, P=private API bit
static constexpr int kResourceAlignment = 16; // Must be a power of two.

template<typename Interface, typename Storage>
class HandleManager
{
    struct alignas(kResourceAlignment) Resource
    {
        // We allow the resource to be reused up to 8 times,
        // the corresponding handle will have a different value each time.
        // After that, a handle to an object that was already destroyed might
        // refer to a different object.
        uint8_t generation : 3; // must be log2(kResourceAlignment-1)

        Resource *next = nullptr;

        Resource();
        ~Resource();

        template<class T, typename... Args>
        T *constructObject(Args &&...args)
        {
            static_assert(std::is_base_of_v<Interface, T>);

            static_assert(sizeof(Storage) >= sizeof(T));
            static_assert(alignof(Storage) % alignof(T) == 0);

            NVCV_ASSERT(!this->live());
            T *obj   = new (m_storage) T{std::forward<Args>(args)...};
            m_ptrObj = obj;
            this->generation++;

            NVCV_ASSERT(this->live());

            return obj;
        }

        void destroyObject();

        Interface *obj() const
        {
            return m_ptrObj;
        }

        bool live() const
        {
            return m_ptrObj != nullptr;
        }

    private:
        Interface *m_ptrObj = nullptr;
        alignas(Storage) std::byte m_storage[sizeof(Storage)];
    };

public:
    using HandleType = GetHandleType<Interface>;

    HandleManager(const char *name);
    ~HandleManager();

    template<class T, typename... Args>
    std::pair<HandleType, T *> create(Args &&...args)
    {
        Resource *res = doFetchFreeResource();
        T        *obj = res->template constructObject<T>(std::forward<Args>(args)...);
        return std::make_pair(doGetHandleFromResource(res), obj);
    }

    // true if handle is destroyed, false if handle is invalid (or already removed)
    bool destroy(HandleType handle);

    Interface *validate(HandleType handle) const;

    void setFixedSize(int32_t maxSize);
    void setDynamicSize(int32_t minSize = 0);

    void clear();

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl;

    void       doAllocate(size_t count);
    void       doGrow();
    Resource  *doFetchFreeResource();
    void       doReturnResource(Resource *r);
    uint8_t    doGetHandleGeneration(HandleType handle) const;
    HandleType doGetHandleFromResource(Resource *r) const;
    Resource  *doGetResourceFromHandle(HandleType handle) const;
    bool       isManagedResource(Resource *r) const;
};

inline bool MustProvideHiddenFunctionality(void *h)
{
    // Handle LSB tells us whether C public function that receives it
    // must provide special/hidden functionality.
    return ((uintptr_t)h) & 1;
}

template<class... AA>
struct alignas(util::Max(alignof(AA)...)) CompatibleStorage
{
    std::byte storage[util::Max(sizeof(AA)...)];
};

} // namespace nvcv::priv

#endif // NVCV_PRIV_CORE_HANDLE_MANAGER_HPP
