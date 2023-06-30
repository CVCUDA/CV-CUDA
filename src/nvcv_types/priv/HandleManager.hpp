/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <nvcv/Fwd.h>
#include <nvcv/alloc/Fwd.h>
#include <util/Algorithm.hpp>
#include <util/Assert.h>

#include <atomic>
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

// The 4 least significant bits are used to store the handle generation.
// For this to work, Resource object address must be aligned to 16 bytes.
// Handle:        RRRR.RRRR.RRRR.GGGG
// Resource addr: RRRR.RRRR.RRRR.0000
// R=resource address, G=generation
static constexpr int kResourceAlignment = 16; // Must be a power of two.

/** A type trait that defines storage for objects implementing given interface
 *
 * This struct must define a ::type that is sufficiently large and aligned to contain
 * any type that the user might want to store inside a HandleManager<Interface>.
 */
template<typename Interface>
struct ResourceStorage;

template<typename Interface>
class HandleManager
{
    struct ResourceBase
    {
        // We allow the resource to be reused up to 16 times,
        // the corresponding handle will have a different value each time.
        // After that, a handle to an object that was already destroyed might
        // refer to a different object.
        uint8_t generation : 4; // must be log2(kResourceAlignment-1)

        ResourceBase *next = nullptr;

        ResourceBase();

        template<class T, typename... Args>
        T *constructObject(Args &&...args)
        {
            static_assert(std::is_base_of_v<Interface, T>);

            using Storage = typename ResourceStorage<Interface>::type;
            static_assert(sizeof(Storage) >= sizeof(T));
            static_assert(alignof(Storage) % alignof(T) == 0);

            NVCV_ASSERT(!this->live());
            T *obj         = new (getStorage()) T{std::forward<Args>(args)...};
            this->m_ptrObj = obj;
            this->generation++;

            NVCV_ASSERT(this->live());

            return obj;
        }

        void destroyObject();

        int decRef()
        {
            return --m_refCount;
        }

        int incRef()
        {
            return ++m_refCount;
        }

        int refCount()
        {
            return m_refCount;
        }

        Interface *obj() const
        {
            return m_ptrObj;
        }

        bool live() const
        {
            return m_ptrObj != nullptr;
        }

    protected:
        void *getStorage();

        ~ResourceBase();
        Interface      *m_ptrObj = nullptr;
        std::atomic_int m_refCount{0};
    };

public:
    using HandleType = GetHandleType<Interface>;

    HandleManager(const char *name);
    ~HandleManager();

    template<class T, typename... Args>
    std::pair<HandleType, T *> create(Args &&...args)
    {
        ResourceBase *res = doFetchFreeResource();
        try
        {
            T *obj = res->template constructObject<T>(std::forward<Args>(args)...);
            return std::make_pair(doGetHandleFromResource(res), obj); // noexcept
        }
        catch (...)
        {
            // If object ctor threw an exception, we must return
            // the resource we would have used for it.
            res->decRef();
            doReturnResource(res);
            throw;
        }
    }

    /** Decrements the reference count of the object pointed to by the handle and destroys
     *  it if no longer referenced
     *
     * @return The remaining reference count if the handle is valid, 0 if the object is destroyed.
     */
    int decRef(HandleType handle);

    /** Increments the reference count of the object pointed to by the handle.
     *
     * @return The new reference count.
     */
    int incRef(HandleType handle);

    /** Returns the current reference count of the object pointed to by the handle;
     */
    int refCount(HandleType handle);

    Interface *validate(HandleType handle) const;

    void setFixedSize(int32_t maxSize);
    void setDynamicSize(int32_t minSize = 0);

    void clear();

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl;

    void doAllocate(size_t count);
    void doGrow();

    ResourceBase *getValidResource(HandleType handle) const;

    ResourceBase *doFetchFreeResource();
    void          doReturnResource(ResourceBase *r);
    uint8_t       doGetHandleGeneration(HandleType handle) const noexcept;
    HandleType    doGetHandleFromResource(ResourceBase *r) const noexcept;
    ResourceBase *doGetResourceFromHandle(HandleType handle) const noexcept;
    bool          isManagedResource(ResourceBase *r) const;
};

template<class... AA>
struct alignas(util::Max(alignof(AA)...)) CompatibleStorage
{
    std::byte storage[util::Max(sizeof(AA)...)];
};

} // namespace nvcv::priv

#endif // NVCV_PRIV_CORE_HANDLE_MANAGER_HPP
