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

#ifndef NVCV_CORE_PRIV_ICOREOBJECT_HPP
#define NVCV_CORE_PRIV_ICOREOBJECT_HPP

#include "Exception.hpp"
#include "HandleManager.hpp"
#include "IContext.hpp"
#include "Version.hpp"

#include <memory>
#include <type_traits>

// Here we define the base classes for all objects that can be created/destroyed
// by the user, the so-called "core objects".
//
// IMPORTANT: their dynamic definitions must NEVER change, as we must keep backward ABI compatibility
// in case users are mixing objects created by different NVCV versions. This is
// valid even between different major versions.

// Interface classes that inherit from ICoreObject must also obey some rules:
// 1. Once the interface is released to public, its dynamic definition
//    must never change if major versions are kept the same. To change them, major
//    NVCV version must be bumped.
// 2. If new virtual methods need to be added to an interface, a new interface that
//    inherits from the previous one must be created. Code that need the new interface
//    must do down casts.

namespace nvcv::priv {

class alignas(kResourceAlignment) ICoreObject
{
public:
    // Disable copy/move to avoid slicing.
    ICoreObject(const ICoreObject &) = delete;

    virtual ~ICoreObject() = default;

    virtual Version version() const = 0;

    virtual void  setUserPointer(void *ptr) = 0;
    virtual void *userPointer() const       = 0;

    virtual void setCXXObject(void *ptr)        = 0;
    virtual void getCXXObject(void **ptr) const = 0;

protected:
    ICoreObject() = default;
};

template<class HANDLE>
class IHandleHolder
{
public:
    using HandleType = HANDLE;

    virtual void       setHandle(HandleType h) = 0;
    virtual HandleType handle()                = 0;
};

// Base class for all core objects that exposes a handle with a particular type.
// Along with the ToPtr and ToRef methods below, it provides facilities to convert
// between external C handles to the actual internal object instance they refer to.
template<class I, class HANDLE>
class ICoreObjectHandle
    : public ICoreObject
    , public IHandleHolder<HANDLE>
{
public:
    using InterfaceType = I;
};

template<class Interface>
class CoreObjectBase : public Interface
{
public:
    using HandleType = typename Interface::HandleType;

    CoreObjectBase()
        : m_cxxPtr(&m_wrapHandleStorage)
    {
        // Set initial bytes of storage to 0 so that
        // we can tell whether there's an object constructed in
        // it or not. The first sizeof(void *) bytes correspond
        // to the vtbl, which is definitely not 0.
        memset(&m_wrapHandleStorage, 0, sizeof(void *));
    }

    void setHandle(HandleType h) final
    {
        m_handle = h;
    }

    HandleType handle() final
    {
        return m_handle;
    }

    Version version() const final
    {
        return CURRENT_VERSION;
    }

    void setUserPointer(void *ptr) final
    {
        m_userPtr = ptr;
    }

    void *userPointer() const final
    {
        return m_userPtr;
    }

private:
    HandleType m_handle  = {};
    void      *m_userPtr = nullptr;

    // 512 bytes should be enough for all types we need.
    // This value must never decrease, or else it'll break
    // ABI compatibility.
    using WrapHandleStorage = std::aligned_storage_t<512>;
    WrapHandleStorage m_wrapHandleStorage;

    void *m_cxxPtr = nullptr;

    void setCXXObject(void *ptr) final
    {
        m_cxxPtr = ptr;
    }

    void getCXXObject(void **ptr) const final
    {
        NVCV_ASSERT(ptr != nullptr);

        // Already pointing to the corresponding public C++ object?
        // PS: m_cxxPtr initially points to m_wrapHandleStorage,
        // which is initialized to zeros at ctor. This makes it easy
        // for us to detect whether it was initialized or not.
        NVCV_ASSERT(m_cxxPtr != nullptr);
        if (*static_cast<uintptr_t *>(m_cxxPtr) != 0)
        {
            // return it!
            *ptr = m_cxxPtr;
        }
        else
        {
            // Caller will allocate a HandleWrapper, we need to provide space.
            // '*ptr' points to a value with the required size (C++ WrapHandle object).
            uintptr_t reqsize = *static_cast<uintptr_t *>(*ptr);
            // Do we have enough space?
            if (reqsize <= sizeof(WrapHandleStorage))
            {
                // An uninitialized c++ obj pointer always point to the wrap handle storage.
                NVCV_ASSERT(m_cxxPtr == &m_wrapHandleStorage);

                // Return the wrap handle storage.
                *ptr = m_cxxPtr;
            }
            else
            {
                // Oops, version mismatch. We can't risk a buffer overrun.
                throw Exception(NVCV_ERROR_INTERNAL)
                    << "Not enough space in internal wrap object storage for allocating " << reqsize
                    << " bytes (limit=" << sizeof(WrapHandleStorage) << ")"
                    << ", NVCV C++ API version not compatible with the C ABI version " << this->version()
                    << " being used.";
            }
        }
    }
};

template<class HandleType>
class CoreObjManager;

template<class T, class... ARGS>
typename T::HandleType CreateCoreObject(ARGS &&...args)
{
    using H   = typename T::HandleType;
    auto &mgr = GlobalContext().manager<H>();

    auto [h, obj] = mgr.template create<T>(std::forward<ARGS>(args)...);
    obj->setHandle(h);

    return h;
}

template<class HandleType>
void DestroyCoreObject(HandleType handle)
{
    auto &mgr = GlobalContext().manager<HandleType>();

    mgr.destroy(handle);
}

template<class, class = void>
constexpr bool HasObjManager = false;

template<class T>
constexpr bool HasObjManager<T, std::void_t<decltype(sizeof(CoreObjManager<T>))>> = true;

template<class HandleType>
inline ICoreObject *ToCoreObjectPtr(HandleType h)
{
    ICoreObject *core;

    if constexpr (HasObjManager<HandleType>)
    {
        auto &mgr = GlobalContext().manager<HandleType>();
        core      = mgr.validate(h);
    }
    else
    {
        // First cast to the core interface, this must always succeed.
        core = reinterpret_cast<ICoreObject *>(h);
    }

    if (core != nullptr)
    {
        // If major version are the same,
        if (core->version().major() == CURRENT_VERSION.major())
        {
            return core;
        }
        else
        {
            throw Exception(NVCV_ERROR_NOT_COMPATIBLE)
                << "Object version " << core->version() << " not compatible with NVCV version " << CURRENT_VERSION;
        }
    }
    else
    {
        return nullptr;
    }
}

template<class T>
T *ToStaticPtr(typename T::HandleType h)
{
    return static_cast<T *>(ToCoreObjectPtr(h));
}

template<class T>
T &ToStaticRef(typename T::HandleType h)
{
    T *child = ToStaticPtr<T>(h);

    if (child == nullptr)
    {
        throw Exception(NVCV_ERROR_INVALID_ARGUMENT, "Handle was already destroyed");
    }

    return *child;
}

template<class T>
T *ToDynamicPtr(typename T::HandleType h)
{
    return dynamic_cast<T *>(ToCoreObjectPtr(h));
}

template<class T>
T &ToDynamicRef(typename T::HandleType h)
{
    if (T *child = ToDynamicPtr<T>(h))
    {
        return *child;
    }
    else
    {
        throw Exception(NVCV_ERROR_NOT_COMPATIBLE,
                        "Handle doesn't correspond to the requested object or was already destroyed.");
    }
}

} // namespace nvcv::priv

#endif // NVCV_CORE_PRIV_ICOREOBJECT_HPP
