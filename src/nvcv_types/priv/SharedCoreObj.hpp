/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef NVCV_CORE_PRIV_SHARED_CORE_OBJ_HPP
#define NVCV_CORE_PRIV_SHARED_CORE_OBJ_HPP

#include "ICoreObject.hpp"

#include <nvcv/detail/CompilerUtils.h>

#include <cassert>
#include <type_traits>

namespace nvcv::priv {

template<typename CoreObj>
class SharedCoreObj
{
public:
    static_assert(std::is_base_of_v<ICoreObject, CoreObj>,
                  "The CoreObj type must inherit from the ICoreObject interface.");

    SharedCoreObj() = default;

    SharedCoreObj(std::nullptr_t) {}

    static SharedCoreObj FromHandle(typename CoreObj::HandleType handle, bool incRef)
    {
        if (handle)
        {
            if (incRef)
                CoreObjectIncRef(handle);
            auto *obj = ToStaticPtr<CoreObj>(handle);
            assert(obj);
            assert(obj->handle() == handle);
            return SharedCoreObj(std::move(obj));
        }
        else
        {
            return {};
        }
    }

    static SharedCoreObj FromPointer(CoreObj *obj, bool incRef)
    {
        if (obj && incRef)
        {
            if (auto h = obj->handle())
                CoreObjectIncRef(h);
            else
                throw std::logic_error("Cannot use incRef on an object without a handle");
        }
        return SharedCoreObj(std::move(obj));
    }

    explicit SharedCoreObj(CoreObj *&&obj)
        : m_obj(obj)
    {
        obj = nullptr;
    }

    SharedCoreObj(const SharedCoreObj &obj)
    {
        *this = obj;
    }

    SharedCoreObj(CoreObj &&obj) = delete;

    // Temporary workaround to avoid too many changes in the code
    SharedCoreObj(CoreObj &obj)
        : m_obj(&obj)
    {
        if (auto h = obj.handle())
            CoreObjectIncRef(h);
    }

    template<typename U, std::enable_if_t<std::is_convertible_v<U *, CoreObj *>, int> = 0>
    SharedCoreObj(const SharedCoreObj<U> &obj)
    {
        *this = obj;
    }

    SharedCoreObj(SharedCoreObj &&obj)
    {
        *this = std::move(obj);
    }

    ~SharedCoreObj()
    {
        reset(nullptr);
    }

    int reset(CoreObj *&&obj)
    {
        int ret = 0;
        if (m_obj)
            if (auto h = m_obj->handle())
                ret = CoreObjectDecRef(h);

        m_obj = obj;
        obj   = nullptr;
        return ret;
    }

    NVCV_NODISCARD CoreObj *release()
    {
        CoreObj *ret = m_obj;
        m_obj        = nullptr;
        return ret;
    }

    SharedCoreObj &operator=(const SharedCoreObj &obj)
    {
        if (obj && obj->handle())
            CoreObjectIncRef(obj->handle());
        reset(obj.get());
        return *this;
    }

    SharedCoreObj &operator=(SharedCoreObj &&obj)
    {
        reset(obj.release());
        return *this;
    }

    template<typename T>
    SharedCoreObj &operator=(const SharedCoreObj<T> &obj)
    {
        if (obj && obj->handle())
            CoreObjectIncRef(obj->handle());
        reset(obj.get());
        return *this;
    }

    template<typename T>
    SharedCoreObj &operator=(SharedCoreObj<T> &&obj)
    {
        reset(obj.release());
        return *this;
    }

    constexpr CoreObj *get() const &noexcept
    {
        return m_obj;
    }

    constexpr CoreObj *operator->() const noexcept
    {
        assert(m_obj && "Accessing a member via a null pointer.");
        return m_obj;
    }

    constexpr CoreObj &operator*() const &noexcept
    {
        assert(m_obj && "Dereferencing a null pointer.");
        return *m_obj;
    }

    template<typename U>
    constexpr bool operator==(const SharedCoreObj<U> &other) const noexcept
    {
        return get() == other.get();
    }

    template<typename U>
    constexpr bool operator!=(const SharedCoreObj<U> &other) const noexcept
    {
        return !(*this == other);
    }

    template<typename U>
    constexpr bool operator<(const SharedCoreObj<U> &other) const noexcept
    {
        return static_cast<void *>(get()) < static_cast<void *>(other.get());
    }

    constexpr operator bool() const noexcept
    {
        return m_obj != nullptr;
    }

private:
    CoreObj *m_obj = nullptr;
};

template<typename T>
constexpr bool operator==(std::nullptr_t, const SharedCoreObj<T> &x)
{
    return !x;
}

template<typename T>
constexpr bool operator!=(std::nullptr_t, const SharedCoreObj<T> &x)
{
    return x;
}

template<typename T, typename HandleType>
inline SharedCoreObj<T> ToSharedObj(HandleType h)
{
    return SharedCoreObj<T>::FromPointer(ToStaticPtr<T>(h), true);
}

} // namespace nvcv::priv

#endif // NVCV_CORE_PRIV_SHARED_CORE_OBJ_HPP
