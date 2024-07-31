/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef NVCV_CASTSIMPL_HPP
#define NVCV_CASTSIMPL_HPP

#ifndef NVCV_CASTS_HPP
#    error "You must not include this header directly"
#endif

#include "../Status.h"
#include "CheckError.hpp"

#include <cassert>
#include <stdexcept>

namespace nvcv { namespace detail {

// Refers to an external handle. It doesn't own it.
// Must be moved to some "detail" namespace at some point,
// users must not use it directly.
template<class IFACE>
class WrapHandle : public IFACE
{
public:
    using HandleType = typename IFACE::HandleType;

    explicit WrapHandle(HandleType handle)
        : m_handle(handle)
    {
        assert(handle != nullptr);
    }

    WrapHandle(const WrapHandle &that)
        : m_handle(that.m_handle)
    {
    }

private:
    HandleType m_handle;

    HandleType doGetHandle() const final override
    {
        return m_handle;
    }
};

template<class IFACE, class H>
void SetObjectAssociation(NVCVStatus (*setUserPointer)(H, void *), IFACE *obj, H handle)
{
    static_assert(std::is_same<typename IFACE::HandleType, H>::value, "handle type must match interface's");

    assert(((uintptr_t)handle & 1) == 0);
    detail::CheckThrow((*setUserPointer)((H)(((uintptr_t)handle) | 1), obj));
}

template<class IFACE, class H>
IFACE *CastImpl(NVCVStatus (*getUserPointer)(H, void **), NVCVStatus (*setUserPointer)(H, void *), H handle)
{
    static_assert(std::is_same<typename IFACE::HandleType, H>::value, "handle type must matchinterface's");
    assert(getUserPointer != nullptr);
    assert(setUserPointer != nullptr);

    if (handle != nullptr)
    {
        H privHandle = reinterpret_cast<H>(((uintptr_t)handle) | 1);

        uintptr_t allocSize = sizeof(detail::WrapHandle<IFACE>);
        void     *ptr       = &allocSize;

        NVCVStatus status = (*getUserPointer)(privHandle, &ptr);
        if (status == NVCV_SUCCESS)
        {
            if (ptr != nullptr)
            {
                if (*static_cast<int *>(ptr) == 0)
                {
                    auto *wrap = new (ptr) detail::WrapHandle<IFACE>{handle};

                    status = (*setUserPointer)(privHandle, wrap);
                    assert(status == NVCV_SUCCESS);

                    assert(*static_cast<int *>(ptr) != 0);
                    ptr = wrap;
                }

                return static_cast<IFACE *>(ptr);
            }
        }
    }
    return nullptr;
}

template<class T>
struct StaticCast;

template<class T>
struct StaticCast<T *>
{
    static T *cast(const typename T::HandleType h)
    {
        return static_cast<T *>(T::cast(h));
    }
};

template<class T>
struct StaticCast
{
    static T &cast(const typename T::HandleType h)
    {
        T *p = detail::StaticCast<T *>::cast(h);
        assert(p != nullptr);
        return *p;
    }
};

template<class T>
struct DynamicCast;

template<class T>
struct DynamicCast<T *>
{
    static T *cast(const typename T::HandleType h)
    {
        auto *base = StaticCast<typename T::BaseInterface *>::cast(h);
        return dynamic_cast<T *>(base);
    }
};

template<class T>
struct DynamicCast
{
    static T &cast(const typename T::HandleType h)
    {
        if (auto *base = StaticCast<typename T::BaseInterface *>::cast(h))
        {
            return dynamic_cast<T &>(*base);
        }
        else
        {
            throw std::bad_cast();
        }
    }
};
}} // namespace nvcv::detail

#endif // NVCV_CASTSIMPL_HPP
