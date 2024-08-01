/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#ifndef NVCV_HANDLE_WRAPPER_HPP
#define NVCV_HANDLE_WRAPPER_HPP

#include "detail/CheckError.hpp"
#include "detail/CompilerUtils.h"

#include <cassert>
#include <utility>

namespace nvcv {

template<typename HandleType>
int HandleDecRef(HandleType handle);

template<typename HandleType>
int HandleIncRef(HandleType handle);

template<typename HandleType>
int HandleRefCount(HandleType handle);

template<typename HandleType>
void HandleDestroy(HandleType handle);

template<typename HandleType>
constexpr HandleType NullHandle()
{
    return {};
}

template<typename HandleType>
constexpr bool HandleIsNull(HandleType h)
{
    return h == NullHandle<HandleType>();
}

namespace detail {

template<typename HandleType>
struct SharedHandleOps
{
    static int DecRef(HandleType handle)
    {
        return HandleDecRef(handle);
    }

    static int IncRef(HandleType handle)
    {
        return HandleIncRef(handle);
    }

    static int RefCount(HandleType handle)
    {
        return HandleRefCount(handle);
    }

    static constexpr bool IsNull(HandleType handle)
    {
        return HandleIsNull(handle);
    }

    static constexpr HandleType Null()
    {
        return NullHandle<HandleType>();
    };
};

template<typename HandleType>
struct UniqueHandleOps
{
    static void Destroy(HandleType handle)
    {
        HandleDestroy(handle);
    }

    static constexpr bool IsNull(HandleType handle)
    {
        return HandleIsNull(handle);
    }

    static constexpr HandleType Null()
    {
        return NullHandle<HandleType>();
    };
};

} // namespace detail

/** A handle wrapper that behaves like a unique_ptr.
 *
 * @tparam HandleType   The type of the managed handle
 * @tparam HandleOps    The set of handle operations - can be customized e.g. to add extra tracking
 *                      or suppress object deletion.
 */
template<typename HandleType, typename HandleOps = detail::UniqueHandleOps<HandleType>>
class UniqueHandle
{
public:
    ~UniqueHandle()
    {
        reset();
    }

    /** Constructs a UniqueHandle that from a bare handle.
     *
     * Constructs a UniqueHandle that manages the handles passed in the argument.
     * The ownership of the object is transferred to UniqueHandle and the handle must
     * not be destroyed manually.
     *
     * @param handle    The handle to be managed.
     *                  The handle is passed by an r-value reference and is set to null to
     *                  emphasize the transfer of ownership.
     */
    explicit UniqueHandle(HandleType &&handle)
        : m_handle(handle)
    {
        handle = HandleOps::Null();
    }

    UniqueHandle()                          = default;
    UniqueHandle(const UniqueHandle &other) = delete;

    UniqueHandle(UniqueHandle &&other)
        : m_handle(std::move(other.m_handle))
    {
        other.m_handle = HandleOps::Null();
    }

    /** Moves the handle owned by `other` into this object and releases the old handle.
     */
    UniqueHandle &operator=(UniqueHandle &&other)
    {
        if (&other == this)
            return *this; // avoid self-reset in self-move
        swap(other);
        other.reset();
        return *this;
    }

    /** Swaps the handles managed by `this` and `other`
     */
    void swap(UniqueHandle &other) noexcept
    {
        std::swap(m_handle, other.m_handle);
    }

    /** Replaces the managed handle and destroys the previously owned resource, if the handle was not null.
     *
     * @param handle    The handle to be managed.
     *                  The handle is passed by an r-value reference and is set to null to
     *                  emphasize the transfer of ownership.
     *
     * @remarks Passing a non-empty handle that's already owned by this UniqueHandle is forbidden and will
     *          result in double destruction of the handle.
     */
    void reset(HandleType &&handle = HandleOps::Null())
    {
        assert(HandleOps::IsNull(handle) || handle != m_handle);
        if (*this)
        {
            HandleOps::Destroy(m_handle);
        }
        m_handle = std::move(handle);
        handle   = HandleOps::Null();
    }

    /** Relinquishes the ownership of the handle and returns the formerly managed handle.
     *
     * The function returns the handle and stops managing it. The caller is resposnible
     * for destroying the handle.
     *
     * @return The (formerly) managed handle. The values must not be discarded.
     */
    NVCV_NODISCARD HandleType release() noexcept
    {
        HandleType h = std::move(m_handle);
        m_handle     = HandleOps::Null();
        return h;
    }

    /** Returns the managed handle.
     *
     * @return The managed handle.
     */
    constexpr const HandleType get() const noexcept
    {
        return m_handle;
    }

    constexpr bool empty() const noexcept
    {
        return HandleOps::IsNull(m_handle);
    }

    constexpr explicit operator bool() const noexcept
    {
        return !empty();
    }

    bool operator==(const UniqueHandle &other) const
    {
        return m_handle == other.m_handle;
    }

    bool operator!=(const UniqueHandle &other) const
    {
        return m_handle != other.m_handle;
    }

private:
    HandleType m_handle = HandleOps::Null();
};

/** A handle wrapper that behaves similarly shared_ptr.
 *
 * Copying a SharedHandle increments the reference count.
 * Destroying a SharedHandle decrements the reference count.
 * Swap (and, to some, extent, move) are very simple operations on the handle value.
 *
 * @tparam HandleType   The type of the managed handle
 * @tparam HandleOps    The set of handle operations - can be customized e.g. to add extra operation tracking
 */
template<typename HandleType, typename HandleOps = detail::SharedHandleOps<HandleType>>
class SharedHandle
{
public:
    ~SharedHandle()
    {
        reset();
    }

    SharedHandle() = default;

    /** Manages the handle in a new SharedHandle wrapper.
     *
     * @param handle    The handle to be managed.
     *                  The handle is passed by an r-value reference and is set to null to
     *                  emphasize the transfer of ownership.
     *
     * The reference count on the handle is _not_ incremented.
     */
    explicit SharedHandle(HandleType &&handle) noexcept
        : m_handle(handle)
    {
        handle = HandleOps::Null();
    }

    SharedHandle(SharedHandle &&other) noexcept
        : m_handle(std::move(other.m_handle))
    {
        other.m_handle = HandleOps::Null();
    }

    /** Creates a new shared reference to the handle managed by `other`.
     */
    SharedHandle(const SharedHandle &other)
    {
        *this = other;
    }

    /** Moves the handle owned by `other` into this object and releases the old handle.
     */
    SharedHandle &operator=(SharedHandle &&other)
    {
        if (&other == this)
            return *this; // we must not reset the "other" in case of self-move
        swap(other);
        other.reset();
        return *this;
    }

    /**
     * @brief Copies the handle managed by `other` to `this` and incremenets the reference count.
     *        Decrements the reference count on the previously owned handle.
     *
     * This function performs the following actions (in that order):
     * 1. If the currently managed handle and `other` are equal, the function is a no-op.
     * 2. Increments the reference count on `other`(if not null)
     * 3. Replaces the managed handle by that from `other`.
     * 4. Decrements the reference count on the old handle (if not null).
     */
    SharedHandle &operator=(const SharedHandle &other)
    {
        HandleType new_handle = other.get();
        if (m_handle == new_handle)
            return *this;
        if (!HandleOps::IsNull(new_handle))
            HandleOps::IncRef(new_handle);
        reset(std::move(new_handle));
        return *this;
    }

    /** Swaps the handles managed by `this` and `other`.
     */
    void swap(SharedHandle &other) noexcept
    {
        std::swap(m_handle, other.m_handle);
    }

    /** Replaces the currently managed handle with the one passed in the parameter.
     *
     * This function performs the following actions (in that order):
     * 1. Stores a copy of the "old" handle
     * 2. Replaces the managed handle by that from `other`.
     * 3. Decrements the reference count on the old handle (if not null)
     *
     * @param handle    The handle to be managed.
     *                  The handle is passed by an r-value reference and is set to null to
     *                  emphasize the transfer of ownership.
     *
     * @return The updated reference count of the *old* handle - if it's zero,
     *         the object was destroyed or the handle was already null. If it's >0, the object
     *         still had some live references.
     */
    int reset(HandleType &&handle = HandleOps::Null())
    {
        auto old = m_handle;
        m_handle = std::move(handle);
        handle   = HandleOps::Null();
        if (!HandleOps::IsNull(old))
            return HandleOps::DecRef(old);
        return 0;
    }

    /** Relinquishes the ownership of the handle and returns the formerly managed handle.
     *
     * The function returns the handle and stops managing it. The caller is resposnible
     * for decrementing the reference count on the handle.
     *
     * @return The (formerly) managed handle. The values must not be discarded.
     */
    NVCV_NODISCARD HandleType release() noexcept
    {
        HandleType h = std::move(m_handle);
        m_handle     = HandleOps::Null();
        return h;
    }

    /** Returns the currently managed handle.
     */
    constexpr const HandleType get() const noexcept
    {
        return m_handle;
    }

    int refCount() const noexcept
    {
        HandleType h = m_handle;
        return HandleOps::IsNull(h) ? 0 : HandleOps::RefCount(h);
    }

    constexpr bool empty() const noexcept
    {
        return HandleOps::IsNull(m_handle);
    }

    constexpr explicit operator bool() const noexcept
    {
        return !empty();
    }

    bool operator==(const SharedHandle &other) const
    {
        return m_handle == other.m_handle;
    }

    bool operator!=(const SharedHandle &other) const
    {
        return m_handle != other.m_handle;
    }

private:
    HandleType m_handle = HandleOps::Null();
};

#define NVCV_IMPL_DEC_REF(ObjectKind)                                             \
    template<>                                                                    \
    inline int HandleDecRef<NVCV##ObjectKind##Handle>(NVCV##ObjectKind##Handle h) \
    {                                                                             \
        int ref = -1;                                                             \
        nvcv::detail::CheckThrow(nvcv##ObjectKind##DecRef(h, &ref));              \
        return ref;                                                               \
    }

#define NVCV_IMPL_INC_REF(ObjectKind)                                             \
    template<>                                                                    \
    inline int HandleIncRef<NVCV##ObjectKind##Handle>(NVCV##ObjectKind##Handle h) \
    {                                                                             \
        int ref = -1;                                                             \
        nvcv::detail::CheckThrow(nvcv##ObjectKind##IncRef(h, &ref));              \
        return ref;                                                               \
    }

#define NVCV_IMPL_REF_COUNT(ObjectKind)                                             \
    template<>                                                                      \
    inline int HandleRefCount<NVCV##ObjectKind##Handle>(NVCV##ObjectKind##Handle h) \
    {                                                                               \
        int ref = -1;                                                               \
        nvcv::detail::CheckThrow(nvcv##ObjectKind##RefCount(h, &ref));              \
        return ref;                                                                 \
    }

/** Generates reference-counting handle operations (incRef, decRef, refCount).
 *
 * @param ObjectKind The name of the object, as used in C API (e.g. Image, Tensor)
 *
 * @remarks This macro must be used in nvcv namespace
 */
#define NVCV_IMPL_SHARED_HANDLE(ObjectKind) \
    NVCV_IMPL_DEC_REF(ObjectKind)           \
    NVCV_IMPL_INC_REF(ObjectKind)           \
    NVCV_IMPL_REF_COUNT(ObjectKind)

/** Generates a destroy function wrapper for a handle whose C destroy function returns a status code.
 *
 * @param ObjectKind The name of the object, as used in C API (e.g. Image, Tensor)
 *
 * @remarks This macro must be used in nvcv namespace
 */
#define NVCV_IMPL_DESTROY(ObjectKind)                                               \
    template<>                                                                      \
    inline void HandleDestroy<NVCV##ObjectKind##Handle>(NVCV##ObjectKind##Handle h) \
    {                                                                               \
        nvcv::detail::CheckThrow(nvcv##ObjectKind##Destroy(h));                     \
    }

/** Generates a destroy function wrapper for a handle whose C destroy function doesn't return a value.
 *
 * @param ObjectKind The name of the object, as used in C API (e.g. Image, Tensor)
 *
 * @remarks This macro must be used in nvcv namespace
 */
#define NVCV_IMPL_DESTROY_VOID(ObjectKind)                                          \
    template<>                                                                      \
    inline void HandleDestroy<NVCV##ObjectKind##Handle>(NVCV##ObjectKind##Handle h) \
    {                                                                               \
        nvcv##ObjectKind##Destroy(h);                                               \
    }

} // namespace nvcv

#endif
