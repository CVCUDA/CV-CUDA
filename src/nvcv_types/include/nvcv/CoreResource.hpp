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
#ifndef NVCV_CORE_RESOURCE_HPP
#define NVCV_CORE_RESOURCE_HPP

#include "HandleWrapper.hpp"

namespace nvcv {

/** CRTP base for resources based on reference-counting handles
 *
 * This class adds constructors and assignment with the `Actual` type, so that Actual doesn't need to
 * tediously reimplement them, but can simply reexpose them with `using`.
 *
 * It also exposes the `handle` with a `get` and re-exposes `reset` and `release` functions.
 *
 * @tparam Handle   The handle type, e.g. NVCVImageHandle
 * @tparam Actual   The actual class, e.g. Image
 */
template<typename Handle, typename Actual>
class CoreResource : private SharedHandle<Handle>
{
public:
    using HandleType = Handle;
    using Base       = SharedHandle<Handle>;

    CoreResource() = default;

    CoreResource(std::nullptr_t) {}

    /** Wraps and assumes ownership of a handle.
     *
     * This functions stores the resource handle in the wrapper object.
     * The handle passed in the argument is reset to a null handle value to prevent
     * inadvertent usage by the caller after the ownership has been transfered to the wrapper.
     */
    explicit CoreResource(HandleType &&handle)
        : Base(std::move(handle))
    {
    }

    static Actual FromHandle(HandleType handle, bool incRef)
    {
        if (incRef && handle)
            HandleIncRef(handle);
        return Actual(std::move(handle));
    }

    CoreResource(const Actual &other)
        : Base(other)
    {
    }

    CoreResource(Actual &&other)
        : Base(std::move(other))
    {
    }

    CoreResource &operator=(const Actual &other)
    {
        Base::operator=(other);
        return *this;
    }

    CoreResource &operator=(Actual &&other)
    {
        Base::operator=(std::move(other));
        return *this;
    }

    const HandleType handle() const noexcept
    {
        return this->get();
    }

    bool operator==(const Actual &other) const
    {
        return handle() == other.handle();
    }

    bool operator!=(const Actual &other) const
    {
        return handle() != other.handle();
    }

    using Base::empty;
    using Base::refCount;
    using Base::release;
    using Base::reset;
    using Base::operator bool;

    template<typename Derived>
    Derived cast() const
    {
        static_assert(std::is_base_of<Actual, Derived>::value, "Requested a cast to a type not derived from this one.");
        if (*this && Derived::IsCompatibleKind(This().type()))
        {
            return Derived::FromHandle(handle(), true);
        }
        else
        {
            return {};
        }
    }

protected:
    ~CoreResource() = default;

private:
    Actual &This()
    {
        return *static_cast<Actual *>(this);
    }

    const Actual &This() const
    {
        return *static_cast<const Actual *>(this);
    }
};

#define NVCV_IMPLEMENT_SHARED_RESOURCE(ClassName, BaseClassName) \
    using BaseClassName::BaseClassName;                          \
    using BaseClassName::operator=;                              \
    ClassName(const ClassName &other)                            \
        : BaseClassName(other)                                   \
    {                                                            \
    }                                                            \
    ClassName(ClassName &&other)                                 \
        : BaseClassName(std::move(other))                        \
    {                                                            \
    }                                                            \
    ClassName &operator=(const ClassName &other)                 \
    {                                                            \
        BaseClassName::operator=(other);                         \
        return *this;                                            \
    }                                                            \
    ClassName &operator=(ClassName &&other)                      \
    {                                                            \
        BaseClassName::operator=(std::move(other));              \
        return *this;                                            \
    }

/** A non-owning wrapper around a handle which can be trivially converted to a reference-counting wrapper
 *
 * Motivation:
 * When implementing functions that take handles as arguments, but do not take ownership of the object passed by
 * handle, it's beneficial to have some way of wrapping the handle into a CoreResource but avoid the calls to
 * incRef/decRef. This class bypasses these calls in construction/destruction. Internally this object store the
 * actual resource and can return a reference to it, so it can be seamlessly used with C++ APIs that operate on
 * the resource class reference. The original resource's interface is not (fully) reexposed.
 *
 * Example:
 *
 * ```
 * void bar(const Image &img)  // takes a reference to the Image shared handle wrapper
 * {
 *     doStuff(img);
 * }
 *
 * void foo(NVCVImageHandle handle)
 * {
 *     NonOwningResource<Image> img(handle);    // no incRef on construction
 *     bar(img);                                // no incRef/decRef when converting to Image
 * }                                            // no decRef on destruction
 * ```
 */
template<typename Resource>
class NonOwningResource
{
public:
    using HandleType = typename Resource::HandleType;

    NonOwningResource(HandleType handle)
        : m_resource(std::move(handle))
    {
    }

    NonOwningResource(const NonOwningResource &) = delete;
    NonOwningResource(NonOwningResource &&)      = default;

    NonOwningResource &operator=(const NonOwningResource &) = delete;
    NonOwningResource &operator=(NonOwningResource &&)      = default;

    const HandleType handle() const
    {
        return m_resource.handle();
    }

    ~NonOwningResource()
    {
        (void)m_resource.release();
    }

    operator const Resource &() const &
    {
        return m_resource;
    }

private:
    Resource m_resource;
};

} // namespace nvcv

#endif // NVCV_CORE_RESOURCE_HPP
