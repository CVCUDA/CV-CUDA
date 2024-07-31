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

    /**
     * @brief A default constructor that constructs an empty `CoreResource`.
     */
    CoreResource() = default;

    /**
    * @brief A constructor that constructs an empty `CoreResource` from a nullptr.
    */
    CoreResource(std::nullptr_t) {}

    /** Wraps and assumes ownership of a handle.
     *
     * This functions stores the resource handle in the wrapper object.
     * The handle passed in the argument is reset to a null handle value to prevent
     * inadvertent usage by the caller after the ownership has been transferred to the wrapper.
     *
     * @param handle The handle to the resource.
     */
    explicit CoreResource(HandleType &&handle)
        : Base(std::move(handle))
    {
    }

    /**
     * @brief A factory method to create a `CoreResource` from a handle.
     *
     * @param handle The handle to the resource.
     * @param incRef Indicates whether to increment the reference count of the handle.
     * @return A `CoreResource` wrapping the provided handle.
     */
    static Actual FromHandle(HandleType handle, bool incRef)
    {
        if (incRef && handle)
            HandleIncRef(handle);
        return Actual(std::move(handle));
    }

    /**
     * @brief A copy constructor that creates a `CoreResource` from another instance.
     *
     * @param other The other instance to copy from.
     */
    CoreResource(const Actual &other)
        : Base(other)
    {
    }

    /**
     * @brief A move constructor that transfers ownership from another instance.
     *
     * @param other The other instance to move from.
     */
    CoreResource(Actual &&other)
        : Base(std::move(other))
    {
    }

    /**
     * @brief A copy assignment operator that copies from another instance.
     *
     * @param other The other instance to copy from.
     * @return This instance after the copy.
     */
    CoreResource &operator=(const Actual &other)
    {
        Base::operator=(other);
        return *this;
    }

    /**
     * @brief A move assignment operator that moves from another instance.
     *
     * @param other The other instance to move from.
     * @return This instance after the move.
     */
    CoreResource &operator=(Actual &&other)
    {
        Base::operator=(std::move(other));
        return *this;
    }

    /**
     * @brief Returns the handle to the resource.
     *
     * @return The handle to the resource.
     */
    const HandleType handle() const noexcept
    {
        return this->get();
    }

    /**
     * @brief Checks if this instance is equal to another.
     *
     * @param other The other instance to compare with.
     * @return true if they are equal, false otherwise.
     */
    bool operator==(const Actual &other) const
    {
        return handle() == other.handle();
    }

    /**
     * @brief Checks if this instance is not equal to another.
     *
     * @param other The other instance to compare with.
     * @return true if they are not equal, false otherwise.
     */
    bool operator!=(const Actual &other) const
    {
        return handle() != other.handle();
    }

    using Base::empty;
    using Base::refCount;
    using Base::release;
    using Base::reset;
    using Base::operator bool;

    /**
     * @brief Casts this instance to a derived type.
     *
     * @tparam Derived The derived type to cast to.
     * @return The instance cast to the derived type.
     */
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
    /**
     * @brief Type alias for the handle type of the resource.
     */
    using HandleType = typename Resource::HandleType;

    /**
     * @brief A constructor that creates a `NonOwningResource` from a resource handle.
     *
     * @param handle The handle to the resource.
     */
    NonOwningResource(HandleType handle)
        : m_resource(std::move(handle))
    {
    }

    /**
     * @brief The copy constructor is deleted to prevent copying.
     */
    NonOwningResource(const NonOwningResource &) = delete;

    /**
    * @brief The move constructor is defaulted.
    */
    NonOwningResource(NonOwningResource &&) = default;

    /**
     * @brief The copy assignment operator is deleted to prevent copying.
     */
    NonOwningResource &operator=(const NonOwningResource &) = delete;

    /**
    * @brief The move assignment operator is defaulted.
    */
    NonOwningResource &operator=(NonOwningResource &&) = default;

    /**
     * @brief Returns the handle to the resource.
     *
     * @return The handle to the resource.
     */
    const HandleType handle() const
    {
        return m_resource.handle();
    }

    /**
     * @brief The destructor releases the handle to the resource.
     */
    ~NonOwningResource()
    {
        (void)m_resource.release();
    }

    /**
     * @brief Conversion operator to the underlying resource type.
     *
     * @return A const reference to the underlying resource.
     */
    operator const Resource &() const &
    {
        return m_resource;
    }

private:
    Resource m_resource;
};

} // namespace nvcv

#endif // NVCV_CORE_RESOURCE_HPP
