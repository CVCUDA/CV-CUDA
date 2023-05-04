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

    HandleType handle() const noexcept
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

protected:
    ~CoreResource() = default;
};

} // namespace nvcv

#endif // NVCV_CORE_RESOURCE_HPP
