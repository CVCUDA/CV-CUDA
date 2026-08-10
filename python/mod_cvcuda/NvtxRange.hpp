/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef CVCUDA_PYTHON_NVTXRANGE_HPP
#define CVCUDA_PYTHON_NVTXRANGE_HPP

#include <nvtx3/nvToolsExt.h>

namespace cvcudapy {

// RAII push/pop of an NVTX range on the calling host thread: pushed on
// construction, popped on destruction, so the range spans the enclosing scope.
// Mirrors src/cvcuda/priv/Nvtx.hpp on the C++ core side; the Python module
// cannot include that private core header, so the helper is duplicated here.
class NvtxRange final
{
public:
    explicit NvtxRange(const char *name) noexcept
    {
        nvtxRangePushA(name);
    }

    ~NvtxRange() noexcept
    {
        nvtxRangePop();
    }

    NvtxRange(const NvtxRange &)            = delete;
    NvtxRange(NvtxRange &&)                 = delete;
    NvtxRange &operator=(const NvtxRange &) = delete;
    NvtxRange &operator=(NvtxRange &&)      = delete;
};

// Wrap a bound operator function so every Python-side call pushes an NVTX range
// named `name` for the duration of the call, nesting the C-API submit ranges
// underneath it. The returned closure keeps the exact parameter signature of
// `fn`, so pybind11 still introspects the argument types and the argument
// annotations / default values on the m.def() site continue to apply unchanged.
// `name` must have static lifetime (string literals do); only the pointer is
// captured.
template<typename R, typename... Args>
auto NvtxTrace(const char *name, R (*fn)(Args...))
{
    return [name, fn](Args... args) -> R
    {
        NvtxRange range(name);
        // The lambda mirrors fn's parameter types (Args, not forwarding
        // references) so pybind11 keeps the original signature; static_cast<Args&&>
        // applies the same value-category cast std::forward would, and stays
        // correct when Args is a reference type.
        return fn(static_cast<Args &&>(args)...);
    };
}

// Member-function overload, for bindings registered with `cls.def(...)` on a
// pybind11 class (e.g. Tensor.cuda / Image.cpu). The closure takes the instance
// as its first parameter, which is exactly how pybind11 binds a free callable as
// an instance method, so the resulting method keeps the original signature. Only
// const methods are wrapped today; add a non-const overload if that changes.
template<typename R, typename C, typename... Args>
auto NvtxTrace(const char *name, R (C::*fn)(Args...) const)
{
    return [name, fn](const C &self, Args... args) -> R
    {
        NvtxRange range(name);
        // See the free-function overload: static_cast<Args&&> reproduces
        // std::forward's cast while keeping fn's exact signature for pybind11.
        return (self.*fn)(static_cast<Args &&>(args)...);
    };
}

} // namespace cvcudapy

#endif // CVCUDA_PYTHON_NVTXRANGE_HPP
