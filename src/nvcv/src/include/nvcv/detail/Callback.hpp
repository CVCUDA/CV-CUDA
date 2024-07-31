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

#ifndef NVCV_CALLBACK_HPP
#define NVCV_CALLBACK_HPP

#include "TypeTraits.hpp"

#include <cstring>
#include <functional>
#include <memory>
#include <type_traits>
#include <utility>

namespace nvcv {

namespace detail {

struct NoTranslation
{
    template<typename Callable, typename... Args>
    auto operator()(Callable &&c, Args &&...args) -> decltype(c(std::forward<Args>(args)...))
    {
        return c(std::forward<Args>(args)...);
    }
};

template<typename Func>
struct AddContext;

template<typename Ret, typename... Args>
struct AddContext<Ret(Args...)>
{
    using type = Ret(void *, Args...);
};

template<typename Func>
using AddContext_t = typename AddContext<Func>::type;

} // namespace detail

template<typename CppFuncType, typename CFuncType = detail::AddContext_t<CppFuncType>,
         typename TranslateCall = detail::NoTranslation, bool SingleUse = false>
class Callback;

template<typename CppFuncType, typename CFuncType = detail::AddContext_t<CppFuncType>,
         typename TranslateCall = detail::NoTranslation>
using CleanupCallback = Callback<CppFuncType, CFuncType, TranslateCall, true>;

namespace detail {

template<typename X>
struct IsCallback : std::false_type
{
};

template<typename Cpp, typename C, typename Tr, bool SingleUse>
struct IsCallback<Callback<Cpp, C, Tr, SingleUse>> : std::true_type
{
};

} // namespace detail

/** Manages an object callable with Args and returning Ret
 *
 * In many ways, this object is similar to std::function, but aimed at providing C interface.
 * The C interface consists of the following triple:
 * - a "call" function with a context prepended to the argument list
 * - a context blob
 * - a context cleanup function
 *
 * The user may also add extra marshalling step by providing a (stateless) adapter between the C++ and C callbacks.
 * This is useful for avoiding C API leaking to C++ in callback signatures while still allowing to pass them with
 * little state.
 *
 * In case when the C consumer cannot handle the destruction of a callback, the presence thereof
 * can be detected at run-time.
 *
 * @tparam Ret              The return type of the C++ callback
 * @tparam Args             Argument types of the C++ callback
 * @tparam CtxArg           Type of the context argument in C callback; must be void*, used only to improve diagnostics
 * @tparam CRet             The return type of the C callback (see TranslateCall)
 * @tparam CArgs            Argument types of the C callback (see TranslateCall)
 * @tparam TranslateCall    A type of a stateless functor that translates C arguments to C++,
 *                          invokes the C++ callback target and translates the return value (or exceptions)
 *                          to C return value.
 * @tparam SingleUse        If true, the callable object can only be called once and, upon invocation,
 *                          the callable object wrapped by this Callback will be destroyed.
 */
template<typename Ret, typename... Args, typename CRet, typename CtxArg, typename... CArgs, typename TranslateCall,
         bool SingleUse>
class Callback<Ret(Args...), CRet(CtxArg, CArgs...), TranslateCall, SingleUse>
{
public:
    static_assert(std::is_same<CtxArg, void *>::value, "The first argument to the C wrapper must be of type void *");

    using FunctionType = Ret(Args...);
    using WrappedFunc  = CRet(void *, CArgs...);
    using CleanupFunc  = void(void *);

    Callback() = default;

    Callback(const Callback &) = delete;

    Callback(Callback &&cb)
    {
        *this = std::move(cb);
    }

    ~Callback()
    {
        reset();
    }

    /** Resets the contents of the callback, performing cleanup, if necessary
     *
     * @param function  the function to be called, taking the target and Args as arguments.
     *                  If SingleUse is true, the function must contain the necessary self-cleanup
     * @param target    pointer or an opaque value used by function anc cleanup as a context
     * @param cleanup   cleanup function used when the callback is destroyed.
     *                  If SingleUse is true, all required cleanup must be also performed by `function`
     *                  and `cleanup` will not be called.
     */
    void reset(WrappedFunc *function = nullptr, void *target = nullptr, CleanupFunc *cleanup = nullptr)
    {
        if (m_cleanup)
            m_cleanup(m_target.asOpaqueHandle());
        m_call = function;
        m_target.fromOpaqueHandle(target);
        m_cleanup = cleanup;
    }

    std::tuple<WrappedFunc *, void *, CleanupFunc *> release()
    {
        auto ret  = std::make_tuple(m_call, m_target.asOpaqueHandle(), m_cleanup);
        m_call    = nullptr;
        m_target  = {};
        m_cleanup = nullptr;
        return ret;
    }

    Callback &operator=(const Callback &cb) = delete;

    template<typename Function>
    Callback &operator=(const Callback<Function> &f) = delete;

    Callback &operator=(Callback &&cb)
    {
        if (&cb != this)
        {
            std::swap(m_call, cb.m_call);
            std::swap(m_target, cb.m_target);
            std::swap(m_cleanup, cb.m_cleanup);
            cb.reset();
        }
        return *this;
    }

    template<typename FunctionLike>
    detail::EnableIf_t<std::is_convertible<FunctionLike &&, Callback>::value, Callback &> &operator=(FunctionLike &&f)
    {
        return *this = Callback(std::forward<FunctionLike>(f));
    }

    /** Wraps a callable object
     *
     * This overload handles objects that are not function pointers and not std::functions
     */
    // clang-format off
    template<
        typename Callable,
        typename = detail::EnableIf_t<
            detail::IsInvocableR<Ret, Callable, Args...>::value &&
            !detail::IsCallback<detail::RemoveCVRef_t<Callable>>::value &&
            !detail::IsStdFunction<detail::RemoveCVRef_t<Callable>>::value
            >>
    // clang-format on
    Callback(Callable &&c)
    {
        fromCallable(std::forward<Callable>(c));
    }

    /** Wraps a function pointer
     */
    Callback(FunctionType *f)
    {
        fromFunction(f);
    }

    /** Wraps a compatible function pointer
     */
    template<typename RetF, typename... ArgsF,
             typename = detail::EnableIf_t<detail::IsInvocableR<Ret, RetF (*)(ArgsF...), Args...>::value>>
    Callback(RetF (*f)(ArgsF...))
    {
        fromFunction(f);
    }

    /** Creates a Callback object from a `std::function`.
     *
     * When the `std::function` wraps a simple pointer, it is unwrapped
     * and treated as a plain function, obviating the need for cleanup.
     */
    template<typename FunctionSig,
             typename = detail::EnableIf_t<detail::IsInvocableR<Ret, std::function<FunctionSig>, Args...>::value>>
    Callback(const std::function<FunctionSig> &f)
    {
        if (FunctionSig *const *fn = f.template target<FunctionSig *>())
            fromFunction(*fn);
        else
            fromCallable(f);
    }

    /** Tells whether the target needs any cleanup step or can just be thrown away.
     *
     * The result of this function may be more precise than simply checking the callable type with
     * `requiresCleanup<T>`, because the latter is pessimistic about `std::function`.
     */
    bool requiresCleanup() const
    {
        return m_cleanup != nullptr;
    }

    /** Tells whether given callable type _may_ require any cleanup.
     *
     * This function is pessimistic - it always returns `true` for std::function, while `std::function` that
     * wraps a function of the target type doesn't carry any meaningful state and doesn't require cleanup.
     */
    template<typename Callable>
    static constexpr bool requiresCleanup()
    {
        static_assert(detail::IsInvocableR<Ret, Callable, Args...>::value,
                      "The callable object is not invocable as the target function type");
        return !std::is_function<detail::RemovePointer_t<Callable>>::value && !isEmpty<Callable>()
            && !isByValue<Callable>();
    }

    /** Tells whether given callable _will_ require any cleanup.
     *
     * This function should return the same result as:
     * Callable<FunctionType>(c).requiresCleanup()
     * but it doesn't have the overhead of actually creating a callback nor does it move
     * out the contents of c.
     */
    template<typename Callable>
    static bool requiresCleanup(Callable &&c)
    {
        return requiresCleanupImpl(std::forward<Callable>(c));
    }

    /** Returns an opaque, type-erased value that describes the invocation target.
     */
    void *targetHandle() const
    {
        return m_target.asOpaqueHandle();
    }

    /** Returns the target function for interfacing with C - it adds a `void*` target as the first argument
     */
    WrappedFunc *targetFunc() const
    {
        return m_call;
    }

    /** Returns the cleanup function that destroys the target/context.
     *
     * The return value may be nullptr, indicating that no cleanup is necessary
     */
    CleanupFunc *cleanupFunc() const
    {
        return m_cleanup;
    }

    /** Returns true if this Callback has a target
     */
    explicit operator bool() const
    {
        return m_call != nullptr;
    }

    /** Invokes the target function, if any.
     *
     * If there's no target function, `std::bad_function_call` is thrown.
     */
    template<bool single_use = SingleUse>
    detail::EnableIf_t<!single_use, Ret> operator()(Args... args) const
    {
        static_assert(single_use == SingleUse, "The single_use template argument must not be modified manually.");
        if (!m_call)
            throw std::bad_function_call();
        return m_call(m_target.asOpaqueHandle(), std::move(args)...);
    }

    /** Invokes the target function, if any.
     *
     * If there's no target function, `std::bad_function_call` is thrown.
     */
    template<bool single_use = SingleUse>
    detail::EnableIf_t<single_use, Ret> operator()(Args... args)
    {
        static_assert(single_use == SingleUse, "The single_use template argument must not be modified manually.");
        if (!m_call)
            throw std::bad_function_call();
        auto *call   = m_call;
        auto  target = m_target;
        m_target     = {};
        m_call       = nullptr;
        m_cleanup    = nullptr;
        return call(target.asOpaqueHandle(), std::move(args)...);
    }

private:
    struct alignas(void *) TargetBlob
    {
        char data[sizeof(void *)];

        /** Reinterprets the contents of the blob as `void*` */
        void *asOpaqueHandle() const noexcept
        {
            void *h;
            std::memcpy(&h, data, sizeof(h));
            return h;
        }

        /** Copies the opaque handle to the data blob */
        void fromOpaqueHandle(void *h) noexcept
        {
            std::memcpy(data, &h, sizeof(data));
        }
    };

    template<typename Callable>
    static constexpr bool isEmpty()
    {
        using C = detail::RemoveRef_t<Callable>;
        return std::is_empty<C>::value && std::is_trivially_default_constructible<C>::value;
    }

    template<typename Callable>
    static constexpr bool isByValue()
    {
        using C = detail::RemoveRef_t<Callable>;
        return sizeof(C) <= sizeof(TargetBlob) && alignof(C) <= alignof(TargetBlob)
            && std::is_trivially_copyable<C>::value && std::is_trivially_destructible<C>::value;
    }

    enum class CallableKind
    {
        Other = 0,
        Empty,
        ByValue
    };

    template<typename Callable>
    void fromCallable(Callable &&c)
    {
        using C                     = detail::RemoveCVRef_t<Callable>;
        constexpr CallableKind kind = isEmpty<C>()   ? CallableKind::Empty
                                    : isByValue<C>() ? CallableKind::ByValue
                                                     : CallableKind::Other;
        fromCallable(std::forward<Callable>(c), std::integral_constant<CallableKind, kind>(),
                     std::integral_constant<bool, SingleUse>());
    }

    template<typename Callable>
    void fromCallable(Callable &&, std::integral_constant<CallableKind, CallableKind::Empty>,
                      std::integral_constant<bool, SingleUse>)
    {
        m_call = [](void *, CArgs... args) -> CRet
        {
            TranslateCall tr;
            return tr(Callable{}, std::move(args)...);
        };
    }

    template<typename Callable>
    void fromCallable(Callable &&c, std::integral_constant<CallableKind, CallableKind::ByValue>,
                      std::integral_constant<bool, SingleUse>)
    {
        using C = detail::RemoveCVRef_t<Callable>;
        new (m_target.data) C{std::forward<Callable>(c)};
        m_call = [](void *target, CArgs... args) -> CRet
        {
            C *c = reinterpret_cast<C *>(&target);

            TranslateCall tr;
            return tr(*c, std::move(args)...);
        };
    }

    template<typename Callable>
    void fromCallable(Callable &&c, std::integral_constant<CallableKind, CallableKind::Other>,
                      std::integral_constant<bool, false>)
    {
        using C = detail::RemoveCVRef_t<Callable>;

        std::unique_ptr<C> ptr(new C(std::forward<Callable>(c)));

        m_call = [](void *target, CArgs... args) -> CRet
        {
            TranslateCall tr;
            return tr(*static_cast<C *>(target), std::move(args)...);
        };
        m_cleanup = [](void *target)
        {
            delete static_cast<C *>(target);
        };
        m_target.fromOpaqueHandle(ptr.release());
    }

    template<typename Callable>
    void fromCallable(Callable &&c, std::integral_constant<CallableKind, CallableKind::Other>,
                      std::integral_constant<bool, true>)
    {
        using C = detail::RemoveCVRef_t<Callable>;

        std::unique_ptr<C> ptr(new C(std::forward<Callable>(c)));

        m_call = [](void *target, CArgs... args) -> CRet
        {
            // this will get destroyed even if the invocation or translation throws
            std::unique_ptr<C> c(static_cast<C *>(target));
            TranslateCall      tr;
            return tr(*c, std::move(args)...);
        };
        m_cleanup = [](void *target)
        {
            delete static_cast<C *>(target);
        };
        m_target.fromOpaqueHandle(ptr.release());
    }

    template<typename RetF, typename... ArgsF>
    void fromFunction(RetF (*f)(ArgsF...))
    {
        using FuncType = RetF(ArgsF...);
        m_target.fromOpaqueHandle(reinterpret_cast<void *>(f));
        m_call = [](void *target, CArgs... args) -> CRet
        {
            FuncType     *ff = reinterpret_cast<FuncType *>(target);
            TranslateCall tr;
            return tr(ff, std::move(args)...);
        };
    }

    template<typename Callable>
    static bool requiresCleanupImpl()
    {
        return requiresCleanup<Callable>();
    }

    template<typename FunctionSig,
             typename = detail::EnableIf_t<detail::IsInvocableR<Ret, std::function<FunctionSig>, Args...>::value>>
    static bool requiresCleanupImpl(const std::function<FunctionSig> &f)
    {
        return f.template target<FunctionType *>() == nullptr;
    }

    WrappedFunc *m_call    = nullptr;
    TargetBlob   m_target  = {};
    CleanupFunc *m_cleanup = nullptr;
};

} // namespace nvcv

#endif // NVCV_CALLBACK_HPP
