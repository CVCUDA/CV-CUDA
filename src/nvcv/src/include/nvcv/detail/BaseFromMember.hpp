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

#ifndef NVCV_DETAIL_BASE_FROM_MEMBER_HPP
#define NVCV_DETAIL_BASE_FROM_MEMBER_HPP

#include <utility>

namespace nvcv { namespace detail {

// Base-from-member idiom
// Ref: https://en.wikibooks.org/wiki/More_C%2B%2B_Idioms/Base-from-Member

/* Needed when we have to pass a member variable to a base class.
   To make sure the member is fully constructed, it must be defined as
   a base class, coming *before* the base that needs it. C++ rules
   guarantees that base classes are constructed in definition order.
   If there are multiple members with the same type, you must give them
   different IDs.

   Ex:
     struct Bar{};

     struct Foo
     {
         Foo(Bar &, Bar * =nullptr);
     };

     struct A
        : BaseFromMember<Bar>
        , Foo
     {
        using MemberBar = BaseFromMember<Bar>;

        A()
            : Foo(MemberBar::member)
        {
        }
     };

     struct B
        : BaseFromMember<Bar,0>
        , BaseFromMember<Bar,1>
        , Foo
     {
        using MemberBar0 = BaseFromMember<Bar,0>;
        using MemberBar1 = BaseFromMember<Bar,1>;

        A()
            : Foo(MemberBar0::member, MemberBar1::member)
        {
        }
     };
*/

template<class T, int ID = 0>
class BaseFromMember
{
public:
    T member;
};

template<class T, int ID>
class BaseFromMember<T &, ID>
{
public:
    BaseFromMember(T &m)
        : member(m)
    {
    }

    T &member;
};

}} // namespace nvcv::detail

#endif // NVCV_DETAIL_BASE_FROM_MEMBER_HPP
