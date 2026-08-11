/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef NVCV_TEST_COMMON_HASHMD5_HPP
#define NVCV_TEST_COMMON_HASHMD5_HPP

#include <nvcv/util/Ranges.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>

namespace nvcv::test {

class HashMD5
{
public:
    HashMD5();
    HashMD5(const HashMD5 &) = delete;
    ~HashMD5();

    void                    operator()(const std::byte *data, size_t lenBytes) const;
    std::array<uint8_t, 16> getHashAndReset() const;

    template<class T>
    void operator()(const T &value) const
    {
        static_assert(std::has_unique_object_representations_v<T>, "Can't hash this type");
        auto  bytes = std::as_bytes(std::span{&value, size_t{1}});
        this->operator()(bytes.data(), bytes.size());
    }

private:
    struct Impl;

    struct ImplDeleter
    {
        void operator()(Impl *impl) const;
    };

    using ImplPtr = std::unique_ptr<Impl, ImplDeleter>;

    static ImplPtr CreateImpl();

    ImplPtr pimpl = CreateImpl();
};

template<class T>
requires(std::has_unique_object_representations_v<T> && !util::ranges::IsRange<T>) void Update(HashMD5 &hash,
                                                                                               const T &value)
{
    hash(value);
}

template<class T>
void Update(HashMD5 &, const T *)
{
    static_assert(sizeof(T) == 0, "Won't do md5 of a pointer");
}

void Update(HashMD5 &hash, const char *value);

template<class R>
requires util::ranges::IsRange<R> void Update(nvcv::test::HashMD5 &hash, const R &r)
{
    Update(hash, util::ranges::Size(r));
    // With C++20 we should use std::ranges::contiguous_range instead
    if constexpr (util::ranges::IsRandomAccessRange<
                      R> && std::has_unique_object_representations_v<util::ranges::RangeValue<R>>)
    {
        // It's faster to do this if range is contiguous and elements have unique object representation
        auto bytes = std::as_bytes(std::span{util::ranges::Data(r), static_cast<size_t>(util::ranges::Size(r))});
        hash(bytes.data(), bytes.size());
    }
    else
    {
        // Must go one by one
        for (auto &v : r)
        {
            Update(hash, v);
        }
    }
}

template<class T>
requires std::is_floating_point_v<T> void Update(HashMD5 &hash, const T &value)
{
    hash(std::hash<T>()(value));
}

template<typename... TT>
void Update(HashMD5 &hash, const std::tuple<TT...> &t)
{
    if constexpr (std::has_unique_object_representations_v<std::tuple<TT...>>)
    {
        return hash(t);
    }

    std::apply([&hash](const auto &...v) { (..., Update(hash, v)); }, t);
};

inline void Update(HashMD5 &hash, const std::string &s)
{
    auto bytes = std::as_bytes(std::span{s.data(), s.size()});
    return hash(bytes.data(), bytes.size());
}

inline void Update(HashMD5 &hash, const std::string_view &s)
{
    auto bytes = std::as_bytes(std::span{s.data(), s.size()});
    return hash(bytes.data(), bytes.size());
}

inline void Update(HashMD5 &hash, const std::type_info &t)
{
    return hash(t.hash_code());
}

template<class T>
void Update(HashMD5 &hash, const std::optional<T> &o)
{
    // We can't rely on std::hash<T> for optionals because they
    // require a valid hash specialization for T. Since our
    // types use HashValue overloads, we have to do this instead.
    if (o)
    {
        return Update(hash, *o);
    }
    else
    {
        return Update(hash, std::hash<std::optional<int>>()(std::nullopt));
    }
}

template<class T1, class T2, class... TT>
void Update(HashMD5 &hash, const T1 &v1, const T2 &v2, const TT &...v)
{
    Update(hash, v1);
    Update(hash, v2);

    (..., Update(hash, v));
}

} // namespace nvcv::test

#endif // NVCV_TEST_COMMON_HASHMD5_HPP
