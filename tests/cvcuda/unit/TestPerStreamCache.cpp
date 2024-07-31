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

#include "Definitions.hpp"

#include <cvcuda/util/Event.hpp>
#include <cvcuda/util/PerStreamCache.hpp>
#include <cvcuda/util/Stream.hpp>

#include <random>

namespace {

class Hog
{
public:
    Hog(size_t size = 64 << 20)
        : m_size(size)
    {
        NVCV_CHECK_THROW(cudaMalloc(&m_buf, m_size));
    }

    ~Hog()
    {
        (void)cudaFree(m_buf);
    }

    void Run(cudaStream_t stream, int iters)
    {
        for (int i = 0; i < iters; i++)
        {
            NVCV_CHECK_THROW(cudaMemsetAsync(m_buf, i, m_size, stream));
        }
    }

private:
    void  *m_buf = nullptr;
    size_t m_size;
};

struct DummyPayload
{
    size_t      size = 0, alignment = 1;
    cudaEvent_t ready = nullptr;
};

using ItemAlloc = nvcv::util::detail::StreamCacheItemAllocator<DummyPayload>;

} // namespace

namespace nvcv::util {

TEST(StreamCacheItemAllocator, BasicTest)
{
    ItemAlloc                        alloc;
    std::vector<ItemAlloc::item_t *> items;
    std::mt19937_64                  rng;
    std::bernoulli_distribution      action;
    for (int i = 0; i < 1000; i++)
    {
        if (action(rng) || items.empty())
        {
            items.push_back(alloc.allocate());
        }
        else
        {
            int                                n = items.size();
            std::uniform_int_distribution<int> dist(0, n - 1);
            int                                i = dist(rng);
            std::swap(items[i], items.back());
            alloc.deallocate(items.back());
            items.pop_back();
        }
    }
    while (!items.empty())
    {
        alloc.deallocate(items.back());
        items.pop_back();
    }
}

namespace {

struct EventAlloc
{
    void reserve(int count)
    {
        for (int i = 0; i < count; i++) get();
        clear();
    }

    void clear()
    {
        for (auto &event : events) cache.put(std::move(event));
        events.clear();
    }

    std::vector<CudaEvent> events;
    EventCache             cache;

    cudaEvent_t get()
    {
        events.push_back(cache.get());
        return events.back().get();
    };
};

} // namespace

TEST(StreamOrderedCacheTest, InsertGet)
{
    EventAlloc                               events;
    ItemAlloc                                alloc;
    detail::StreamOrderedCache<DummyPayload> cache(&alloc);

    EXPECT_FALSE(cache.get(1000, 1).has_value()) << "The cache should be empty";
    DummyPayload p{};
    p.size  = 1000;
    p.ready = events.get();
    cache.put(std::move(p));
    EXPECT_FALSE(cache.get(2000, 1).has_value()) << "The cache doesn't contain any element large enough";
    auto v = cache.get(500, 1);
    ASSERT_TRUE(v.has_value());
    EXPECT_EQ(v->size, 1000) << "The cache contains a suitable element";
    v = cache.get(500, 1);
    EXPECT_FALSE(cache.get(0, 0)) << "The element was already removed";
}

TEST(StreamOrderedCacheTest, FindNextReady)
{
    EventAlloc                               events;
    ItemAlloc                                alloc;
    detail::StreamOrderedCache<DummyPayload> cache(&alloc);

    int N = 801;
    events.reserve(N);
    CudaStream stream = CudaStream::Create(true);

    Hog hog;

    const int kMaxRetries = 10;
    int       retries     = kMaxRetries;

    for (int split = 0; split < N; split += 20)
    {
        std::cout << split + 1 << "/" << N << std::endl;
        events.clear();

        int i;
        for (i = 0; i < split; i++)
        {
            DummyPayload dp{};
            dp.ready = events.get();
            dp.size  = i;
            ASSERT_EQ(cudaSuccess, cudaEventRecord(dp.ready, stream.get()));
            cache.put(std::move(dp));
        }

        hog.Run(stream.get(), 50);

        for (; i < N; i++)
        {
            DummyPayload dp{};
            dp.ready = events.get();
            dp.size  = i;
            ASSERT_EQ(cudaSuccess, cudaEventRecord(dp.ready, stream.get()));
            cache.put(std::move(dp));
        }

        if (split > 0)
        {
            ASSERT_EQ(cudaSuccess, cudaEventSynchronize(events.events[split - 1]));
        }
        auto *item = cache.findNewestReady();
        if (cudaEventQuery(events.events[split]) == cudaSuccess)
        {
            if (--retries < 0)
                GTEST_SKIP() << "Unreliable test";
            split--;
            cache.waitAndPurge();
            ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());
            continue;
        }
        retries = kMaxRetries;

        if (split == 0)
            EXPECT_EQ(item, nullptr);
        else
        {
            EXPECT_NE(item, nullptr);
            if (item)
            {
                EXPECT_EQ(item->payload.size, split - 1);
            }
        }

        ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

        cache.waitAndPurge();
    }
}

TEST(StreamOrderedCacheTest, RemoveAllReady)
{
    EventAlloc                               events;
    ItemAlloc                                alloc;
    detail::StreamOrderedCache<DummyPayload> cache(&alloc);

    CudaStream stream = CudaStream::Create(true);
    int        N      = 801;
    events.reserve(N);

    Hog hog;

    const int kMaxRetries = 10;
    int       retries     = kMaxRetries;

    std::vector<bool> mask(N);

    for (int split = 0; split < N; split += 20)
    {
        std::cout << split + 1 << "/" << N << std::endl;
        events.clear();
        for (int i = 0; i < N; i++) mask[i] = false;

        int i;
        for (i = 0; i < split; i++)
        {
            DummyPayload dp{};
            dp.ready = events.get();
            dp.size  = i;
            ASSERT_EQ(cudaSuccess, cudaEventRecord(dp.ready, stream.get()));
            cache.put(std::move(dp));
        }

        hog.Run(stream.get(), 50);

        for (; i < N; i++)
        {
            DummyPayload dp{};
            dp.ready = events.get();
            dp.size  = i;
            ASSERT_EQ(cudaSuccess, cudaEventRecord(dp.ready, stream.get()));
            cache.put(std::move(dp));
        }

        if (split > 0)
        {
            ASSERT_EQ(cudaSuccess, cudaEventSynchronize(events.events[split - 1]));
        }
        cache.removeAllReady([&](const DummyPayload &p) { mask[p.size] = true; });
        if (cudaEventQuery(events.events[split]) != cudaErrorNotReady)
        {
            if (--retries < 0)
                GTEST_SKIP() << "Unreliable test";
            split--;
            cache.waitAndPurge();
            ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());
            continue;
        }
        retries = kMaxRetries;
        for (int i = 0; i < N; i++)
        {
            EXPECT_EQ(mask[i], (i < split)) << "@ i = " << i << " split = " << split;
        }

        ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

        cache.waitAndPurge();
    }
}

TEST(PerStreamCacheTest, NoStream)
{
    PerStreamCache<DummyPayload> cache;

    EventAlloc events;

    DummyPayload p1{1000, 1, events.get()};
    DummyPayload p2{2000, 1, events.get()};
    DummyPayload p3{3000, 1, events.get()};
    cache.put(std::move(p1), std::nullopt);
    cache.put(std::move(p2), std::nullopt);
    cache.put(std::move(p3), std::nullopt);
    auto v1 = cache.get(1001, 0, std::nullopt);
    ASSERT_TRUE(v1.has_value());
    EXPECT_EQ(v1->size, 2000);
    auto v2 = cache.get(900, 0, std::nullopt);
    ASSERT_TRUE(v2.has_value());
    EXPECT_EQ(v2->size, 1000);
    auto v3 = cache.get(900, 0, std::nullopt);
    ASSERT_TRUE(v3.has_value());
    EXPECT_EQ(v3->size, 3000);
}

TEST(PerStreamCacheTest, TwoStream)
{
    for (int attempt = 0; attempt < 10; attempt++)
    {
        ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());
        EventAlloc                   events;
        PerStreamCache<DummyPayload> cache;

        Hog hog;

        CudaStream s1 = CudaStream::Create(true);
        CudaStream s2 = CudaStream::Create(true);

        DummyPayload p1{1000, 1, events.get()};
        DummyPayload p2{2000, 1, events.get()};
        DummyPayload p3{3000, 1, events.get()};
        DummyPayload p4{4000, 1, events.get()};

        hog.Run(s1.get(), 100);
        hog.Run(s2.get(), 100);

        ASSERT_EQ(cudaSuccess, cudaEventRecord(p1.ready, s1.get()));
        ASSERT_EQ(cudaSuccess, cudaEventRecord(p2.ready, s2.get()));
        ASSERT_EQ(cudaSuccess, cudaEventRecord(p3.ready, s1.get()));
        ASSERT_EQ(cudaSuccess, cudaEventRecord(p4.ready, s2.get()));

        auto s = std::chrono::high_resolution_clock::now();
        cache.put(std::move(p1), s1.get());
        cache.put(std::move(p2), s2.get());
        cache.put(std::move(p3), s1.get());
        cache.put(std::move(p4), s2.get());
        auto   e               = std::chrono::high_resolution_clock::now();
        double insert_time     = (e - s).count() / 4;
        double stream_get_time = 0;

        s                      = std::chrono::high_resolution_clock::now();
        auto v0                = cache.get(1, 0, std::nullopt);
        e                      = std::chrono::high_resolution_clock::now();
        double failed_get_time = (e - s).count();
        if (v0.has_value())
        {
            if (cudaSuccess == cudaEventQuery(p1.ready) || cudaSuccess == cudaEventQuery(p1.ready))
                continue;
            EXPECT_FALSE(v0.has_value()) << "The resources are not ready - none should be returned for null stream.";
        }

        s               = std::chrono::high_resolution_clock::now();
        auto v1s1       = cache.get(1001, 0, s1);
        e               = std::chrono::high_resolution_clock::now();
        stream_get_time = (e - s).count();
        ASSERT_TRUE(v1s1.has_value());
        EXPECT_EQ(v1s1->size, 3000);

        s               = std::chrono::high_resolution_clock::now();
        auto v2s1       = cache.get(900, 0, s1);
        e               = std::chrono::high_resolution_clock::now();
        stream_get_time = (e - s).count();
        ASSERT_TRUE(v2s1.has_value());
        EXPECT_EQ(v2s1->size, 1000);

        s               = std::chrono::high_resolution_clock::now();
        auto v1s2       = cache.get(900, 0, s2);
        e               = std::chrono::high_resolution_clock::now();
        stream_get_time = (e - s).count();
        stream_get_time /= 3;
        ASSERT_TRUE(v1s2.has_value());
        EXPECT_EQ(v1s2->size, 2000);

        ASSERT_EQ(cudaSuccess, cudaEventSynchronize(events.events[3]));
        s               = std::chrono::high_resolution_clock::now();
        auto v0ready    = cache.get(1, 0, std::nullopt);
        e               = std::chrono::high_resolution_clock::now();
        double get_time = (e - s).count();
        ASSERT_TRUE(v0ready.has_value());
        EXPECT_EQ(v0ready->size, 4000);

        std::cout << "Insert time = " << insert_time << "ns" << std::endl;
        std::cout << "Get time (stream) = " << stream_get_time << "ns" << std::endl;
        std::cout << "Get time (global, success) = " << get_time << "ns" << std::endl;
        std::cout << "Get time (global, failed) = " << failed_get_time << "ns" << std::endl;

        return;
    }
    GTEST_SKIP() << "Test unreliable - cannot make the CPU wait for the GPU";
}

} // namespace nvcv::util
