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

#ifndef CVCUDA_PRIV_WORKSPACE_ALLOCATOR_HPP
#define CVCUDA_PRIV_WORKSPACE_ALLOCATOR_HPP

#include <cvcuda/Workspace.hpp>

#include <optional>

namespace cvcuda {

class WorkspaceMemAllocator
{
public:
    WorkspaceMemAllocator(const WorkspaceMemAllocator &)            = delete;
    WorkspaceMemAllocator &operator=(const WorkspaceMemAllocator &) = delete;

    /**
     * @brief Construct a new workspace memory allocator
     *
     * The function constructs a new allocator. Subsequent calls to `get` will obtain memory pointers to
     * workspace entries.
     *
     * This function sets the default acquire and release streams, but doesn't call acquire - this is deferred to the
     * first call to `get`.
     * The streams can be overriden in manual calls to `acquire` and `release`.
     *
     * @param mem Workspace memory
     * @param acquireReleaseStream A stream on which the data will be used (or nullopt to denote host usage)
     */
    WorkspaceMemAllocator(const WorkspaceMem &mem, std::optional<cudaStream_t> acquireReleaseStream = std::nullopt)
        : WorkspaceMemAllocator(mem, acquireReleaseStream, acquireReleaseStream)
    {
    }

    /**
     * @brief Construct a new workspace memory allocator
     *
     * The function constructs a new allocator. Subsequent calls to `get` will obtain memory pointers to
     * workspace entries.
     *
     * This function sets the default acquire and release streams, but doesn't call acquire - this is deferred to the
     * first call to `get`.
     * The streams can be overriden in manual calls to `acquire` and `release`.
     *
     * @param mem Workspace memory
     * @param acquireStream A stream on which the data will be used first (or nullopt to denote host usage)
     * @param acquireStream A stream on which the data will be used last (or nullopt to denote host usage)
     */
    WorkspaceMemAllocator(const WorkspaceMem &mem, std::optional<cudaStream_t> acquireStream,
                          std::optional<cudaStream_t> releaseStream)
        : m_mem(mem)
        , m_acquireStream(acquireStream)
        , m_releaseStream(releaseStream)
    {
    }

    ~WorkspaceMemAllocator()
    {
        if (!m_released)
            release(m_releaseStream);
    }

    /**
     * @brief Allocates `count` elements of type `T` from the workspace memory.
     *
     * This function calls `acquire` if not called explicitly before.
     *
     * @tparam T        the type of the object to get
     * @param count     the number of objects to allocate
     * @param alignment the extra alignment, must not be less than `alignof(T)`
     * @return T*       a pointer to the workspace buffer where the requested object is located
     */
    template<typename T = char>
    T *get(size_t count = 1, size_t alignment = alignof(T))
    {
        assert(alignment >= alignof(T));

        if (m_released)
            throw std::logic_error("This workspace memory has been released.");

        if (!m_acquired && count > 0)
            acquire(m_acquireStream);

        if ((uintptr_t)m_mem.data & (alignment - 1))
        {
            throw nvcv::Exception(
                nvcv::Status::ERROR_INVALID_ARGUMENT,
                "The workspace base pointer is not aligned to match the required alignment of a workspace entry.");
        }

        size_t offset    = nvcv::detail::AlignUp(m_offset, alignment);
        T     *ret       = reinterpret_cast<T *>(static_cast<char *>(m_mem.data) + offset);
        size_t real_size = nvcv::detail::AlignUp(count * sizeof(T), alignment);
        offset += real_size;
        if (offset > m_mem.req.size)
            throw nvcv::Exception(nvcv::Status::ERROR_OUT_OF_MEMORY, "Operator workspace too small.");
        m_offset = offset;
        return ret;
    }

    constexpr size_t capacity() const
    {
        return m_mem.req.size;
    }

    constexpr size_t allocated() const
    {
        return m_offset;
    }

    /**
     * @brief Waits for the memory to become ready for use on the acquire stream, if specified, or on host.
     */
    void acquire(std::optional<cudaStream_t> stream)
    {
        if (m_acquired)
            throw std::logic_error("Acquire called multiple times");

        if (m_released)
            throw std::logic_error("This workspace memory has been released.");

        if (m_mem.ready)
        {
            if (stream)
            {
                if (cudaStreamWaitEvent(*stream, m_mem.ready) != cudaSuccess)
                    throw nvcv::Exception(nvcv::Status::ERROR_INTERNAL, "cudaStreamWairEvent failed");
            }
            else
            {
                if (cudaEventSynchronize(m_mem.ready) != cudaSuccess)
                    throw nvcv::Exception(nvcv::Status::ERROR_INTERNAL, "cudaEventSynchronize failed");
            }
        }
        m_acquired = true;
    }

    /**
     * @brief Declares that the memory is ready for reuse by the release stream (if specified) or any thread or stream.
     */
    void release(std::optional<cudaStream_t> stream)
    {
        if (m_released)
            throw std::logic_error("Release called multiple times");

        if (m_mem.ready && m_offset)
        {
            assert(m_acquired);

            if (stream)
                if (cudaEventRecord(m_mem.ready, *stream) != cudaSuccess)
                    throw nvcv::Exception(nvcv::Status::ERROR_INTERNAL, "cudaEventRecord failed");
        }
        m_released = true;
    }

private:
    WorkspaceMem m_mem;
    size_t       m_offset   = 0;
    bool         m_acquired = false, m_released = false;

    std::optional<cudaStream_t> m_acquireStream, m_releaseStream;
};

struct WorkspaceAllocator
{
public:
    explicit WorkspaceAllocator(const Workspace &ws)
        : hostMem(ws.hostMem)
        , pinnedMem(ws.pinnedMem)
        , cudaMem(ws.cudaMem)
    {
    }

    template<typename T = char>
    T *getHost(size_t count = 1, size_t alignment = alignof(T))
    {
        return hostMem.get<T>(count, alignment);
    }

    template<typename T = char>
    T *getPinned(size_t count = 1, size_t alignment = alignof(T))
    {
        return pinnedMem.get<T>(count, alignment);
    }

    template<typename T = char>
    T *getCuda(size_t count = 1, size_t alignment = alignof(T))
    {
        return cudaMem.get<T>(count, alignment);
    }

    WorkspaceMemAllocator hostMem;
    WorkspaceMemAllocator pinnedMem;
    WorkspaceMemAllocator cudaMem;
};

} // namespace cvcuda

#endif // CVCUDA_PRIV_WORKSPACE_ALLOCATOR_HPP
