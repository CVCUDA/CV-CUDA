/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#ifndef MEMORY_HPP
#define MEMORY_HPP

#include <cuda_runtime.h>
#include <stdio.h>

#define checkRuntime(call) check_runtime(call, #call, __LINE__, __FILE__)

#define CUOSD_PRINT_E(f_, ...) \
    fprintf(stderr, "[cuOSD Error] at %s:%d : " f_, (const char *)__FILE__, __LINE__, ##__VA_ARGS__)

#define CUOSD_PRINT_W(f_, ...) printf("[cuOSD Warning] at %s:%d : " f_, (const char *)__FILE__, __LINE__, ##__VA_ARGS__)

inline static bool check_runtime(cudaError_t e, const char *call, int line, const char *file)
{
    if (e != cudaSuccess)
    {
        fprintf(stderr, "CUDA Runtime error %s # %s, code = %s [ %d ] in file %s:%d\n", call, cudaGetErrorString(e),
                cudaGetErrorName(e), e, file, line);
        return false;
    }
    return true;
}

template<typename T>
class Memory
{
public:
    T *host() const
    {
        return host_;
    }

    T *device() const
    {
        return device_;
    }

    size_t size() const
    {
        return size_;
    }

    size_t bytes() const
    {
        return size_ * sizeof(T);
    }

    virtual ~Memory()
    {
        free_memory();
    }

    void copy_host_to_device(cudaStream_t stream = nullptr)
    {
        checkRuntime(cudaMemcpyAsync(device_, host_, bytes(), cudaMemcpyHostToDevice, stream));
    }

    void copy_device_to_host(cudaStream_t stream = nullptr)
    {
        checkRuntime(cudaMemcpyAsync(host_, device_, bytes(), cudaMemcpyDeviceToHost, stream));
    }

    void alloc_or_resize_to(size_t size)
    {
        if (capacity_ < size)
        {
            free_memory();

            checkRuntime(cudaMallocHost(&host_, size * sizeof(T)));
            checkRuntime(cudaMalloc(&device_, size * sizeof(T)));
            capacity_ = size;
        }
        size_ = size;
    }

    void free_memory()
    {
        if (host_ || device_)
        {
            checkRuntime(cudaFreeHost(host_));
            checkRuntime(cudaFree(device_));
            host_     = nullptr;
            device_   = nullptr;
            capacity_ = 0;
            size_     = 0;
        }
    }

private:
    T     *host_     = nullptr;
    T     *device_   = nullptr;
    size_t size_     = 0;
    size_t capacity_ = 0;
};

#endif // MEMORY_HPP
