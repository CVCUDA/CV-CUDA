/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef CV_CUDA_OSD_HPP
#define CV_CUDA_OSD_HPP

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#include <memory>
#include <vector>

namespace nvcv::cuda { namespace osd {

#define PREALLOC_CMD_NUM 100

#define checkRuntime(call) check_runtime(call, #call, __LINE__, __FILE__)

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

// RectangleCommand:
// ax1, ..., dy1: 4 outer corner points coordinate of the rectangle
// ax2, ..., dy2: 4 inner corner points coordinate of the rectangle
//    a1 ------ d1
//    | a2---d2 |
//    | |     | |
//    | b2---c2 |
//    b1 ------ c1
// thickness: border width in case > 0, -1 stands for fill mode
struct RectangleCommand
{
    uint8_t c0, c1, c2, c3;
    int     bounding_left   = 0;
    int     bounding_top    = 0;
    int     bounding_right  = 0;
    int     bounding_bottom = 0;

    int  batch_index   = 0;
    int  thickness     = -1;
    bool interpolation = false;

    float ax1, ay1, bx1, by1, cx1, cy1, dx1, dy1;
    float ax2, ay2, bx2, by2, cx2, cy2, dx2, dy2;
};

struct BoxBlurCommand
{
    uint8_t c0, c1, c2, c3;
    int     bounding_left   = 0;
    int     bounding_top    = 0;
    int     bounding_right  = 0;
    int     bounding_bottom = 0;

    int batch_index = 0;
    int kernel_size = 7;
};

struct cuOSDContext
{
    std::vector<std::shared_ptr<RectangleCommand>> rect_commands;
    std::unique_ptr<Memory<RectangleCommand>>      gpu_rect_commands;
    std::vector<std::shared_ptr<BoxBlurCommand>>   blur_commands;
    std::unique_ptr<Memory<BoxBlurCommand>>        gpu_blur_commands;

    int bounding_left   = 0;
    int bounding_top    = 0;
    int bounding_right  = 0;
    int bounding_bottom = 0;
};

typedef cuOSDContext *cuOSDContext_t;

}} // namespace nvcv::cuda::osd

#endif // CV_CUDA_OSD_HPP
