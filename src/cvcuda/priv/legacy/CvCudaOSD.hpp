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

#include "textbackend/backend.hpp"

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#include <memory>
#include <string>
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

enum class cuOSDClockFormat : int
{
    None          = 0,
    YYMMDD_HHMMSS = 1,
    YYMMDD        = 2,
    HHMMSS        = 3
};

enum class cuOSDTextBackend : int
{
    None        = 0,
    StbTrueType = 1
};

struct cuOSDColor
{
    unsigned char r;
    unsigned char g;
    unsigned char b;
    unsigned char a;
};

enum class cuOSDImageFormat : int
{
    None = 0,
    RGB  = 1,
    RGBA = 2
};

enum class CommandType : int
{
    None      = 0,
    Circle    = 1,
    Rectangle = 2,
    Text      = 3,
    Segment   = 4,
    PolyFill  = 5,
    BoxBlur   = 6
};

struct TextLocation
{
    int image_x, image_y;
    int text_x;
    int text_w, text_h;
};

// cuOSDContextCommand includes basic attributes for color and bounding box coordinate
struct cuOSDContextCommand
{
    CommandType   type = CommandType::None;
    unsigned char c0, c1, c2, c3;
    int           bounding_left   = 0;
    int           bounding_top    = 0;
    int           bounding_right  = 0;
    int           bounding_bottom = 0;
    int           batch_index     = 0;
    int           reserved        = 0;
};

// CircleCommand:
// cx, cy: center point coordinate of the circle
// thickness: border width in case > 0, -1 stands for fill mode
struct CircleCommand : cuOSDContextCommand
{
    int cx, cy, radius, thickness;

    CircleCommand(int batch_idx, int cx, int cy, int radius, int thickness, unsigned char c0, unsigned char c1,
                  unsigned char c2, unsigned char c3)
    {
        this->batch_index = batch_idx;
        this->type        = CommandType::Circle;
        this->cx          = cx;
        this->cy          = cy;
        this->radius      = radius;
        this->thickness   = thickness;
        this->c0          = c0;
        this->c1          = c1;
        this->c2          = c2;
        this->c3          = c3;

        int half_thickness    = (thickness + 1) / 2 + 2;
        this->bounding_left   = cx - radius - half_thickness;
        this->bounding_right  = cx + radius + half_thickness;
        this->bounding_top    = cy - radius - half_thickness;
        this->bounding_bottom = cy + radius + half_thickness;
    }
};

// SegmentCommand:
// scale_x: seg mask w / outer rect w
// scale_y: seg mask h / outer rect h
struct SegmentCommand : cuOSDContextCommand
{
    float *dSeg;
    int    segWidth, segHeight;
    float  scale_x, scale_y;
    float  segThreshold;

    SegmentCommand()
    {
        this->type = CommandType::Segment;
    }
};

// PolyFillCommand:
struct PolyFillCommand : cuOSDContextCommand
{
    int *dPoints;
    int  numPoints;

    PolyFillCommand()
    {
        this->type = CommandType::PolyFill;
    }
};

// RectangleCommand:
// ax1, ..., dy1: 4 outer corner points coordinate of the rectangle
// ax2, ..., dy2: 4 inner corner points coordinate of the rectangle
// thickness: border width in case > 0, -1 stands for fill mode
struct RectangleCommand : cuOSDContextCommand
{
    int   thickness     = -1;
    bool  interpolation = false;
    float ax1, ay1, bx1, by1, cx1, cy1, dx1, dy1;
    float ax2, ay2, bx2, by2, cx2, cy2, dx2, dy2;

    RectangleCommand()
    {
        this->type = CommandType::Rectangle;
    }
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

// TextCommand:
// text_line_size && ilocation are inner attribute for text memory management
struct TextCommand : cuOSDContextCommand
{
    int text_line_size;
    int ilocation;

    TextCommand() = default;

    TextCommand(int text_line_size, int ilocation, unsigned char c0, unsigned char c1, unsigned char c2,
                unsigned char c3)
    {
        this->text_line_size = text_line_size;
        this->ilocation      = ilocation;
        this->type           = CommandType::Text;
        this->c0             = c0;
        this->c1             = c1;
        this->c2             = c2;
        this->c3             = c3;
    }
};

struct TextHostCommand : cuOSDContextCommand
{
    TextCommand                    gputile;
    std::vector<unsigned long int> text;
    unsigned short                 font_size;
    std::string                    font_name;
    int                            x, y;

    TextHostCommand(int batch_idx, const std::vector<unsigned long int> &text, unsigned short font_size,
                    const char *font, int x, int y, unsigned char c0, unsigned char c1, unsigned char c2,
                    unsigned char c3)
    {
        this->batch_index = batch_idx;
        this->text        = text;
        this->font_size   = font_size;
        this->font_name   = font;
        this->x           = x;
        this->y           = y;
        this->c0          = c0;
        this->c1          = c1;
        this->c2          = c2;
        this->c3          = c3;
        this->type        = CommandType::Text;
    }
};

struct cuOSDContext
{
    std::unique_ptr<Memory<TextLocation>> text_location;
    std::unique_ptr<Memory<int>>          line_location_base;

    std::vector<std::shared_ptr<cuOSDContextCommand>> commands;
    std::unique_ptr<Memory<unsigned char>>            gpu_commands;
    std::unique_ptr<Memory<int>>                      gpu_commands_offset;

    // For OpBndBox only, to be deprecated.
    std::vector<std::shared_ptr<RectangleCommand>> rect_commands;
    std::unique_ptr<Memory<RectangleCommand>>      gpu_rect_commands;

    std::vector<std::shared_ptr<BoxBlurCommand>> blur_commands;
    std::unique_ptr<Memory<BoxBlurCommand>>      gpu_blur_commands;

    std::shared_ptr<TextBackend> text_backend;
    cuOSDTextBackend             text_backend_type = cuOSDTextBackend::StbTrueType;

    bool have_rotate_msaa = false;
    int  bounding_left    = 0;
    int  bounding_top     = 0;
    int  bounding_right   = 0;
    int  bounding_bottom  = 0;
};

typedef cuOSDContext *cuOSDContext_t;

}} // namespace nvcv::cuda::osd

#endif // CV_CUDA_OSD_HPP
