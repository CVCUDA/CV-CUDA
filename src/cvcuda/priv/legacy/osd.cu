/* Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * SPDX-License-Identifier: Apache-2.0
 *
 * Copyright (C) 2021-2022, Bytedance Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
*/

#include "CvCudaLegacy.h"
#include "CvCudaLegacyHelpers.hpp"

#include "CvCudaUtils.cuh"

#include <cvcuda/priv/Types.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/ImageData.hpp>
#include <nvcv/TensorData.hpp>

#include <chrono>
#include <cstdio>
#include <ctime>
#include <iomanip>

using namespace nvcv::legacy::cuda_op;
using namespace nvcv::legacy::helpers;
using namespace nvcv::cuda::osd;
using namespace cvcuda::priv;

namespace nvcv::legacy::cuda_op {

template<typename _T>
static __host__ __device__ unsigned char u8cast(_T value)
{
    return value < 0 ? 0 : (value > 255 ? 255 : value);
}

inline static unsigned int round_down2(unsigned int num)
{
    return num & (~1);
}

template<typename _T>
static __forceinline__ __device__ _T limit(_T value, _T low, _T high)
{
    return value < low ? low : (value > high ? high : value);
}

#define INTER_RESIZE_COEF_BITS  11
#define INTER_RESIZE_COEF_SCALE (1 << INTER_RESIZE_COEF_BITS)

// inbox_single_pixel:
// check if given coordinate is in box
//      a --- d
//      |     |
//      b --- c
static __device__ __forceinline__ bool inbox_single_pixel(float ix, float iy, float ax, float ay, float bx, float by,
                                                          float cx, float cy, float dx, float dy)
{
    return ((bx - ax) * (iy - ay) - (by - ay) * (ix - ax)) < 0 && ((cx - bx) * (iy - by) - (cy - by) * (ix - bx)) < 0
        && ((dx - cx) * (iy - cy) - (dy - cy) * (ix - cx)) < 0 && ((ax - dx) * (iy - dy) - (ay - dy) * (ix - dx)) < 0;
}

static __device__ void blend_single_color(uchar4 &color, unsigned char &c0, unsigned char &c1, unsigned char &c2,
                                          unsigned char a)
{
    int foreground_alpha = a;
    int background_alpha = color.w;
    int blend_alpha      = ((background_alpha * (255 - foreground_alpha)) >> 8) + foreground_alpha;
    color.x = u8cast((((color.x * background_alpha * (255 - foreground_alpha)) >> 8) + (c0 * foreground_alpha))
                     / blend_alpha);
    color.y = u8cast((((color.y * background_alpha * (255 - foreground_alpha)) >> 8) + (c1 * foreground_alpha))
                     / blend_alpha);
    color.z = u8cast((((color.z * background_alpha * (255 - foreground_alpha)) >> 8) + (c2 * foreground_alpha))
                     / blend_alpha);
    color.w = blend_alpha;
}

static TextBackendType convert_to_text_backend_type(cuOSDTextBackend backend)
{
    switch (backend)
    {
    case cuOSDTextBackend::StbTrueType:
        return TextBackendType::StbTrueType;
    default:
        return TextBackendType::None;
    }
}

static void cuosd_draw_line(cuOSDContext_t context, int batch_idx, int x0, int y0, int x1, int y1, int thickness,
                            cuOSDColor color, bool interpolation)
{
    float length         = std::sqrt((float)((y1 - y0) * (y1 - y0) + (x1 - x0) * (x1 - x0)));
    float angle          = std::atan2((float)y1 - y0, (float)x1 - x0);
    float cos_angle      = std::cos(angle);
    float sin_angle      = std::sin(angle);
    float half_thickness = thickness / 2.0f;

    // upline
    // a    b
    // d    c
    auto cmd         = std::make_shared<RectangleCommand>();
    cmd->batch_index = batch_idx;
    cmd->ax1         = -half_thickness * cos_angle + x0 - sin_angle * half_thickness;
    cmd->ay1         = -half_thickness * sin_angle + cos_angle * half_thickness + y0;
    cmd->bx1         = (length + half_thickness) * cos_angle - sin_angle * half_thickness + x0;
    cmd->by1         = (length + half_thickness) * sin_angle + cos_angle * half_thickness + y0;
    cmd->dx1         = -half_thickness * cos_angle + x0 + sin_angle * half_thickness;
    cmd->dy1         = -half_thickness * sin_angle + y0 - cos_angle * half_thickness;
    cmd->cx1         = (length + half_thickness) * cos_angle + sin_angle * half_thickness + x0;
    cmd->cy1         = (length + half_thickness) * sin_angle - cos_angle * half_thickness + y0;

    if (x0 == x1 || y0 == y1)
        interpolation = false;

    if (interpolation)
        context->have_rotate_msaa = true;

    cmd->interpolation = interpolation;
    cmd->thickness     = -1;
    cmd->c0            = color.r;
    cmd->c1            = color.g;
    cmd->c2            = color.b;
    cmd->c3            = color.a;

    // a   d
    // b   c
    cmd->bounding_left   = min(min(min(cmd->ax1, cmd->bx1), cmd->cx1), cmd->dx1);
    cmd->bounding_right  = ceil(max(max(max(cmd->ax1, cmd->bx1), cmd->cx1), cmd->dx1));
    cmd->bounding_top    = min(min(min(cmd->ay1, cmd->by1), cmd->cy1), cmd->dy1);
    cmd->bounding_bottom = ceil(max(max(max(cmd->ay1, cmd->by1), cmd->cy1), cmd->dy1));
    context->commands.emplace_back(cmd);
}

static void cuosd_draw_rectangle(cuOSDContext_t context, int batch_idx, int left, int top, int right, int bottom,
                                 int thickness, cuOSDColor borderColor, cuOSDColor bgColor)
{
    int tl = min(left, right);
    int tt = min(top, bottom);
    int tr = max(left, right);
    int tb = max(top, bottom);
    left   = tl;
    top    = tt;
    right  = tr;
    bottom = tb;

    if (borderColor.a == 0)
        return;
    if (bgColor.a || thickness == -1)
    {
        if (thickness == -1)
        {
            bgColor = borderColor;
        }

        auto cmd           = std::make_shared<RectangleCommand>();
        cmd->batch_index   = batch_idx;
        cmd->thickness     = -1;
        cmd->interpolation = false;
        cmd->c0            = bgColor.r;
        cmd->c1            = bgColor.g;
        cmd->c2            = bgColor.b;
        cmd->c3            = bgColor.a;

        // a   d
        // b   c
        cmd->ax1             = left;
        cmd->ay1             = top;
        cmd->dx1             = right;
        cmd->dy1             = top;
        cmd->cx1             = right;
        cmd->cy1             = bottom;
        cmd->bx1             = left;
        cmd->by1             = bottom;
        cmd->bounding_left   = left;
        cmd->bounding_right  = right;
        cmd->bounding_top    = top;
        cmd->bounding_bottom = bottom;
        context->commands.emplace_back(cmd);
    }
    if (thickness == -1)
        return;

    auto cmd           = std::make_shared<RectangleCommand>();
    cmd->batch_index   = batch_idx;
    cmd->thickness     = thickness;
    cmd->interpolation = false;
    cmd->c0            = borderColor.r;
    cmd->c1            = borderColor.g;
    cmd->c2            = borderColor.b;
    cmd->c3            = borderColor.a;

    float half_thickness = thickness / 2.0f;
    cmd->ax2             = left + half_thickness;
    cmd->ay2             = top + half_thickness;
    cmd->dx2             = right - half_thickness;
    cmd->dy2             = top + half_thickness;
    cmd->cx2             = right - half_thickness;
    cmd->cy2             = bottom - half_thickness;
    cmd->bx2             = left + half_thickness;
    cmd->by2             = bottom - half_thickness;

    // a   d
    // b   c
    cmd->ax1 = left - half_thickness;
    cmd->ay1 = top - half_thickness;
    cmd->dx1 = right + half_thickness;
    cmd->dy1 = top - half_thickness;
    cmd->cx1 = right + half_thickness;
    cmd->cy1 = bottom + half_thickness;
    cmd->bx1 = left - half_thickness;
    cmd->by1 = bottom + half_thickness;

    int int_half         = ceil(half_thickness);
    cmd->bounding_left   = left - int_half;
    cmd->bounding_right  = right + int_half;
    cmd->bounding_top    = top - int_half;
    cmd->bounding_bottom = bottom + int_half;
    context->commands.emplace_back(cmd);
}

static void cuosd_draw_text(cuOSDContext_t context, int batch_idx, const char *utf8_text, int font_size,
                            const char *font, int x, int y, cuOSDColor borderColor, cuOSDColor bgColor)
{
    if (context->text_backend == nullptr)
        context->text_backend = create_text_backend(convert_to_text_backend_type(context->text_backend_type));

    if (context->text_backend == nullptr)
    {
        LOG_ERROR("There are no valid backend, please make sure your settings\n");
        return;
    }

    auto words = context->text_backend->split_utf8(utf8_text);
    if (words.empty() && strlen(utf8_text) > 0)
    {
        LOG_ERROR("There are some errors during converting UTF8 to Unicode.\n");
        return;
    }
    if (words.empty() || font_size <= 0)
        return;

    // Scale to 3x, in order to align with nvOSD effect
    font_size = context->text_backend->uniform_font_size(font_size);
    font_size = std::max(10, std::min(MAX_FONT_SIZE, font_size));

    int xmargin = font_size * 0.5;
    int ymargin = font_size * 0.25;

    int width, height, yoffset;
    std::tie(width, height, yoffset) = context->text_backend->measure_text(words, font_size, font);

    // add rectangle cmd as background color
    if (bgColor.a)
    {
        cuosd_draw_rectangle(context, batch_idx, x, y, x + width + 2 * xmargin - 1, y + height + 2 * ymargin - 1, -1,
                             *(cuOSDColor *)(&bgColor), {0, 0, 0, 0});
    }
    context->commands.emplace_back(std::make_shared<TextHostCommand>(batch_idx, words, font_size, font, x + xmargin,
                                                                     y + ymargin - yoffset, borderColor.r,
                                                                     borderColor.g, borderColor.b, borderColor.a));
}

static void cuosd_text_prepare(cuOSDContext_t context, int width, int height, cudaStream_t stream)
{
    if (context->commands.empty() || context->text_backend == nullptr)
        return;
    for (auto &cmd : context->commands)
    {
        if (cmd->type == CommandType::Text)
        {
            auto text_cmd = std::static_pointer_cast<TextHostCommand>(cmd);
            context->text_backend->add_build_text(text_cmd->text, text_cmd->font_size, text_cmd->font_name.c_str());
        }
    }
    context->text_backend->build_bitmap((void *)stream);

    std::vector<std::vector<TextLocation>> locations;
    int                                    total_locations = 0;
    for (auto &cmd : context->commands)
    {
        if (cmd->type == CommandType::Text)
        {
            auto text_cmd                     = std::static_pointer_cast<TextHostCommand>(cmd);
            int  draw_x                       = text_cmd->x;
            text_cmd->gputile.batch_index     = text_cmd->batch_index;
            text_cmd->gputile.bounding_left   = text_cmd->x;
            text_cmd->gputile.bounding_bottom = text_cmd->y + text_cmd->font_size;
            text_cmd->gputile.bounding_top    = text_cmd->gputile.bounding_bottom;

            auto glyph_map = context->text_backend->query(text_cmd->font_name.c_str(), text_cmd->font_size);
            if (glyph_map == nullptr)
                continue;

            std::vector<TextLocation> textline_locations;
            int                       max_glyph_height = 0;
            for (auto &word : text_cmd->text)
            {
                auto meta = glyph_map->query(word);
                if (meta == nullptr)
                    continue;

                max_glyph_height = max(max_glyph_height, meta->height());
            }

            for (auto &word : text_cmd->text)
            {
                auto meta = glyph_map->query(word);
                if (meta == nullptr)
                    continue;

                int w        = meta->width();
                int h        = meta->height();
                int xadvance = meta->xadvance(text_cmd->font_size);
                if (w < 1 || h < 1)
                {
                    draw_x += meta->xadvance(text_cmd->font_size, true);
                    continue;
                }

                TextLocation location;
                location.image_x = draw_x;
                location.image_y
                    = text_cmd->y
                    + context->text_backend->compute_y_offset(max_glyph_height, h, meta, text_cmd->font_size);
                location.text_x = meta->x_offset_on_bitmap();
                location.text_w = w;
                location.text_h = h;

                // Ignore if out of image area.
                if (location.image_x + location.text_w < 0 || location.image_x >= (int)width
                    || location.image_y + location.text_h < 0 || location.image_y >= (int)height)
                {
                    draw_x += xadvance;
                    continue;
                }

                textline_locations.emplace_back(location);
                draw_x += xadvance;
                text_cmd->gputile.bounding_bottom
                    = max(text_cmd->gputile.bounding_bottom, location.image_y + location.text_h);
                text_cmd->gputile.bounding_top = min(text_cmd->gputile.bounding_top, location.image_y);
            }

            text_cmd->gputile.bounding_right = draw_x;
            text_cmd->gputile.bounding_left  = text_cmd->gputile.bounding_left;

            if (!textline_locations.empty())
            {
                text_cmd->gputile.text_line_size = textline_locations.size();
                text_cmd->gputile.ilocation      = total_locations;
                text_cmd->gputile.c0             = cmd->c0;
                text_cmd->gputile.c1             = cmd->c1;
                text_cmd->gputile.c2             = cmd->c2;
                text_cmd->gputile.c3             = cmd->c3;
                text_cmd->gputile.type           = CommandType::Text;
                text_cmd->bounding_left          = text_cmd->gputile.bounding_left;
                text_cmd->bounding_top           = text_cmd->gputile.bounding_top;
                text_cmd->bounding_right         = text_cmd->gputile.bounding_right;
                text_cmd->bounding_bottom        = text_cmd->gputile.bounding_bottom;
                locations.emplace_back(textline_locations);
                total_locations += textline_locations.size();
            }
        }
    }
    if (locations.empty())
        return;

    if (context->text_location == nullptr)
        context->text_location.reset(new Memory<TextLocation>());
    if (context->line_location_base == nullptr)
        context->line_location_base.reset(new Memory<int>());
    context->text_location->alloc_or_resize_to(total_locations);
    context->line_location_base->alloc_or_resize_to(locations.size() + 1);

    int ilocation                          = 0;
    context->line_location_base->host()[0] = 0;

    for (int i = 0; i < (int)locations.size(); ++i)
    {
        auto &text_line = locations[i];
        memcpy(context->text_location->host() + ilocation, text_line.data(), sizeof(TextLocation) * text_line.size());
        ilocation += text_line.size();
        context->line_location_base->host()[i + 1] = ilocation;
    }

    context->line_location_base->copy_host_to_device(stream);
    context->text_location->copy_host_to_device(stream);
}

static void cuosd_apply(cuOSDContext_t context, int width, int height, cuOSDImageFormat format, cudaStream_t stream)
{
    if (context->commands.empty())
    {
        LOG_WARNING("Please check if there is anything to draw.\n");
        return;
    }

    if (!context->commands.empty())
    {
        cuosd_text_prepare(context, width, height, stream);

        context->bounding_left   = width;
        context->bounding_top    = height;
        context->bounding_right  = 0;
        context->bounding_bottom = 0;

        size_t                    byte_of_commands = 0;
        std::vector<unsigned int> cmd_offset(context->commands.size());
        for (int i = 0; i < (int)context->commands.size(); ++i)
        {
            auto &cmd     = context->commands[i];
            cmd_offset[i] = byte_of_commands;

            context->bounding_left   = min(context->bounding_left, cmd->bounding_left);
            context->bounding_top    = min(context->bounding_top, cmd->bounding_top);
            context->bounding_right  = max(context->bounding_right, cmd->bounding_right);
            context->bounding_bottom = max(context->bounding_bottom, cmd->bounding_bottom);

            if (cmd->type == CommandType::Text)
                byte_of_commands += sizeof(TextCommand);
            else if (cmd->type == CommandType::Rectangle)
                byte_of_commands += sizeof(RectangleCommand);
            else if (cmd->type == CommandType::Circle)
                byte_of_commands += sizeof(CircleCommand);
            else if (cmd->type == CommandType::Segment)
                byte_of_commands += sizeof(SegmentCommand);
            else if (cmd->type == CommandType::PolyFill)
                byte_of_commands += sizeof(PolyFillCommand);
        }

        if (context->gpu_commands == nullptr)
            context->gpu_commands.reset(new Memory<unsigned char>());
        if (context->gpu_commands_offset == nullptr)
            context->gpu_commands_offset.reset(new Memory<int>());

        context->gpu_commands->alloc_or_resize_to(byte_of_commands);
        context->gpu_commands_offset->alloc_or_resize_to(cmd_offset.size());
        memcpy(context->gpu_commands_offset->host(), cmd_offset.data(), sizeof(int) * cmd_offset.size());

        for (int i = 0; i < (int)context->commands.size(); ++i)
        {
            auto          &cmd    = context->commands[i];
            unsigned char *pg_cmd = context->gpu_commands->host() + cmd_offset[i];

            if (cmd->type == CommandType::Text)
            {
                memcpy(pg_cmd, (&(std::static_pointer_cast<TextHostCommand>(cmd)->gputile)), sizeof(TextCommand));
            }
            else if (cmd->type == CommandType::Rectangle)
            {
                memcpy(pg_cmd, cmd.get(), sizeof(RectangleCommand));
            }
            else if (cmd->type == CommandType::Circle)
            {
                memcpy(pg_cmd, cmd.get(), sizeof(CircleCommand));
            }
            else if (cmd->type == CommandType::Segment)
            {
                memcpy(pg_cmd, cmd.get(), sizeof(SegmentCommand));
            }
            else if (cmd->type == CommandType::PolyFill)
            {
                memcpy(pg_cmd, cmd.get(), sizeof(PolyFillCommand));
            }
        }
        context->gpu_commands->copy_host_to_device(stream);
        context->gpu_commands_offset->copy_host_to_device(stream);
    }
}

static __device__ bool render_text(int ix, int iy, const TextLocation &location, const unsigned char *text_bitmap,
                                   int text_bitmap_width, uchar4 color[4], unsigned char &c0, unsigned char &c1,
                                   unsigned char &c2, unsigned char &a)
{
    if (ix + 1 < location.image_x || iy + 1 < location.image_y || ix >= location.image_x + location.text_w
        || iy >= location.image_y + location.text_h)
        return false;

    int           fx     = ix - location.image_x;
    int           fy     = iy - location.image_y;
    int           bfx    = fx + location.text_x;
    unsigned char alpha0 = fx < 0 || fy < 0 || fx >= location.text_w || fy >= location.text_h
                             ? 0
                             : ((text_bitmap[fy * text_bitmap_width + bfx + 0] * (int)a) >> 8);
    unsigned char alpha1 = fx + 1 < 0 || fy < 0 || fx + 1 >= location.text_w || fy >= location.text_h
                             ? 0
                             : ((text_bitmap[fy * text_bitmap_width + bfx + 1] * (int)a) >> 8);
    unsigned char alpha2 = fx < 0 || fy + 1 < 0 || fx >= location.text_w || fy + 1 >= location.text_h
                             ? 0
                             : ((text_bitmap[(fy + 1) * text_bitmap_width + bfx + 0] * (int)a) >> 8);
    unsigned char alpha3 = fx + 1 < 0 || fy + 1 < 0 || fx + 1 >= location.text_w || fy + 1 >= location.text_h
                             ? 0
                             : ((text_bitmap[(fy + 1) * text_bitmap_width + bfx + 1] * (int)a) >> 8);

    if (alpha0)
    {
        blend_single_color(color[0], c0, c1, c2, alpha0);
    }
    if (alpha1)
    {
        blend_single_color(color[1], c0, c1, c2, alpha1);
    }
    if (alpha2)
    {
        blend_single_color(color[2], c0, c1, c2, alpha2);
    }
    if (alpha3)
    {
        blend_single_color(color[3], c0, c1, c2, alpha3);
    }
    return true;
}

template<class SrcWrapper, class DstWrapper, typename T, cuOSDImageFormat format>
struct BlendingPixel
{
};

template<class SrcWrapper, class DstWrapper, typename T>
struct BlendingPixel<SrcWrapper, DstWrapper, T, cuOSDImageFormat::RGBA>
{
    static __device__ void call(SrcWrapper src, DstWrapper dst, int x, int y, int stride, uchar4 plot_colors[4])
    {
        const int batch_idx = get_batch_idx();

        for (int i = 0; i < 2; ++i)
        {
            T *in  = src.ptr(batch_idx, y + i, x, 0);
            T *out = dst.ptr(batch_idx, y + i, x, 0);
            for (int j = 0; j < 2; ++j, in += 4, out += 4)
            {
                uchar4 &rcolor           = plot_colors[i * 2 + j];
                int     foreground_alpha = rcolor.w;
                int     background_alpha = in[3];
                int     blend_alpha      = ((background_alpha * (255 - foreground_alpha)) >> 8) + foreground_alpha;
                out[0]                   = u8cast(
                                      (((in[0] * background_alpha * (255 - foreground_alpha)) >> 8) + (rcolor.x * foreground_alpha))
                                      / blend_alpha);
                out[1] = u8cast(
                    (((in[1] * background_alpha * (255 - foreground_alpha)) >> 8) + (rcolor.y * foreground_alpha))
                    / blend_alpha);
                out[2] = u8cast(
                    (((in[2] * background_alpha * (255 - foreground_alpha)) >> 8) + (rcolor.z * foreground_alpha))
                    / blend_alpha);
                out[3] = blend_alpha;
            }
        }
    }
};

template<class SrcWrapper, class DstWrapper, typename T>
struct BlendingPixel<SrcWrapper, DstWrapper, T, cuOSDImageFormat::RGB>
{
    static __device__ void call(SrcWrapper src, DstWrapper dst, int x, int y, int stride, uchar4 plot_colors[4])
    {
        const int batch_idx = get_batch_idx();

        for (int i = 0; i < 2; ++i)
        {
            T *in  = src.ptr(batch_idx, y + i, x, 0);
            T *out = dst.ptr(batch_idx, y + i, x, 0);
            for (int j = 0; j < 2; ++j, in += 3, out += 3)
            {
                uchar4 &rcolor           = plot_colors[i * 2 + j];
                int     foreground_alpha = rcolor.w;
                int     background_alpha = 255;
                int     blend_alpha      = ((background_alpha * (255 - foreground_alpha)) >> 8) + foreground_alpha;
                out[0]                   = u8cast(
                                      (((in[0] * background_alpha * (255 - foreground_alpha)) >> 8) + (rcolor.x * foreground_alpha))
                                      / blend_alpha);
                out[1] = u8cast(
                    (((in[1] * background_alpha * (255 - foreground_alpha)) >> 8) + (rcolor.y * foreground_alpha))
                    / blend_alpha);
                out[2] = u8cast(
                    (((in[2] * background_alpha * (255 - foreground_alpha)) >> 8) + (rcolor.z * foreground_alpha))
                    / blend_alpha);
            }
        }
    }
};

// external_msaa4x:
// check if given coordinate is on border or outside the border, do msaa4x for border pixels
static __device__ __forceinline__ bool external_msaa4x(float ix, float iy, float ax, float ay, float bx, float by,
                                                       float cx, float cy, float dx, float dy, unsigned char a,
                                                       unsigned char &alpha)
{
    bool h0 = !inbox_single_pixel(ix - 0.25f, iy - 0.25f, ax, ay, bx, by, cx, cy, dx, dy);
    bool h1 = !inbox_single_pixel(ix + 0.25f, iy - 0.25f, ax, ay, bx, by, cx, cy, dx, dy);
    bool h2 = !inbox_single_pixel(ix + 0.25f, iy + 0.25f, ax, ay, bx, by, cx, cy, dx, dy);
    bool h3 = !inbox_single_pixel(ix - 0.25f, iy + 0.25f, ax, ay, bx, by, cx, cy, dx, dy);
    if (h0 || h1 || h2 || h3)
    {
        if (h0 && h1 && h2 && h3)
            return true;
        alpha = a * (h0 + h1 + h2 + h3) * 0.25f;
        return true;
    }
    return false;
}

// internal_msaa4x:
// check if given coordinate is on border or inside the border, do msaa4x for border pixels
static __device__ __forceinline__ bool internal_msaa4x(float ix, float iy, float ax, float ay, float bx, float by,
                                                       float cx, float cy, float dx, float dy, unsigned char a,
                                                       unsigned char &alpha)
{
    bool h0 = inbox_single_pixel(ix - 0.25f, iy - 0.25f, ax, ay, bx, by, cx, cy, dx, dy);
    bool h1 = inbox_single_pixel(ix + 0.25f, iy - 0.25f, ax, ay, bx, by, cx, cy, dx, dy);
    bool h2 = inbox_single_pixel(ix + 0.25f, iy + 0.25f, ax, ay, bx, by, cx, cy, dx, dy);
    bool h3 = inbox_single_pixel(ix - 0.25f, iy + 0.25f, ax, ay, bx, by, cx, cy, dx, dy);
    if (h0 || h1 || h2 || h3)
    {
        alpha = a * (h0 + h1 + h2 + h3) * 0.25f;
        return true;
    }
    return false;
}

// render_rectangle_fill_msaa4x:
// render filled rectangle with border msaa4x interpolation on
static __device__ void render_rectangle_fill_msaa4x(int ix, int iy, RectangleCommand *p, uchar4 color[4])
{
    unsigned char alpha;
    if (internal_msaa4x(ix, iy, p->ax1, p->ay1, p->bx1, p->by1, p->cx1, p->cy1, p->dx1, p->dy1, p->c3, alpha))
    {
        blend_single_color(color[0], p->c0, p->c1, p->c2, alpha);
    }
    if (internal_msaa4x(ix + 1, iy, p->ax1, p->ay1, p->bx1, p->by1, p->cx1, p->cy1, p->dx1, p->dy1, p->c3, alpha))
    {
        blend_single_color(color[1], p->c0, p->c1, p->c2, alpha);
    }
    if (internal_msaa4x(ix, iy + 1, p->ax1, p->ay1, p->bx1, p->by1, p->cx1, p->cy1, p->dx1, p->dy1, p->c3, alpha))
    {
        blend_single_color(color[2], p->c0, p->c1, p->c2, alpha);
    }
    if (internal_msaa4x(ix + 1, iy + 1, p->ax1, p->ay1, p->bx1, p->by1, p->cx1, p->cy1, p->dx1, p->dy1, p->c3, alpha))
    {
        blend_single_color(color[3], p->c0, p->c1, p->c2, alpha);
    }
}

// render_rectangle_fill:
// render filled rectangle with border msaa4x interpolation off
static __device__ void render_rectangle_fill(int ix, int iy, RectangleCommand *p, uchar4 color[4])
{
    if (inbox_single_pixel(ix, iy, p->ax1, p->ay1, p->bx1, p->by1, p->cx1, p->cy1, p->dx1, p->dy1))
    {
        blend_single_color(color[0], p->c0, p->c1, p->c2, p->c3);
    }
    if (inbox_single_pixel(ix + 1, iy, p->ax1, p->ay1, p->bx1, p->by1, p->cx1, p->cy1, p->dx1, p->dy1))
    {
        blend_single_color(color[1], p->c0, p->c1, p->c2, p->c3);
    }
    if (inbox_single_pixel(ix, iy + 1, p->ax1, p->ay1, p->bx1, p->by1, p->cx1, p->cy1, p->dx1, p->dy1))
    {
        blend_single_color(color[2], p->c0, p->c1, p->c2, p->c3);
    }
    if (inbox_single_pixel(ix + 1, iy + 1, p->ax1, p->ay1, p->bx1, p->by1, p->cx1, p->cy1, p->dx1, p->dy1))
    {
        blend_single_color(color[3], p->c0, p->c1, p->c2, p->c3);
    }
}

// render_rectangle_border_msaa4x:
// render hollow rectangle with border msaa4x interpolation on
static __device__ void render_rectangle_border_msaa4x(int ix, int iy, RectangleCommand *p, uchar4 color[4])
{
    unsigned char alpha;
    if (internal_msaa4x(ix, iy, p->ax1, p->ay1, p->bx1, p->by1, p->cx1, p->cy1, p->dx1, p->dy1, p->c3, alpha)
        && external_msaa4x(ix, iy, p->ax2, p->ay2, p->bx2, p->by2, p->cx2, p->cy2, p->dx2, p->dy2, p->c3, alpha))
    {
        blend_single_color(color[0], p->c0, p->c1, p->c2, alpha);
    }
    if (internal_msaa4x(ix + 1, iy, p->ax1, p->ay1, p->bx1, p->by1, p->cx1, p->cy1, p->dx1, p->dy1, p->c3, alpha)
        && external_msaa4x(ix + 1, iy, p->ax2, p->ay2, p->bx2, p->by2, p->cx2, p->cy2, p->dx2, p->dy2, p->c3, alpha))
    {
        blend_single_color(color[1], p->c0, p->c1, p->c2, alpha);
    }
    if (internal_msaa4x(ix, iy + 1, p->ax1, p->ay1, p->bx1, p->by1, p->cx1, p->cy1, p->dx1, p->dy1, p->c3, alpha)
        && external_msaa4x(ix, iy + 1, p->ax2, p->ay2, p->bx2, p->by2, p->cx2, p->cy2, p->dx2, p->dy2, p->c3, alpha))
    {
        blend_single_color(color[2], p->c0, p->c1, p->c2, alpha);
    }
    if (internal_msaa4x(ix + 1, iy + 1, p->ax1, p->ay1, p->bx1, p->by1, p->cx1, p->cy1, p->dx1, p->dy1, p->c3, alpha)
        && external_msaa4x(ix + 1, iy + 1, p->ax2, p->ay2, p->bx2, p->by2, p->cx2, p->cy2, p->dx2, p->dy2, p->c3,
                           alpha))
    {
        blend_single_color(color[3], p->c0, p->c1, p->c2, alpha);
    }
}

// render_rectangle_border:
// render hollow rectangle with border msaa4x interpolation off
static __device__ void render_rectangle_border(int ix, int iy, RectangleCommand *p, uchar4 color[4])
{
    if (!inbox_single_pixel(ix, iy, p->ax2, p->ay2, p->bx2, p->by2, p->cx2, p->cy2, p->dx2, p->dy2)
        && inbox_single_pixel(ix, iy, p->ax1, p->ay1, p->bx1, p->by1, p->cx1, p->cy1, p->dx1, p->dy1))
    {
        blend_single_color(color[0], p->c0, p->c1, p->c2, p->c3);
    }
    if (!inbox_single_pixel(ix + 1, iy, p->ax2, p->ay2, p->bx2, p->by2, p->cx2, p->cy2, p->dx2, p->dy2)
        && inbox_single_pixel(ix + 1, iy, p->ax1, p->ay1, p->bx1, p->by1, p->cx1, p->cy1, p->dx1, p->dy1))
    {
        blend_single_color(color[1], p->c0, p->c1, p->c2, p->c3);
    }
    if (!inbox_single_pixel(ix, iy + 1, p->ax2, p->ay2, p->bx2, p->by2, p->cx2, p->cy2, p->dx2, p->dy2)
        && inbox_single_pixel(ix, iy + 1, p->ax1, p->ay1, p->bx1, p->by1, p->cx1, p->cy1, p->dx1, p->dy1))
    {
        blend_single_color(color[2], p->c0, p->c1, p->c2, p->c3);
    }
    if (!inbox_single_pixel(ix + 1, iy + 1, p->ax2, p->ay2, p->bx2, p->by2, p->cx2, p->cy2, p->dx2, p->dy2)
        && inbox_single_pixel(ix + 1, iy + 1, p->ax1, p->ay1, p->bx1, p->by1, p->cx1, p->cy1, p->dx1, p->dy1))
    {
        blend_single_color(color[3], p->c0, p->c1, p->c2, p->c3);
    }
}

// interpolation_fn:
// interpolate alpha for border pixels
static __device__ unsigned char interpolation_fn(float x, int a, int b, int padding, unsigned char origin_alpha)
{
    int x0 = a - padding < 0 ? 0 : a - padding;
    int x1 = b + padding;
    if (x < x0 || x > x1)
        return 0;
    if (x >= a && x < b)
        return origin_alpha;
    if (x >= b && x <= x1)
        return (x1 - x) / padding * origin_alpha;
    if (x < a && x >= x0)
        return (x - x0) / padding * origin_alpha;
    return 0;
}

// render_circle_interpolation:
// render cicle with border interpolation
static __device__ void render_circle_interpolation(int ix, int iy, CircleCommand *p, uchar4 color[4])
{
    float tr0 = sqrt((float)(ix - p->cx) * (ix - p->cx) + (iy - p->cy) * (iy - p->cy));
    float tr1 = sqrt((float)(ix + 1 - p->cx) * (ix + 1 - p->cx) + (iy - p->cy) * (iy - p->cy));
    float tr2 = sqrt((float)(ix - p->cx) * (ix - p->cx) + (iy + 1 - p->cy) * (iy + 1 - p->cy));
    float tr3 = sqrt((float)(ix + 1 - p->cx) * (ix + 1 - p->cx) + (iy + 1 - p->cy) * (iy + 1 - p->cy));

    int inner_boundsize    = p->radius - p->thickness / 2;
    int external_boundsize = inner_boundsize + p->thickness;

    if (p->thickness < 0)
    {
        if (p->thickness == -1)
        {
            external_boundsize = p->radius;
        }
        else
        {
            external_boundsize = inner_boundsize;
        }
        inner_boundsize = 0;
    }

    unsigned char alpha0 = interpolation_fn(tr0, inner_boundsize, external_boundsize, 1, p->c3);
    unsigned char alpha1 = interpolation_fn(tr1, inner_boundsize, external_boundsize, 1, p->c3);
    unsigned char alpha2 = interpolation_fn(tr2, inner_boundsize, external_boundsize, 1, p->c3);
    unsigned char alpha3 = interpolation_fn(tr3, inner_boundsize, external_boundsize, 1, p->c3);

    if (alpha0)
    {
        blend_single_color(color[0], p->c0, p->c1, p->c2, alpha0);
    }
    if (alpha1)
    {
        blend_single_color(color[1], p->c0, p->c1, p->c2, alpha1);
    }
    if (alpha2)
    {
        blend_single_color(color[2], p->c0, p->c1, p->c2, alpha2);
    }
    if (alpha3)
    {
        blend_single_color(color[3], p->c0, p->c1, p->c2, alpha3);
    }
}

static __device__ void sample_pixel_bilinear(float *d_ptr, int x, int y, float sx, float sy, int width, int height,
                                             float threshold, unsigned char &a)
{
    float src_x  = (x + 0.5f) * sx - 0.5f;
    float src_y  = (y + 0.5f) * sy - 0.5f;
    int   y_low  = floorf(src_y);
    int   x_low  = floorf(src_x);
    int   y_high = limit(y_low + 1, 0, height - 1);
    int   x_high = limit(x_low + 1, 0, width - 1);
    y_low        = limit(y_low, 0, height - 1);
    x_low        = limit(x_low, 0, width - 1);

    int ly = rint((src_y - y_low) * INTER_RESIZE_COEF_SCALE);
    int lx = rint((src_x - x_low) * INTER_RESIZE_COEF_SCALE);
    int hy = INTER_RESIZE_COEF_SCALE - ly;
    int hx = INTER_RESIZE_COEF_SCALE - lx;

    uchar4 _scr;

    _scr.x = d_ptr[x_low + y_low * width] > threshold ? 127 : 0;
    _scr.y = d_ptr[x_high + y_low * width] > threshold ? 127 : 0;
    _scr.z = d_ptr[x_low + y_high * width] > threshold ? 127 : 0;
    _scr.w = d_ptr[x_high + y_high * width] > threshold ? 127 : 0;

    a = (((hy * ((hx * _scr.x + lx * _scr.y) >> 4)) >> 16) + ((ly * ((hx * _scr.z + lx * _scr.w) >> 4)) >> 16) + 2)
     >> 2;
}

static __device__ bool isRayIntersectsSegment(int p0, int p1, int s0, int s1, int e0, int e1)
{
    if (s1 == e1)
        return false;
    if (s1 > p1 && e1 > p1)
        return false;
    if (s1 < p1 && e1 < p1)
        return false;
    if (s1 == p1 && e1 > p1)
        return false;
    if (e1 == p1 && s1 > p1)
        return false;
    if (s0 < p0 && e0 < p0)
        return false;
    int xseg = e0 - (e0 - s0) * (e1 - p1) / (e1 - s1);
    if (xseg < p0)
        return false;
    return true;
}

static __device__ void render_polyfill(int ix, int iy, PolyFillCommand *p, uchar4 color[4])
{
    if (ix + 1 < p->bounding_left || iy + 1 < p->bounding_top || ix >= p->bounding_right || iy >= p->bounding_bottom)
        return;

    int sinsc[4] = {0, 0, 0, 0};
    for (int i = 0; i < p->numPoints; i++)
    {
        if (i == 0)
        {
            if (isRayIntersectsSegment(ix, iy, p->dPoints[0], p->dPoints[1], p->dPoints[p->numPoints * 2 - 2],
                                       p->dPoints[p->numPoints * 2 - 1]))
                sinsc[0] += 1;
            if (isRayIntersectsSegment(ix + 1, iy, p->dPoints[0], p->dPoints[1], p->dPoints[p->numPoints * 2 - 2],
                                       p->dPoints[p->numPoints * 2 - 1]))
                sinsc[1] += 1;
            if (isRayIntersectsSegment(ix, iy + 1, p->dPoints[0], p->dPoints[1], p->dPoints[p->numPoints * 2 - 2],
                                       p->dPoints[p->numPoints * 2 - 1]))
                sinsc[2] += 1;
            if (isRayIntersectsSegment(ix + 1, iy + 1, p->dPoints[0], p->dPoints[1], p->dPoints[p->numPoints * 2 - 2],
                                       p->dPoints[p->numPoints * 2 - 1]))
                sinsc[3] += 1;
        }
        else
        {
            if (isRayIntersectsSegment(ix, iy, p->dPoints[i * 2 - 2], p->dPoints[i * 2 - 1], p->dPoints[i * 2],
                                       p->dPoints[i * 2 + 1]))
                sinsc[0] += 1;
            if (isRayIntersectsSegment(ix + 1, iy, p->dPoints[i * 2 - 2], p->dPoints[i * 2 - 1], p->dPoints[i * 2],
                                       p->dPoints[i * 2 + 1]))
                sinsc[1] += 1;
            if (isRayIntersectsSegment(ix, iy + 1, p->dPoints[i * 2 - 2], p->dPoints[i * 2 - 1], p->dPoints[i * 2],
                                       p->dPoints[i * 2 + 1]))
                sinsc[2] += 1;
            if (isRayIntersectsSegment(ix + 1, iy + 1, p->dPoints[i * 2 - 2], p->dPoints[i * 2 - 1], p->dPoints[i * 2],
                                       p->dPoints[i * 2 + 1]))
                sinsc[3] += 1;
        }
    }

    if (sinsc[0] % 2 != 0)
    {
        blend_single_color(color[0], p->c0, p->c1, p->c2, p->c3);
    }

    if (sinsc[1] % 2 != 0)
    {
        blend_single_color(color[1], p->c0, p->c1, p->c2, p->c3);
    }

    if (sinsc[2] % 2 != 0)
    {
        blend_single_color(color[2], p->c0, p->c1, p->c2, p->c3);
    }

    if (sinsc[3] % 2 != 0)
    {
        blend_single_color(color[3], p->c0, p->c1, p->c2, p->c3);
    }
}

static __device__ void render_segment_bilinear(int ix, int iy, SegmentCommand *p, uchar4 color[4])
{
    if (ix + 1 < p->bounding_left || iy + 1 < p->bounding_top || ix >= p->bounding_right || iy >= p->bounding_bottom)
        return;

    unsigned char alpha0
        = ix < p->bounding_left || iy < p->bounding_top || ix >= p->bounding_right || iy >= p->bounding_bottom ? 0
                                                                                                               : 127;
    unsigned char alpha1
        = ix + 1 < p->bounding_left || iy < p->bounding_top || ix + 1 >= p->bounding_right || iy >= p->bounding_bottom
            ? 0
            : 127;
    unsigned char alpha2
        = ix < p->bounding_left || iy + 1 < p->bounding_top || ix >= p->bounding_right || iy + 1 >= p->bounding_bottom
            ? 0
            : 127;
    unsigned char alpha3 = ix + 1 < p->bounding_left || iy + 1 < p->bounding_top || ix + 1 >= p->bounding_right
                                || iy + 1 >= p->bounding_bottom
                             ? 0
                             : 127;

    int fx = ix - p->bounding_left;
    int fy = iy - p->bounding_top;

    if (alpha0)
    {
        sample_pixel_bilinear(p->dSeg, fx, fy, p->scale_x, p->scale_y, p->segWidth, p->segHeight, p->segThreshold,
                              alpha0);
        blend_single_color(color[0], p->c0, p->c1, p->c2, alpha0);
    }

    if (alpha1)
    {
        sample_pixel_bilinear(p->dSeg, fx + 1, fy, p->scale_x, p->scale_y, p->segWidth, p->segHeight, p->segThreshold,
                              alpha1);
        blend_single_color(color[1], p->c0, p->c1, p->c2, alpha1);
    }

    if (alpha2)
    {
        sample_pixel_bilinear(p->dSeg, fx, fy + 1, p->scale_x, p->scale_y, p->segWidth, p->segHeight, p->segThreshold,
                              alpha2);
        blend_single_color(color[2], p->c0, p->c1, p->c2, alpha2);
    }

    if (alpha3)
    {
        sample_pixel_bilinear(p->dSeg, fx + 1, fy + 1, p->scale_x, p->scale_y, p->segWidth, p->segHeight,
                              p->segThreshold, alpha3);
        blend_single_color(color[3], p->c0, p->c1, p->c2, alpha3);
    }
}

template<bool have_rotate_msaa>
static __device__ void do_rectangle(RectangleCommand *cmd, int ix, int iy, uchar4 context_color[4]);

template<>
__device__ void do_rectangle<true>(RectangleCommand *cmd, int ix, int iy, uchar4 context_color[4])
{
    if (cmd->thickness == -1)
    {
        if (cmd->interpolation)
        {
            render_rectangle_fill_msaa4x(ix, iy, cmd, context_color);
        }
        else
        {
            render_rectangle_fill(ix, iy, cmd, context_color);
        }
    }
    else
    {
        if (cmd->interpolation)
        {
            render_rectangle_border_msaa4x(ix, iy, cmd, context_color);
        }
        else
        {
            render_rectangle_border(ix, iy, cmd, context_color);
        }
    }
}

template<>
__device__ void do_rectangle<false>(RectangleCommand *cmd, int ix, int iy, uchar4 context_color[4])
{
    if (cmd->thickness == -1)
    {
        render_rectangle_fill(ix, iy, cmd, context_color);
    }
    else
    {
        render_rectangle_border(ix, iy, cmd, context_color);
    }
}

// render_elements_kernel:
// main entry for launching render CUDA kernel
template<cuOSDImageFormat format, bool have_rotate_msaa, class SrcWrapper, class DstWrapper,
         typename T = typename DstWrapper::ValueType>
static __global__ void render_elements_kernel(int bx, int by, const TextLocation *text_locations,
                                              const unsigned char *text_bitmap, int text_bitmap_width,
                                              const int *line_location_base, const unsigned char *commands,
                                              const int *command_offsets, int num_command, SrcWrapper src,
                                              DstWrapper dst, int image_width, int stride, int image_height,
                                              bool inplace)
{
    int ix = ((blockDim.x * blockIdx.x + threadIdx.x) << 1) + bx;
    int iy = ((blockDim.y * blockIdx.y + threadIdx.y) << 1) + by;
    if (ix < 0 || iy < 0 || ix >= image_width - 1 || iy >= image_height - 1)
        return;

    int       itext_line       = 0;
    uchar4    context_color[4] = {0};
    const int batch_idx        = get_batch_idx();

    for (int i = 0; i < num_command; ++i)
    {
        cuOSDContextCommand *pcommand = (cuOSDContextCommand *)(commands + command_offsets[i]);

        // because there is four pixel to operator
        if (pcommand->batch_index != batch_idx || ix + 1 < pcommand->bounding_left || ix > pcommand->bounding_right
            || iy + 1 < pcommand->bounding_top || iy > pcommand->bounding_bottom)
        {
            if (pcommand->type == CommandType::Text)
                itext_line++;
            continue;
        }

        switch (pcommand->type)
        {
        case CommandType::Rectangle:
        {
            RectangleCommand *rect_cmd = (RectangleCommand *)pcommand;
            do_rectangle<have_rotate_msaa>(rect_cmd, ix, iy, context_color);
            break;
        }
        case CommandType::Text:
        {
            int ilocation_begin = line_location_base[itext_line];
            int ilocation_end   = line_location_base[itext_line + 1];
            itext_line++;

            for (int j = ilocation_begin; j < ilocation_end; ++j)
            {
                bool hit = render_text(ix, iy, text_locations[j], text_bitmap, text_bitmap_width, context_color,
                                       pcommand->c0, pcommand->c1, pcommand->c2, pcommand->c3);
                if (hit)
                    break;
            }
            break;
        }
        case CommandType::Circle:
        {
            CircleCommand *circle_cmd = (CircleCommand *)pcommand;
            render_circle_interpolation(ix, iy, circle_cmd, context_color);
            break;
        }
        case CommandType::Segment:
        {
            SegmentCommand *seg_cmd = (SegmentCommand *)pcommand;
            render_segment_bilinear(ix, iy, seg_cmd, context_color);
            break;
        }
        case CommandType::PolyFill:
        {
            PolyFillCommand *poly_cmd = (PolyFillCommand *)pcommand;
            render_polyfill(ix, iy, poly_cmd, context_color);
            break;
        }
        }
    }

    if (context_color[0].w == 0 && context_color[1].w == 0 && context_color[2].w == 0 && context_color[3].w == 0)
    {
        if (inplace)
            return;
        if (format == cuOSDImageFormat::RGB)
        {
            *(uchar3 *)(dst.ptr(batch_idx, iy, ix, 0))         = *(uchar3 *)(src.ptr(batch_idx, iy, ix, 0));
            *(uchar3 *)(dst.ptr(batch_idx, iy, ix + 1, 0))     = *(uchar3 *)(src.ptr(batch_idx, iy, ix + 1, 0));
            *(uchar3 *)(dst.ptr(batch_idx, iy + 1, ix, 0))     = *(uchar3 *)(src.ptr(batch_idx, iy + 1, ix, 0));
            *(uchar3 *)(dst.ptr(batch_idx, iy + 1, ix + 1, 0)) = *(uchar3 *)(src.ptr(batch_idx, iy + 1, ix + 1, 0));
        }
        else if (format == cuOSDImageFormat::RGBA)
        {
            *(uchar4 *)(dst.ptr(batch_idx, iy, ix, 0))         = *(uchar4 *)(src.ptr(batch_idx, iy, ix, 0));
            *(uchar4 *)(dst.ptr(batch_idx, iy, ix + 1, 0))     = *(uchar4 *)(src.ptr(batch_idx, iy, ix + 1, 0));
            *(uchar4 *)(dst.ptr(batch_idx, iy + 1, ix, 0))     = *(uchar4 *)(src.ptr(batch_idx, iy + 1, ix, 0));
            *(uchar4 *)(dst.ptr(batch_idx, iy + 1, ix + 1, 0)) = *(uchar4 *)(src.ptr(batch_idx, iy + 1, ix + 1, 0));
        }
        return;
    }

    BlendingPixel<SrcWrapper, DstWrapper, T, format>::call(src, dst, ix, iy, stride, context_color);
}

static void cuosd_clear(cuOSDContext_t context)
{
    if (context)
    {
        context->commands.clear();
        context->blur_commands.clear();
    }
}

typedef void (*cuosd_launch_kernel_impl_fptr)(void *src, void *dst, int width, int stride, int height,
                                              const TextLocation *text_location, const unsigned char *text_bitmap,
                                              int text_bitmap_width, const int *line_location_base,
                                              const unsigned char *commands, const int *commands_offset,
                                              int num_commands, int bounding_left, int bounding_top, int bounding_right,
                                              int bounding_bottom, bool inplace, int batch, void *_stream);

template<class SrcWrapper, class DstWrapper, cuOSDImageFormat format, bool have_rotate_msaa>
static void cuosd_launch_kernel_impl(void *src, void *dst, int width, int stride, int height,
                                     const TextLocation *text_location, const unsigned char *text_bitmap,
                                     int text_bitmap_width, const int *line_location_base,
                                     const unsigned char *commands, const int *commands_offset, int num_commands,
                                     int bounding_left, int bounding_top, int bounding_right, int bounding_bottom,
                                     bool inplace, int batch, void *_stream)
{
    bounding_left   = max(min(bounding_left, width - 1), 0);
    bounding_top    = max(min(bounding_top, height - 1), 0);
    bounding_right  = max(min(bounding_right, width - 1), 0);
    bounding_bottom = max(min(bounding_bottom, height - 1), 0);

    bounding_left = round_down2(bounding_left);
    bounding_top  = round_down2(bounding_top);

    int bounding_width  = bounding_right - bounding_left + 1;
    int bounding_height = bounding_bottom - bounding_top + 1;
    if (bounding_width < 1 || bounding_height < 1)
    {
        LOG_WARNING("Please check if there is anything to draw, or cuosd_apply has been called\n");
        return;
    }

    cudaStream_t stream = (cudaStream_t)_stream;
    dim3         blockSize(16, 8);
    dim3         gridSize(divUp(int(((inplace ? bounding_width : width) + 1) / 2), (int)blockSize.x),
                          divUp(int(((inplace ? bounding_height : height) + 1) / 2), (int)blockSize.y), batch);

    render_elements_kernel<format, have_rotate_msaa, SrcWrapper, DstWrapper><<<gridSize, blockSize, 0, stream>>>(
        inplace ? bounding_left : 0, inplace ? bounding_top : 0, text_location, text_bitmap, text_bitmap_width,
        line_location_base, commands, commands_offset, num_commands, *(SrcWrapper *)src, *(DstWrapper *)dst, width,
        stride, height, inplace);
    cudaError_t code = cudaPeekAtLastError();
    if (code != cudaSuccess)
    {
        LOG_ERROR("Launch kernel (render_elements_kernel) failed, code = " << static_cast<int>(code));
    }
}

template<class SrcWrapper, class DstWrapper>
void cuosd_launch_kernel(SrcWrapper src, DstWrapper dst, int width, int stride, int height, cuOSDImageFormat format,
                         const TextLocation *text_location, const unsigned char *text_bitmap, int text_bitmap_width,
                         const int *line_location_base, const unsigned char *commands, const int *commands_offset,
                         int num_commands, int bounding_left, int bounding_top, int bounding_right, int bounding_bottom,
                         bool have_rotate_msaa, bool inplace, int batch, void *_stream)
{
    if (num_commands > 0)
    {
        static const cuosd_launch_kernel_impl_fptr func_list[] = {
            cuosd_launch_kernel_impl<SrcWrapper, DstWrapper, cuOSDImageFormat::RGB, false>,
            cuosd_launch_kernel_impl<SrcWrapper, DstWrapper, cuOSDImageFormat::RGBA, false>,
            cuosd_launch_kernel_impl<SrcWrapper, DstWrapper, cuOSDImageFormat::RGB, true>,
            cuosd_launch_kernel_impl<SrcWrapper, DstWrapper, cuOSDImageFormat::RGBA, true>,
        };

        int index = (int)(have_rotate_msaa)*2 + (int)format - 1;
        if (index < 0 || index >= (int)sizeof(func_list) / (int)sizeof(func_list[0]))
        {
            LOG_ERROR("Unsupported configure " << (int)index);
            return;
        }

        func_list[index]((void *)(&src), (void *)(&dst), width, stride, height, text_location, text_bitmap,
                         text_bitmap_width, line_location_base, commands, commands_offset, num_commands, bounding_left,
                         bounding_top, bounding_right, bounding_bottom, inplace, batch, _stream);
    }
}

template<class SrcWrapper, class DstWrapper>
void cuosd_launch(cuOSDContext_t context, SrcWrapper src, DstWrapper dst, int width, int stride, int height,
                  cuOSDImageFormat format, bool inplace, int batch, void *stream)
{
    if (context->commands.empty())
    {
        LOG_WARNING("Please check if there is anything to draw\n");
        return;
    }

    unsigned char *text_bitmap       = nullptr;
    int            text_bitmap_width = 0;
    if (context->text_backend)
    {
        text_bitmap       = context->text_backend->bitmap_device_pointer();
        text_bitmap_width = context->text_backend->bitmap_width();
    }

    cuosd_launch_kernel(
        src, dst, width, stride, height, format, context->text_location ? context->text_location->device() : nullptr,
        text_bitmap, text_bitmap_width, context->line_location_base ? context->line_location_base->device() : nullptr,
        context->gpu_commands ? context->gpu_commands->device() : nullptr,
        context->gpu_commands_offset ? context->gpu_commands_offset->device() : nullptr, context->commands.size(),
        context->bounding_left, context->bounding_top, context->bounding_right, context->bounding_bottom,
        context->have_rotate_msaa, inplace, batch, stream);
    checkRuntime(cudaPeekAtLastError());
}

static ErrorCode cuosd_draw_text(cuOSDContext_t context, int batch_idx, NVCVText text)
{
    const char *utf8_text   = text.utf8Text;
    const char *font        = text.fontName;
    int         font_size   = text.fontSize;
    int         x           = text.tlPos.x;
    int         y           = text.tlPos.y;
    cuOSDColor  borderColor = *(cuOSDColor *)(&text.fontColor);
    cuOSDColor  bgColor     = *(cuOSDColor *)(&text.bgColor);

    if (context->text_backend == nullptr)
        context->text_backend = create_text_backend(convert_to_text_backend_type(context->text_backend_type));

    if (context->text_backend == nullptr)
    {
        LOG_ERROR("There are no valid backend, please make sure your settings\n");
        return ErrorCode::INVALID_PARAMETER;
    }

    auto words = context->text_backend->split_utf8(utf8_text);
    if (words.empty() && strlen(utf8_text) > 0)
    {
        LOG_ERROR("There are some errors during converting UTF8 to Unicode.\n");
        return ErrorCode::INVALID_PARAMETER;
    }
    if (words.empty() || font_size <= 0)
        return ErrorCode::INVALID_PARAMETER;

    // Scale to 3x, in order to align with nvOSD effect
    font_size = context->text_backend->uniform_font_size(font_size);
    font_size = std::max(10, std::min(MAX_FONT_SIZE, font_size));

    int xmargin = font_size * 0.5;
    int ymargin = font_size * 0.25;

    int width, height, yoffset;
    std::tie(width, height, yoffset) = context->text_backend->measure_text(words, font_size, font);

    // add rectangle cmd as background color
    if (bgColor.a)
    {
        cuosd_draw_rectangle(context, batch_idx, x, y, x + width + 2 * xmargin - 1, y + height + 2 * ymargin - 1, -1,
                             bgColor, {0, 0, 0, 0});
    }
    context->commands.emplace_back(std::make_shared<TextHostCommand>(batch_idx, words, font_size, font, x + xmargin,
                                                                     y + ymargin - yoffset, borderColor.r,
                                                                     borderColor.g, borderColor.b, borderColor.a));

    return ErrorCode::SUCCESS;
}

static ErrorCode cuosd_draw_rectangle(cuOSDContext_t context, int batch_idx, int width, int height, NVCVBndBoxI bbox)
{
    int left   = max(min(bbox.box.x, width - 1), 0);
    int top    = max(min(bbox.box.y, height - 1), 0);
    int right  = max(min(left + bbox.box.width - 1, width - 1), 0);
    int bottom = max(min(top + bbox.box.height - 1, height - 1), 0);

    if (left == right || top == bottom || bbox.box.width <= 0 || bbox.box.height <= 0)
    {
        LOG_DEBUG("Skipped bnd_box(" << bbox.box.x << ", " << bbox.box.y << ", " << bbox.box.width << ", "
                                     << bbox.box.height << ") in image(" << width << ", " << height << ")");
        return ErrorCode::SUCCESS;
    }

    if (bbox.borderColor.a == 0)
    {
        return ErrorCode::SUCCESS;
    }

    if (bbox.fillColor.a || bbox.thickness == -1)
    {
        if (bbox.thickness == -1)
        {
            bbox.fillColor = bbox.borderColor;
        }

        auto cmd           = std::make_shared<RectangleCommand>();
        cmd->batch_index   = batch_idx;
        cmd->thickness     = -1;
        cmd->interpolation = false;
        cmd->c0            = bbox.fillColor.r;
        cmd->c1            = bbox.fillColor.g;
        cmd->c2            = bbox.fillColor.b;
        cmd->c3            = bbox.fillColor.a;

        // a   d
        // b   c
        cmd->ax1             = left;
        cmd->ay1             = top;
        cmd->dx1             = right;
        cmd->dy1             = top;
        cmd->cx1             = right;
        cmd->cy1             = bottom;
        cmd->bx1             = left;
        cmd->by1             = bottom;
        cmd->bounding_left   = left;
        cmd->bounding_right  = right;
        cmd->bounding_top    = top;
        cmd->bounding_bottom = bottom;
        context->commands.emplace_back(cmd);
    }
    if (bbox.thickness == -1)
    {
        return ErrorCode::SUCCESS;
    }

    auto cmd           = std::make_shared<RectangleCommand>();
    cmd->batch_index   = batch_idx;
    cmd->thickness     = bbox.thickness;
    cmd->interpolation = false;
    cmd->c0            = bbox.borderColor.r;
    cmd->c1            = bbox.borderColor.g;
    cmd->c2            = bbox.borderColor.b;
    cmd->c3            = bbox.borderColor.a;

    float half_thickness = bbox.thickness / 2.0f;
    cmd->ax2             = left + half_thickness;
    cmd->ay2             = top + half_thickness;
    cmd->dx2             = right - half_thickness;
    cmd->dy2             = top + half_thickness;
    cmd->cx2             = right - half_thickness;
    cmd->cy2             = bottom - half_thickness;
    cmd->bx2             = left + half_thickness;
    cmd->by2             = bottom - half_thickness;

    // a   d
    // b   c
    cmd->ax1 = left - half_thickness;
    cmd->ay1 = top - half_thickness;
    cmd->dx1 = right + half_thickness;
    cmd->dy1 = top - half_thickness;
    cmd->cx1 = right + half_thickness;
    cmd->cy1 = bottom + half_thickness;
    cmd->bx1 = left - half_thickness;
    cmd->by1 = bottom + half_thickness;

    int int_half         = ceil(half_thickness);
    cmd->bounding_left   = left - int_half;
    cmd->bounding_right  = right + int_half;
    cmd->bounding_top    = top - int_half;
    cmd->bounding_bottom = bottom + int_half;
    context->commands.emplace_back(cmd);

    return ErrorCode::SUCCESS;
}

static ErrorCode cuosd_draw_segmentmask(cuOSDContext_t context, int batch_idx, int width, int height,
                                        const NVCVSegment &segment)
{
    int left   = segment.box.x;
    int top    = segment.box.y;
    int right  = left + segment.box.width - 1;
    int bottom = top + segment.box.height - 1;

    if (left == right || top == bottom || segment.box.width <= 0 || segment.box.height <= 0)
    {
        LOG_DEBUG("Skipped bnd_box(" << segment.box.x << ", " << segment.box.y << ", " << segment.box.width << ", "
                                     << segment.box.height << ") in image(" << width << ", " << height << ")");
        return ErrorCode::SUCCESS;
    }

    float half_thickness = segment.thickness / 2.0f;
    int   int_half       = ceil(half_thickness);

    if (segment.borderColor.a && segment.thickness > 0)
    {
        auto cmd           = std::make_shared<RectangleCommand>();
        cmd->batch_index   = batch_idx;
        cmd->thickness     = segment.thickness;
        cmd->interpolation = false;
        cmd->c0            = segment.borderColor.r;
        cmd->c1            = segment.borderColor.g;
        cmd->c2            = segment.borderColor.b;
        cmd->c3            = segment.borderColor.a;

        cmd->ax2 = left + half_thickness;
        cmd->ay2 = top + half_thickness;
        cmd->dx2 = right - half_thickness;
        cmd->dy2 = top + half_thickness;
        cmd->cx2 = right - half_thickness;
        cmd->cy2 = bottom - half_thickness;
        cmd->bx2 = left + half_thickness;
        cmd->by2 = bottom - half_thickness;

        // a   d
        // b   c
        cmd->ax1 = left - half_thickness;
        cmd->ay1 = top - half_thickness;
        cmd->dx1 = right + half_thickness;
        cmd->dy1 = top - half_thickness;
        cmd->cx1 = right + half_thickness;
        cmd->cy1 = bottom + half_thickness;
        cmd->bx1 = left - half_thickness;
        cmd->by1 = bottom + half_thickness;

        cmd->bounding_left   = left - int_half;
        cmd->bounding_right  = right + int_half;
        cmd->bounding_top    = top - int_half;
        cmd->bounding_bottom = bottom + int_half;
        context->commands.emplace_back(cmd);
    }

    auto cmd         = std::make_shared<SegmentCommand>();
    cmd->batch_index = batch_idx;

    cmd->dSeg      = segment.dSeg;
    cmd->segWidth  = segment.segWidth;
    cmd->segHeight = segment.segHeight;

    cmd->scale_x      = segment.segWidth / (right - left + 1e-5);
    cmd->scale_y      = segment.segHeight / (bottom - top + 1e-5);
    cmd->segThreshold = segment.segThreshold;

    cmd->c0 = segment.segColor.r;
    cmd->c1 = segment.segColor.g;
    cmd->c2 = segment.segColor.b;
    cmd->c3 = segment.segColor.a;

    cmd->bounding_left   = left - int_half + 1;
    cmd->bounding_right  = right + int_half;
    cmd->bounding_top    = top - int_half + 1;
    cmd->bounding_bottom = bottom + int_half;

    context->commands.emplace_back(cmd);

    return ErrorCode::SUCCESS;
}

static ErrorCode cuosd_draw_circle(cuOSDContext_t context, int batch_idx, NVCVCircle circle)
{
    if (circle.bgColor.a && circle.thickness > 0)
    {
        context->commands.emplace_back(std::make_shared<CircleCommand>(
            batch_idx, circle.centerPos.x, circle.centerPos.y, circle.radius, -circle.thickness, circle.bgColor.r,
            circle.bgColor.g, circle.bgColor.b, circle.bgColor.a));
    }
    context->commands.emplace_back(std::make_shared<CircleCommand>(
        batch_idx, circle.centerPos.x, circle.centerPos.y, circle.radius, circle.thickness, circle.borderColor.r,
        circle.borderColor.g, circle.borderColor.b, circle.borderColor.a));
    return ErrorCode::SUCCESS;
}

static ErrorCode cuosd_draw_point(cuOSDContext_t context, int batch_idx, NVCVPoint point)
{
    context->commands.emplace_back(std::make_shared<CircleCommand>(batch_idx, point.centerPos.x, point.centerPos.y,
                                                                   point.radius, -1, point.color.r, point.color.g,
                                                                   point.color.b, point.color.a));
    return ErrorCode::SUCCESS;
}

static ErrorCode cuosd_draw_line(cuOSDContext_t context, int batch_idx, NVCVLine line)
{
    float length         = std::sqrt((float)((line.pos1.y - line.pos0.y) * (line.pos1.y - line.pos0.y)
                                     + (line.pos1.x - line.pos0.x) * (line.pos1.x - line.pos0.x)));
    float angle          = std::atan2((float)line.pos1.y - line.pos0.y, (float)line.pos1.x - line.pos0.x);
    float cos_angle      = std::cos(angle);
    float sin_angle      = std::sin(angle);
    float half_thickness = line.thickness / 2.0f;

    // upline
    // a    b
    // d    c
    auto cmd         = std::make_shared<RectangleCommand>();
    cmd->batch_index = batch_idx;
    cmd->ax1         = -half_thickness * cos_angle + line.pos0.x - sin_angle * half_thickness;
    cmd->ay1         = -half_thickness * sin_angle + cos_angle * half_thickness + line.pos0.y;
    cmd->bx1         = (length + half_thickness) * cos_angle - sin_angle * half_thickness + line.pos0.x;
    cmd->by1         = (length + half_thickness) * sin_angle + cos_angle * half_thickness + line.pos0.y;
    cmd->dx1         = -half_thickness * cos_angle + line.pos0.x + sin_angle * half_thickness;
    cmd->dy1         = -half_thickness * sin_angle + line.pos0.y - cos_angle * half_thickness;
    cmd->cx1         = (length + half_thickness) * cos_angle + sin_angle * half_thickness + line.pos0.x;
    cmd->cy1         = (length + half_thickness) * sin_angle - cos_angle * half_thickness + line.pos0.y;

    if (line.pos0.x == line.pos1.x || line.pos0.y == line.pos1.y)
        line.interpolation = false;

    if (line.interpolation)
        context->have_rotate_msaa = true;

    cmd->interpolation = line.interpolation;
    cmd->thickness     = -1;
    cmd->c0            = line.color.r;
    cmd->c1            = line.color.g;
    cmd->c2            = line.color.b;
    cmd->c3            = line.color.a;

    // a   d
    // b   c
    cmd->bounding_left   = min(min(min(cmd->ax1, cmd->bx1), cmd->cx1), cmd->dx1);
    cmd->bounding_right  = ceil(max(max(max(cmd->ax1, cmd->bx1), cmd->cx1), cmd->dx1));
    cmd->bounding_top    = min(min(min(cmd->ay1, cmd->by1), cmd->cy1), cmd->dy1);
    cmd->bounding_bottom = ceil(max(max(max(cmd->ay1, cmd->by1), cmd->cy1), cmd->dy1));
    context->commands.emplace_back(cmd);

    return ErrorCode::SUCCESS;
}

static ErrorCode cuosd_draw_polyline(cuOSDContext_t context, int batch_idx, const NVCVPolyLine &pl)
{
    if (pl.numPoints < 2)
        return ErrorCode::INVALID_PARAMETER;

    int bleft   = pl.hPoints[0];
    int bright  = pl.hPoints[0];
    int btop    = pl.hPoints[1];
    int bbottom = pl.hPoints[1];
    int nline   = 0;

    for (int i = 1; i < pl.numPoints; i++)
    {
        int x = pl.hPoints[2 * i];
        int y = pl.hPoints[2 * i + 1];

        bleft   = min(x, bleft);
        bright  = max(x, bright);
        btop    = min(y, btop);
        bbottom = max(y, bbottom);
        cuosd_draw_line(context, batch_idx, pl.hPoints[2 * i - 2], pl.hPoints[2 * i - 1], x, y, pl.thickness,
                        *(cuOSDColor *)(&pl.borderColor), pl.interpolation);
        nline++;
    }

    if (pl.numPoints > 2)
    {
        if (pl.isClosed)
        {
            cuosd_draw_line(context, batch_idx, pl.hPoints[0], pl.hPoints[1], pl.hPoints[2 * pl.numPoints - 2],
                            pl.hPoints[2 * pl.numPoints - 1], pl.thickness, *(cuOSDColor *)(&pl.borderColor),
                            pl.interpolation);
            nline++;
        }

        // Fill poly if alpha is not 0 and point num > 2
        if (pl.fillColor.a)
        {
            auto cmd             = std::make_shared<PolyFillCommand>();
            cmd->batch_index     = batch_idx;
            cmd->dPoints         = pl.dPoints;
            cmd->numPoints       = pl.numPoints;
            cmd->bounding_left   = bleft;
            cmd->bounding_right  = bright;
            cmd->bounding_top    = btop;
            cmd->bounding_bottom = bbottom;
            cmd->c0              = pl.fillColor.r;
            cmd->c1              = pl.fillColor.g;
            cmd->c2              = pl.fillColor.b;
            cmd->c3              = pl.fillColor.a;
            context->commands.insert(context->commands.end() - nline, cmd);
        }
    }

    return ErrorCode::SUCCESS;
}

static ErrorCode cuosd_draw_arrow(cuOSDContext_t context, int batch_idx, NVCVArrow arrow)
{
    int e03_x = arrow.pos0.x - arrow.pos1.x;
    int e03_y = arrow.pos0.y - arrow.pos1.y;

    float e03_len    = std::sqrt((float)(e03_x * e03_x) + e03_y * e03_y);
    float e03_norm_x = e03_x / e03_len;
    float e03_norm_y = e03_y / e03_len;

    float cos_theta = std::cos(3.1415926f / 12);
    float sin_theta = std::sin(3.1415926f / 12);

    int x_0 = arrow.pos1.x + int((e03_norm_x * cos_theta - e03_norm_y * sin_theta) * arrow.arrowSize);
    int y_0 = arrow.pos1.y + int((e03_norm_x * sin_theta + e03_norm_y * cos_theta) * arrow.arrowSize);
    int x_1 = arrow.pos1.x + int((e03_norm_x * cos_theta + e03_norm_y * sin_theta) * arrow.arrowSize);
    int y_1 = arrow.pos1.y + int((-e03_norm_x * sin_theta + e03_norm_y * cos_theta) * arrow.arrowSize);

    cuosd_draw_line(context, batch_idx, arrow.pos0.x, arrow.pos0.y, arrow.pos1.x, arrow.pos1.y, arrow.thickness,
                    *(cuOSDColor *)(&arrow.color), arrow.interpolation);
    cuosd_draw_line(context, batch_idx, x_0, y_0, arrow.pos1.x, arrow.pos1.y, arrow.thickness,
                    *(cuOSDColor *)(&arrow.color), arrow.interpolation);
    cuosd_draw_line(context, batch_idx, x_1, y_1, arrow.pos1.x, arrow.pos1.y, arrow.thickness,
                    *(cuOSDColor *)(&arrow.color), arrow.interpolation);

    return ErrorCode::SUCCESS;
}

static ErrorCode cuosd_draw_rotationbox(cuOSDContext_t context, int batch_idx, NVCVRotatedBox rb)
{
    if (rb.borderColor.a == 0)
        return ErrorCode::INVALID_PARAMETER;

    // a   d
    //   o
    // b   c

    float pax = -rb.width / 2.0f;
    float pay = -rb.height / 2.0f;
    float pcx = +rb.width / 2.0f;
    float pcy = +rb.height / 2.0f;
    float pbx = pax;
    float pby = pcy;
    float pdx = pcx;
    float pdy = pay;

    float cos_angle      = std::cos(rb.yaw);
    float sin_angle      = std::sin(rb.yaw);
    float half_thickness = rb.thickness / 2.0f;

    int angle        = (rb.yaw / M_PI * 180) + 0.5f;
    rb.interpolation = rb.interpolation && angle % 90 != 0;

    if (rb.interpolation)
        context->have_rotate_msaa = true;

    if (rb.bgColor.a || rb.thickness == -1)
    {
        if (rb.thickness == -1)
        {
            rb.bgColor = rb.borderColor;
        }

        auto cmd           = std::make_shared<RectangleCommand>();
        cmd->batch_index   = batch_idx;
        cmd->thickness     = -1;
        cmd->interpolation = rb.interpolation;
        cmd->c0            = rb.bgColor.r;
        cmd->c1            = rb.bgColor.g;
        cmd->c2            = rb.bgColor.b;
        cmd->c3            = rb.bgColor.a;

        // a   d
        // b   c
        // cosa, -sina, ox;
        // sina, cosa,  oy;
        cmd->ax1 = cos_angle * pax - sin_angle * pay + rb.centerPos.x;
        cmd->ay1 = sin_angle * pax + cos_angle * pay + rb.centerPos.y;
        cmd->bx1 = cos_angle * pbx - sin_angle * pby + rb.centerPos.x;
        cmd->by1 = sin_angle * pbx + cos_angle * pby + rb.centerPos.y;
        cmd->cx1 = cos_angle * pcx - sin_angle * pcy + rb.centerPos.x;
        cmd->cy1 = sin_angle * pcx + cos_angle * pcy + rb.centerPos.y;
        cmd->dx1 = cos_angle * pdx - sin_angle * pdy + rb.centerPos.x;
        cmd->dy1 = sin_angle * pdx + cos_angle * pdy + rb.centerPos.y;

        cmd->bounding_left   = min(min(min(cmd->ax1, cmd->bx1), cmd->cx1), cmd->dx1);
        cmd->bounding_right  = ceil(max(max(max(cmd->ax1, cmd->bx1), cmd->cx1), cmd->dx1));
        cmd->bounding_top    = min(min(min(cmd->ay1, cmd->by1), cmd->cy1), cmd->dy1);
        cmd->bounding_bottom = ceil(max(max(max(cmd->ay1, cmd->by1), cmd->cy1), cmd->dy1));
        context->commands.emplace_back(cmd);
    }
    if (rb.thickness == -1)
        return ErrorCode::INVALID_PARAMETER;

    auto cmd         = std::make_shared<RectangleCommand>();
    cmd->batch_index = batch_idx;
    cmd->thickness   = rb.thickness;
    cmd->c0          = rb.borderColor.r;
    cmd->c1          = rb.borderColor.g;
    cmd->c2          = rb.borderColor.b;
    cmd->c3          = rb.borderColor.a;

    cmd->interpolation = rb.interpolation;
    cmd->ax1           = cos_angle * (pax - half_thickness) - sin_angle * (pay - half_thickness) + rb.centerPos.x;
    cmd->ay1           = sin_angle * (pax - half_thickness) + cos_angle * (pay - half_thickness) + rb.centerPos.y;
    cmd->bx1           = cos_angle * (pbx - half_thickness) - sin_angle * (pby + half_thickness) + rb.centerPos.x;
    cmd->by1           = sin_angle * (pbx - half_thickness) + cos_angle * (pby + half_thickness) + rb.centerPos.y;
    cmd->cx1           = cos_angle * (pcx + half_thickness) - sin_angle * (pcy + half_thickness) + rb.centerPos.x;
    cmd->cy1           = sin_angle * (pcx + half_thickness) + cos_angle * (pcy + half_thickness) + rb.centerPos.y;
    cmd->dx1           = cos_angle * (pdx + half_thickness) - sin_angle * (pdy - half_thickness) + rb.centerPos.x;
    cmd->dy1           = sin_angle * (pdx + half_thickness) + cos_angle * (pdy - half_thickness) + rb.centerPos.y;

    cmd->ax2 = cos_angle * (pax + half_thickness) - sin_angle * (pay + half_thickness) + rb.centerPos.x;
    cmd->ay2 = sin_angle * (pax + half_thickness) + cos_angle * (pay + half_thickness) + rb.centerPos.y;
    cmd->bx2 = cos_angle * (pbx + half_thickness) - sin_angle * (pby - half_thickness) + rb.centerPos.x;
    cmd->by2 = sin_angle * (pbx + half_thickness) + cos_angle * (pby - half_thickness) + rb.centerPos.y;
    cmd->cx2 = cos_angle * (pcx - half_thickness) - sin_angle * (pcy - half_thickness) + rb.centerPos.x;
    cmd->cy2 = sin_angle * (pcx - half_thickness) + cos_angle * (pcy - half_thickness) + rb.centerPos.y;
    cmd->dx2 = cos_angle * (pdx - half_thickness) - sin_angle * (pdy + half_thickness) + rb.centerPos.x;
    cmd->dy2 = sin_angle * (pdx - half_thickness) + cos_angle * (pdy + half_thickness) + rb.centerPos.y;

    cmd->bounding_left   = min(min(min(cmd->ax1, cmd->bx1), cmd->cx1), cmd->dx1);
    cmd->bounding_right  = ceil(max(max(max(cmd->ax1, cmd->bx1), cmd->cx1), cmd->dx1));
    cmd->bounding_top    = min(min(min(cmd->ay1, cmd->by1), cmd->cy1), cmd->dy1);
    cmd->bounding_bottom = ceil(max(max(max(cmd->ay1, cmd->by1), cmd->cy1), cmd->dy1));
    context->commands.emplace_back(cmd);

    return ErrorCode::SUCCESS;
}

static ErrorCode cuosd_draw_clock(cuOSDContext_t context, int batch_idx, NVCVClock clock)
{
    std::chrono::time_point<std::chrono::system_clock> time_now   = std::chrono::system_clock::now();
    std::time_t                                        time_now_t = std::chrono::system_clock::to_time_t(time_now);
    if (clock.time != 0)
        time_now_t = clock.time;

    std::tm            now_tm = *std::localtime(&time_now_t);
    std::ostringstream oss;
    if (clock.clockFormat == NVCVClockFormat::HHMMSS)
    {
        oss << std::put_time(&now_tm, "%H:%M:%S");
    }
    else if (clock.clockFormat == NVCVClockFormat::YYMMDD)
    {
        oss << std::put_time(&now_tm, "%Y-%m-%d");
    }
    else if (clock.clockFormat == NVCVClockFormat::YYMMDD_HHMMSS)
    {
        oss << std::put_time(&now_tm, "%Y-%m-%d %H:%M:%S");
    }
    auto utf8_str = oss.str();

    cuosd_draw_text(context, batch_idx, utf8_str.c_str(), clock.fontSize, clock.font, clock.tlPos.x, clock.tlPos.y,
                    *(cuOSDColor *)(&clock.fontColor), *(cuOSDColor *)(&clock.bgColor));
    return ErrorCode::SUCCESS;
}

static ErrorCode cuosd_draw_elements(cuOSDContext_t context, int width, int height, NVCVElementsImpl *ctx)
{
    for (int n = 0; n < ctx->batch(); n++)
    {
        auto numElements = ctx->numElementsAt(n);

        for (int i = 0; i < numElements; i++)
        {
            auto element = ctx->elementAt(n, i);
            auto type    = element->type();
            auto data    = element->ptr();
            switch (type)
            {
            case NVCVOSDType::NVCV_OSD_NONE:
            {
                return ErrorCode::INVALID_PARAMETER;
            }
            case NVCVOSDType::NVCV_OSD_RECT:
            {
                cuosd_draw_rectangle(context, n, width, height, *((NVCVBndBoxI *)data));
                break;
            }
            case NVCVOSDType::NVCV_OSD_TEXT:
            {
                cuosd_draw_text(context, n, *((NVCVText *)data));
                break;
            }
            case NVCVOSDType::NVCV_OSD_SEGMENT:
            {
                cuosd_draw_segmentmask(context, n, width, height, *((NVCVSegment *)data));
                break;
            }
            case NVCVOSDType::NVCV_OSD_POINT:
            {
                cuosd_draw_point(context, n, *((NVCVPoint *)data));
                break;
            }
            case NVCVOSDType::NVCV_OSD_LINE:
            {
                cuosd_draw_line(context, n, *((NVCVLine *)data));
                break;
            }
            case NVCVOSDType::NVCV_OSD_POLYLINE:
            {
                cuosd_draw_polyline(context, n, *((NVCVPolyLine *)data));
                break;
            }
            case NVCVOSDType::NVCV_OSD_ROTATED_RECT:
            {
                cuosd_draw_rotationbox(context, n, *((NVCVRotatedBox *)data));
                break;
            }
            case NVCVOSDType::NVCV_OSD_CIRCLE:
            {
                cuosd_draw_circle(context, n, *((NVCVCircle *)data));
                break;
            }
            case NVCVOSDType::NVCV_OSD_ARROW:
            {
                cuosd_draw_arrow(context, n, *((NVCVArrow *)data));
                break;
            }
            case NVCVOSDType::NVCV_OSD_CLOCK:
            {
                cuosd_draw_clock(context, n, *((NVCVClock *)data));
                break;
            }
            default:
                break;
            }
        }
    }
    return ErrorCode::SUCCESS;
}

static ErrorCode cuosd_draw_bndbox(cuOSDContext_t context, int width, int height, NVCVBndBoxesImpl *bboxes)
{
    for (int n = 0; n < bboxes->batch(); n++)
    {
        auto numBoxes = bboxes->numBoxesAt(n);

        for (int i = 0; i < numBoxes; i++)
        {
            auto bbox = bboxes->boxAt(n, i);
            cuosd_draw_rectangle(context, n, width, height, bbox);
        }
    }
    return ErrorCode::SUCCESS;
}

OSD::OSD(DataShape max_input_shape, DataShape max_output_shape)
    : CudaBaseOp(max_input_shape, max_output_shape)
{
    m_context = new cuOSDContext();
}

OSD::~OSD()
{
    if (m_context)
    {
        cuOSDContext *p = (cuOSDContext *)m_context;
        delete p;
    }
}

ErrorCode OSD::infer(const nvcv::TensorDataStridedCuda &inData, const nvcv::TensorDataStridedCuda &outData,
                     NVCVElements elements, cudaStream_t stream)
{
    cuda_op::DataFormat input_format  = GetLegacyDataFormat(inData.layout());
    cuda_op::DataFormat output_format = GetLegacyDataFormat(outData.layout());

    if (!(input_format == kNHWC || input_format == kHWC) || !(output_format == kNHWC || output_format == kHWC))
    {
        LOG_ERROR("Invliad DataFormat both Input and Output must be kNHWC or kHWC");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    if (inData.dtype() != outData.dtype())
    {
        LOG_ERROR("Input and Output formats must be same input format =" << inData.dtype()
                                                                         << " output format = " << outData.dtype());
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(inData);
    if (!inAccess)
    {
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    int batch    = inAccess->numSamples();
    int channels = inAccess->numChannels();
    int rows     = inAccess->numRows();
    int cols     = inAccess->numCols();

    if (channels > 4 || channels < 1)
    {
        LOG_ERROR("Invalid channel number ch = " << channels);
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    NVCVElementsImpl *_elements = (NVCVElementsImpl *)elements;
    if (_elements->batch() != batch)
    {
        LOG_ERROR("Invalid elements batch = " << _elements->batch());
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    auto outAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(outData);
    if (!outAccess)
    {
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    cuda_op::DataType  inDataType = helpers::GetLegacyDataType(inData.dtype());
    cuda_op::DataShape inputShape = helpers::GetLegacyDataShape(inAccess->infoShape());

    cuda_op::DataType  outDataType = helpers::GetLegacyDataType(outData.dtype());
    cuda_op::DataShape outputShape = helpers::GetLegacyDataShape(outAccess->infoShape());

    if (outDataType != inDataType)
    {
        LOG_ERROR("Unsupported input/output DataType " << inDataType << "/" << outDataType);
        return ErrorCode::INVALID_DATA_TYPE;
    }

    if (outputShape.H != inputShape.H || outputShape.W != inputShape.W || outputShape.N != inputShape.N
        || outputShape.C != inputShape.C)
    {
        LOG_ERROR("Invalid input/output shape " << inputShape << "/" << outputShape);
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    auto ret = cuosd_draw_elements(m_context, cols, rows, _elements);
    if (ret != ErrorCode::SUCCESS)
    {
        LOG_ERROR("cuosd_draw_elements failed, ret - " << ret);
        return ret;
    }

    auto format = cuOSDImageFormat::RGBA;
    if (inputShape.C == 3)
        format = cuOSDImageFormat::RGB;

    cuosd_apply(m_context, inputShape.W, inputShape.H, format, stream);

    auto src     = nvcv::cuda::CreateTensorWrapNHWC<uint8_t>(inData);
    auto dst     = nvcv::cuda::CreateTensorWrapNHWC<uint8_t>(outData);
    bool inplace = inData.basePtr() == outData.basePtr();

    cuosd_launch(m_context, src, dst, inputShape.W, inputShape.C * inputShape.W, inputShape.H, format, inplace,
                 inputShape.N, stream);

    checkKernelErrors();

    cuosd_clear(m_context);

    return ErrorCode::SUCCESS;
}

ErrorCode OSD::inferBox(const nvcv::TensorDataStridedCuda &inData, const nvcv::TensorDataStridedCuda &outData,
                        NVCVBndBoxesI bboxes, cudaStream_t stream)
{
    cuda_op::DataFormat input_format  = GetLegacyDataFormat(inData.layout());
    cuda_op::DataFormat output_format = GetLegacyDataFormat(outData.layout());

    if (!(input_format == kNHWC || input_format == kHWC) || !(output_format == kNHWC || output_format == kHWC))
    {
        LOG_ERROR("Invliad DataFormat both Input and Output must be kNHWC or kHWC");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    if (inData.dtype() != outData.dtype())
    {
        LOG_ERROR("Input and Output formats must be same input format =" << inData.dtype()
                                                                         << " output format = " << outData.dtype());
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(inData);
    if (!inAccess)
    {
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    auto outAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(outData);
    if (!outAccess)
    {
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    cuda_op::DataShape inputShape  = helpers::GetLegacyDataShape(inAccess->infoShape());
    cuda_op::DataShape outputShape = helpers::GetLegacyDataShape(outAccess->infoShape());

    if (outputShape.H != inputShape.H || outputShape.W != inputShape.W || outputShape.N != inputShape.N
        || outputShape.C != inputShape.C)
    {
        LOG_ERROR("Invalid input/output shape " << inputShape << "/" << outputShape);
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    int batch    = inAccess->numSamples();
    int channels = inAccess->numChannels();
    int rows     = inAccess->numRows();
    int cols     = inAccess->numCols();

    if (channels > 4 || channels < 1)
    {
        LOG_ERROR("Invalid channel number ch = " << channels);
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    NVCVBndBoxesImpl *_bboxes = (NVCVBndBoxesImpl *)bboxes;
    if (_bboxes->batch() != batch)
    {
        LOG_ERROR("Invalid bboxes batch = " << _bboxes->batch());
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    auto ret = cuosd_draw_bndbox(m_context, cols, rows, _bboxes);
    if (ret != ErrorCode::SUCCESS)
    {
        LOG_ERROR("cuosd_draw_bndbox failed, ret - " << ret);
        return ret;
    }

    auto format = cuOSDImageFormat::RGBA;
    if (inputShape.C == 3)
        format = cuOSDImageFormat::RGB;

    cuosd_apply(m_context, inputShape.W, inputShape.H, format, stream);

    auto src     = nvcv::cuda::CreateTensorWrapNHWC<uint8_t>(inData);
    auto dst     = nvcv::cuda::CreateTensorWrapNHWC<uint8_t>(outData);
    bool inplace = inData.basePtr() == outData.basePtr();

    cuosd_launch(m_context, src, dst, inputShape.W, inputShape.C * inputShape.W, inputShape.H, format, inplace,
                 inputShape.N, stream);

    checkKernelErrors();

    cuosd_clear(m_context);

    return ErrorCode::SUCCESS;
}

} // namespace nvcv::legacy::cuda_op
