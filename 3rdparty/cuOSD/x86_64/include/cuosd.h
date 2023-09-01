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

#ifndef CUOSD_H
#define CUOSD_H

typedef struct
{
} cuOSDContext;

typedef cuOSDContext *cuOSDContext_t;

enum class cuOSDClockFormat : int
{
    None          = 0,
    YYMMDD_HHMMSS = 1,
    YYMMDD        = 2,
    HHMMSS        = 3
};

enum class cuOSDImageFormat : int
{
    None            = 0,
    RGB             = 1,
    RGBA            = 2,
    BlockLinearNV12 = 3,
    PitchLinearNV12 = 4
};

enum class cuOSDTextBackend : int
{
    None        = 0,
    PangoCairo  = 1,
    StbTrueType = 2
};

typedef struct _cuOSDColor
{
    unsigned char r;
    unsigned char g;
    unsigned char b;
    unsigned char a;
} cuOSDColor;

// cuosd_context_create: support online generate text bitmap with required font.
cuOSDContext_t cuosd_context_create();

// set context text rendering backend
void cuosd_set_text_backend(cuOSDContext_t context, cuOSDTextBackend text_backend);

// cuosd_context_destroy: deallocate all resource related to allocated cuOSD context
void cuosd_context_destroy(cuOSDContext_t context);

// cuosd_measure_text: API to get tight width, height and upper offset from the given text's tight bounding box
void cuosd_measure_text(cuOSDContext_t context, const char *utf8_text, int font_size, const char *font, int *width,
                        int *height, int *yoffset);

// cuosd_draw_text: draw utf8 text on given cuOSD context.
// x, y stands for left upper corner of the text's bounding box.
// bg_color stands for textbox background color in case alpha != 0
// Draw nothing if font_size <=0, font_size is scaled by 3 and clamped to 10 - 500 pixels by default
void cuosd_draw_text(cuOSDContext_t context, const char *utf8_text, int font_size, const char *font, int x, int y,
                     cuOSDColor border_color, cuOSDColor bg_color = {0, 0, 0, 0});

// cuosd_draw_clock: draw clock element on given cuOSD context.
// x, y stands for left upper corner of the text's bounding box. 3 clock formats are supported:
//      YYMMDD_HHMMSS, YYMMDD, HHMMSS
// Draw nothing if font_size <=0, font_size is scaled by 3 and clamped to 10 - 500 pixels by default
void cuosd_draw_clock(cuOSDContext_t context, cuOSDClockFormat format, long time, int font_size, const char *font,
                      int x, int y, cuOSDColor border_color, cuOSDColor bg_color = {0, 0, 0, 0});

// cuosd_draw_line: draw line element on given cuOSD context.
// x0, y0 stands for start point coordinate of the line, and x1, y1 stands for end point coordinate of the line.
void cuosd_draw_line(cuOSDContext_t context, int x0, int y0, int x1, int y1, int thickness, cuOSDColor color,
                     bool interpolation = true);

// cuosd_draw_arrow: draw arrow element on given cuOSD context.
// x0, y0 stands for start point coordinate of the arrow, and x1, y1 stands for end point coordinate of the arrow.
void cuosd_draw_arrow(cuOSDContext_t context, int x0, int y0, int x1, int y1, int arrow_size, int thickness,
                      cuOSDColor color, bool interpolation = false);

// cuosd_draw_point: draw point element on given cuOSD context.
// cx, cy stands for center point coordinate of the point.
void cuosd_draw_point(cuOSDContext_t context, int cx, int cy, int radius, cuOSDColor color);

// cuosd_draw_circle: draw circle element on given cuOSD context.
// cx, cy stands for center point coordinate of the circle.
// thickness stands for border width when thickness > 0; stands for filled mode when thickness = -1.
// bg_color stands for inner color inside hollow circle in case alpha != 0
void cuosd_draw_circle(cuOSDContext_t context, int cx, int cy, int radius, int thickness, cuOSDColor border_color,
                       cuOSDColor bg_color = {0, 0, 0, 0});

// cuosd_draw_rectangle: draw rectangle element on given cuOSD context.
// thickness stands for border width when thickness > 0; stands for filled mode when thickness = -1.
// bg_color stands for inner color inside hollow rectangle in case alpha != 0
void cuosd_draw_rectangle(cuOSDContext_t context, int left, int top, int right, int bottom, int thickness,
                          cuOSDColor border_color, cuOSDColor bg_color = {0, 0, 0, 0});

// cuosd_draw_boxblur: Mean filtering in the region of interest
// The region of interest is first scaled to 32x32, filtered, and then scaled to the region of interest by nearest neighbor interpolation
// It is executed by a separate kernel function that is independent from the other drawing functions
void cuosd_draw_boxblur(cuOSDContext_t context, int left, int top, int right, int bottom, int kernel_size = 7);

// cuosd_draw_rotationbox: draw rotated rectangle element on given cuOSD context.
// yaw: rotation angle from y-axis, clockwise +, unit in rad.
void cuosd_draw_rotationbox(cuOSDContext_t _context, int cx, int cy, int width, int height, float yaw, int thickness,
                            cuOSDColor border_color, bool interpolation = false, cuOSDColor bg_color = {0, 0, 0, 0});

// cuosd_draw_segmentmask: draw segmentation mask on given cuOSD context.
// d_seg: device pointer of segmentation mask, alpha in seg_color is ignored.
// thickness should > 0 for drawing border, threshold: Threshold for binarization
//   1. resize mask rect to object rect of given left, top, right, bottom.
//   2. set the alpha to 127 if mask value > threshold, else alpha = 0.
void cuosd_draw_segmentmask(cuOSDContext_t context, int left, int top, int right, int bottom, int thickness,
                            float *d_seg, int seg_width, int seg_height, float seg_threshold, cuOSDColor border_color,
                            cuOSDColor seg_color = {0, 0, 0, 0});

// cuosd_draw_polyline: draw polyline element on given cuOSD context.
// h_pts: host point of polyline points in { int x, int y } data format.
// d_pts: device point of polyline points, shall not be nullptr if fill_color.a != 0.
// n_pts: number of polyline points, thickness: polyline thickness.
// is_closed: if the end point shall be connected to start point.
// border_color: polyline color, fill_color: polyfill color.
void cuosd_draw_polyline(cuOSDContext_t context, int *h_pts, int *d_pts, int n_pts, int thickness, bool is_closed,
                         cuOSDColor border_color, bool interpolation = true, cuOSDColor fill_color = {0, 0, 0, 0});

// cuosd_draw_rgba_source: draw color from rgba source image on given cuOSD context.
//   1. resize incoming rgba source rect to target rect of given left, top, right, bottom.
//   2. blend incoming rgba src on target image rect in bilinear interpolation manner.
void cuosd_draw_rgba_source(cuOSDContext_t _context, int left, int top, int right, int bottom, void *d_src,
                            int src_width, int src_stride, int src_height);

// cuosd_draw_nv12_source: draw color from nv12 source image on given cuOSD context.
//   1. resize incoming nv12 source rect to target rect of given left, top, right, bottom.
//   2. blend incoming nv12 src on target image rect in bilinear interpolation manner.
//   note: use unified alpha and can support both PL and BL nv12 format.
void cuosd_draw_nv12_source(cuOSDContext_t context, int left, int top, int right, int bottom, void *d_src0,
                            void *d_src1, int src_width, int src_stride, int src_height, unsigned char alpha = 127,
                            bool block_linear = false);

// cuosd_apply: calculate bounding box of all elements and transfer drawing commands to GPU.
// If format is RGBA, data0 is RGBA buffer, and data1 must be nullptr.
// If format is BlockLinearNV12, data0 and data1 is cudaSurfaceObject_t for Luma(Y) plane and Chroma(UV) plane
// If format is PitchLinearNV12, data0 is Luma(Y) plane buffer, and data1 is Chroma(UV) plane buffer
void cuosd_apply(cuOSDContext_t context, void *data0, void *data1, int width, int stride, int height,
                 cuOSDImageFormat format, void *stream = nullptr, bool launch_and_clear = true);

// clear all pushed commands
void cuosd_clear(cuOSDContext_t context);

// cuosd_launch: launch drawing kernel in async manner.
// If format is RGBA, data0 is RGBA buffer, and data1 must be nullptr.
// If format is BlockLinearNV12, data0 and data1 is cudaSurfaceObject_t for Luma(Y) plane and Chroma(UV) plane
// If format is PitchLinearNV12, data0 is Luma(Y) plane buffer, and data1 is Chroma(UV) plane buffer
void cuosd_launch(cuOSDContext_t context, void *data0, void *data1, int width, int stride, int height,
                  cuOSDImageFormat format, void *stream = nullptr);

#endif // CUOSD_H
