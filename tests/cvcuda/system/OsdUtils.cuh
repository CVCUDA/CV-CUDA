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

#ifndef NVCV_TEST_COMMON_OSD_UTILS_HPP
#define NVCV_TEST_COMMON_OSD_UTILS_HPP

#include "cuosd.h"

namespace nvcv::test { namespace osd {

enum class ImageFormat : int
{
    None            = 0,
    RGB             = 1,
    RGBA            = 2,
    BlockLinearNV12 = 3,
    PitchLinearNV12 = 4
};

struct Image
{
    void       *data0    = nullptr;
    void       *data1    = nullptr;
    void       *reserve0 = nullptr;
    void       *reserve1 = nullptr;
    int         width    = 0;
    int         height   = 0;
    int         stride   = 0;
    ImageFormat format   = ImageFormat::None;
};

// Get name of enumerate type
const char *image_format_name(ImageFormat format);

// Create gpu image using size and format
Image *create_image(int width, int height, ImageFormat format);

// Set image color
void set_color(Image *image, unsigned char r, unsigned char g, unsigned char b, unsigned char a = 255,
               void *_stream = nullptr);

// Free image pointer
void free_image(Image *image);

void cuosd_apply(cuOSDContext_t context, Image *image, void *_stream, bool launch = true);

void cuosd_launch(cuOSDContext_t context, Image *image, void *_stream);
}} // namespace nvcv::test::osd

#endif // NVCV_TEST_COMMON_OSD_UTILS_HPP
