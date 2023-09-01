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

#ifndef TEXT_BACKEND_HPP
#define TEXT_BACKEND_HPP

#include <memory>
#include <tuple>
#include <vector>

#define MAX_FONT_SIZE 200

enum class TextBackendType : int
{
    None        = 0,
    StbTrueType = 1
};

class WordMeta
{
public:
    virtual int width() const                                     = 0;
    virtual int height() const                                    = 0;
    virtual int x_offset_on_bitmap() const                        = 0;
    virtual int xadvance(int font_size, bool empty = false) const = 0;
};

class WordMetaMapper
{
public:
    virtual WordMeta *query(unsigned long int word) = 0;
};

class TextBackend
{
public:
    virtual std::vector<unsigned long int> split_utf8(const char *utf8_text) = 0;
    virtual std::tuple<int, int, int> measure_text(const std::vector<unsigned long int> &words, unsigned int font_size,
                                                   const char *font)
        = 0;
    virtual void add_build_text(const std::vector<unsigned long int> &words, unsigned int font_size, const char *font)
        = 0;
    virtual void            build_bitmap(void *stream = nullptr)                                               = 0;
    virtual WordMetaMapper *query(const char *font, int font_size)                                             = 0;
    virtual unsigned char  *bitmap_device_pointer() const                                                      = 0;
    virtual int             bitmap_width() const                                                               = 0;
    virtual int             compute_y_offset(int max_glyph_height, int h, WordMeta *word, int font_size) const = 0;
    virtual int             uniform_font_size(int size) const                                                  = 0;
};

const char                  *text_backend_type_name(TextBackendType backend);
std::shared_ptr<TextBackend> create_text_backend(TextBackendType backend);
std::string                  concat_font_name_size(const char *name, int size);

#endif // TEXT_BACKEND_HPP
