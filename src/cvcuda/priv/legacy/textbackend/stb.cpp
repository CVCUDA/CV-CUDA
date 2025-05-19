/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "stb.hpp"

#ifdef ENABLE_TEXT_BACKEND_STB

#    include "memory.hpp"

#    include <dirent.h>
#    include <stdarg.h>
#    include <sys/stat.h>
#    include <sys/types.h>
#    include <unistd.h>

#    include <algorithm>
#    include <fstream>
#    include <map>
#    include <stack>
#    define strtok_s strtok_r

#    define STB_TRUETYPE_IMPLEMENTATION

#    ifdef STB_TRUETYPE_IMPLEMENTATION
#        define STBTT_STATIC
#        include "stb_truetype.h"
#    endif

using namespace std;

class StbWordMeta : public WordMeta
{
public:
    int   x0, y0, x1, y1, advance, glyph, offset_x;
    float scale;

    virtual int width() const override
    {
        return x1 - x0;
    }

    virtual int height() const override
    {
        return y1 - y0;
    }

    virtual int x_offset_on_bitmap() const override
    {
        return offset_x;
    }

    virtual int xadvance(int font_size, bool empty) const override
    {
        (void)font_size;
        if (empty)
        {
            return this->advance * this->scale * 0.5;
        }
        return width() + std::max(1.0f, this->advance * this->scale / 5.0f);
    }

    StbWordMeta() = default;

    StbWordMeta(int x0, int y0, int x1, int y1, float scale, int advance, int glyph, int offset_x)
    {
        this->x0       = x0;
        this->y0       = y0;
        this->x1       = x1;
        this->y1       = y1;
        this->scale    = scale;
        this->advance  = advance;
        this->glyph    = glyph;
        this->offset_x = offset_x;
    }
};

class StbWordMetaMapperImpl
    : public WordMetaMapper
    , public map<unsigned long int, StbWordMeta>
{
public:
    virtual WordMeta *query(unsigned long int word) override
    {
        auto iter = this->find(word);
        if (iter == this->end())
            return nullptr;
        return &iter->second;
    }
};

struct TrueTypeFontInternal
{
    stbtt_fontinfo  font;
    vector<uint8_t> data;
};

static bool file_exist(const string &path)
{
    return access(path.c_str(), R_OK) == 0;
}

static vector<tuple<string, string>> find_files(const string &directory, const string &suffix, bool includeSubDirectory)
{
    string realpath = directory;
    if (realpath.empty())
        realpath = "./";

    char backchar = realpath.back();
    if (backchar not_eq '\\' and backchar not_eq '/')
        realpath += "/";

    struct dirent                *fileinfo;
    DIR                          *handle;
    stack<string>                 ps;
    vector<tuple<string, string>> out;
    ps.push(realpath);

    auto suffix_match = [&](const string &path)
    {
        if (path.size() < suffix.size())
            return false;
        if (suffix.empty())
            return true;
        return path.substr(path.size() - suffix.size()) == suffix;
    };

    while (!ps.empty())
    {
        string search_path = ps.top();
        ps.pop();

        handle = opendir(search_path.c_str());
        if (handle)
        {
            while (true)
            {
                fileinfo = readdir(handle);
                if (fileinfo == nullptr)
                    break;

                struct stat file_stat;
                if (strcmp(fileinfo->d_name, ".") == 0 or strcmp(fileinfo->d_name, "..") == 0)
                    continue;

                if (lstat((search_path + fileinfo->d_name).c_str(), &file_stat) < 0)
                    continue;

                if (!S_ISDIR(file_stat.st_mode))
                {
                    if (suffix_match(fileinfo->d_name))
                        out.push_back(make_tuple(search_path + fileinfo->d_name, fileinfo->d_name));
                }

                if (includeSubDirectory && S_ISDIR(file_stat.st_mode))
                    ps.push(search_path + fileinfo->d_name + "/");
            }
            closedir(handle);
        }
    }
    return out;
}

static void cuda_font_free(TrueTypeFontInternal *ptr)
{
    if (ptr)
        delete ptr;
}

static TrueTypeFontInternal *load_true_type_font(istream &infile, int file_size)
{
    TrueTypeFontInternal *output = new TrueTypeFontInternal();
    output->data.resize(file_size);

    if (!infile.read((char *)output->data.data(), file_size).good())
    {
        cuda_font_free(output);
        CUOSD_PRINT_E("Failed to read %d bytes.\n", file_size);
        return nullptr;
    }

    stbtt_fontinfo *font   = &output->font;
    int             offset = stbtt_GetFontOffsetForIndex(output->data.data(), 0);
    int             ret    = stbtt_InitFont(font, output->data.data(), offset);
    if (ret == 0)
    {
        cuda_font_free(output);
        CUOSD_PRINT_E("Failed to init font, ret = %d.\n", ret);
        return nullptr;
    }
    return output;
}

static string get_ttf_path_from_family_name(const char *_font_family)
{
    auto files = find_files("/usr/share/fonts/truetype", ".ttf", true);
    if (files.empty())
        return "";

    string font_family = _font_family;
    if (font_family.empty())
        return get<0>(files[0]);
    std::transform(font_family.begin(), font_family.end(), font_family.begin(), ::tolower);

    vector<string> match_list{font_family, "dejavusansmono"};
    for (size_t imatch = 0; imatch < match_list.size(); ++imatch)
    {
        auto query_name = match_list[imatch];
        for (auto file : files)
        {
            string path, lower_name, raw_name;
            tie(path, raw_name) = file;

            lower_name = raw_name;
            std::transform(raw_name.begin(), raw_name.end(), lower_name.begin(), ::tolower);

            bool matched = lower_name == query_name;
            if (!matched)
            {
                int p = lower_name.rfind(".");
                if (p != -1)
                    matched = lower_name.substr(0, p) == query_name;
            }
            if (matched)
            {
                if (imatch > 0)
                {
                    CUOSD_PRINT_W("Can not find any fonts to match %s, fallback to %s\n", _font_family,
                                  raw_name.c_str());
                }
                return path;
            }
        }
    }

    CUOSD_PRINT_W("Can not find any fonts to match %s, fallback to %s\n", _font_family, get<0>(files[0]).c_str());
    return get<0>(files[0]);
}

static TrueTypeFontInternal *create_cuda_font(const char *font_file_or_family);

static TrueTypeFontInternal *load_true_type_from_family_name(const char *_font_family)
{
    string ttf = get_ttf_path_from_family_name(_font_family);
    if (ttf.empty())
    {
        CUOSD_PRINT_E("Can not find any fonts to match %s\n", _font_family);
        return nullptr;
    }
    return create_cuda_font(ttf.c_str());
}

static TrueTypeFontInternal *create_cuda_font(const char *font_file_or_family)
{
    if (!file_exist(font_file_or_family))
    {
        // is font family
        return load_true_type_from_family_name(font_file_or_family);
    }

    fstream infile(font_file_or_family, ios::binary | ios::in);
    if (!infile)
    {
        CUOSD_PRINT_E("Failed to open: %s\n", font_file_or_family);
        return nullptr;
    }
    infile.seekg(0, ios::end);

    size_t file_size = infile.tellg();
    if (file_size < 12)
    {
        CUOSD_PRINT_E("Invalid font file. %s\n", font_file_or_family);
        return nullptr;
    }

    infile.seekg(0, ios::beg);
    return load_true_type_font(infile, file_size);
}

class StbTrueTypeBackend : public TextBackend
{
private:
    unique_ptr<Memory<unsigned char>>             text_bitmap;
    unique_ptr<Memory<unsigned char>>             single_word_bitmap;
    map<string, StbWordMetaMapperImpl>            glyph_sets;
    map<string, vector<unsigned long int>>        build_use_textes;
    int                                           text_bitmap_width  = 0;
    int                                           text_bitmap_height = 0;
    int                                           temp_size          = 0;
    map<string, shared_ptr<TrueTypeFontInternal>> font_map;
    bool                                          has_new_text_need_build_bitmap = false;

public:
    StbTrueTypeBackend()
    {
        int temp_size   = MAX_FONT_SIZE * 2;
        this->temp_size = temp_size;
        this->single_word_bitmap.reset(new Memory<unsigned char>);
        this->single_word_bitmap->alloc_or_resize_to(temp_size * temp_size);
        memset(this->single_word_bitmap->host(), 0, this->single_word_bitmap->bytes());
    }

    virtual ~StbTrueTypeBackend() {}

    virtual vector<unsigned long int> split_utf8(const char *utf8_text) override
    {
        vector<unsigned long int> output;
        output.reserve(std::char_traits<char>::length(utf8_text));

        unsigned char *str = (unsigned char *)utf8_text;
        unsigned int   c;
        while (*str)
        {
            if (!(*str & 0x80))
                output.emplace_back(*str++);
            else if ((*str & 0xe0) == 0xc0)
            {
                if (*str < 0xc2)
                    return {};
                c = (*str++ & 0x1f) << 6;
                if ((*str & 0xc0) != 0x80)
                    return {};
                output.emplace_back(c + (*str++ & 0x3f));
            }
            else if ((*str & 0xf0) == 0xe0)
            {
                if (*str == 0xe0 && (str[1] < 0xa0 || str[1] > 0xbf))
                    return {};
                if (*str == 0xed && str[1] > 0x9f)
                    return {}; // str[1] < 0x80 is checked below
                c = (*str++ & 0x0f) << 12;
                if ((*str & 0xc0) != 0x80)
                    return {};
                c += (*str++ & 0x3f) << 6;
                if ((*str & 0xc0) != 0x80)
                    return {};
                output.emplace_back(c + (*str++ & 0x3f));
            }
            else if ((*str & 0xf8) == 0xf0)
            {
                if (*str > 0xf4)
                    return {};
                if (*str == 0xf0 && (str[1] < 0x90 || str[1] > 0xbf))
                    return {};
                if (*str == 0xf4 && str[1] > 0x8f)
                    return {}; // str[1] < 0x80 is checked below
                c = (*str++ & 0x07) << 18;
                if ((*str & 0xc0) != 0x80)
                    return {};
                c += (*str++ & 0x3f) << 12;
                if ((*str & 0xc0) != 0x80)
                    return {};
                c += (*str++ & 0x3f) << 6;
                if ((*str & 0xc0) != 0x80)
                    return {};
                c += (*str++ & 0x3f);
                // utf-8 encodings of values used in surrogate pairs are invalid
                if ((c & 0xFFFFF800) == 0xD800)
                    return {};
                if (c >= 0x10000)
                {
                    c -= 0x10000;
                    output.emplace_back(0xD800 | (0x3ff & (c >> 10)));
                    output.emplace_back(0xDC00 | (0x3ff & (c)));
                }
            }
            else
                return {};
        }
        return output;
    }

    virtual TrueTypeFontInternal *get_font(const char *font_name)
    {
        auto &font = this->font_map[font_name];
        if (font == nullptr)
        {
            font.reset(create_cuda_font(font_name), cuda_font_free);
        }
        return font.get();
    }

    virtual std::tuple<int, int, int> measure_text(const std::vector<unsigned long int> &words, unsigned int font_size,
                                                   const char *font_name) override
    {
        int                   draw_x        = 0;
        int                   xadvance      = font_size * 0.1;
        int                   min_y         = font_size;
        int                   max_b         = 0;
        auto                  font_and_size = concat_font_name_size(font_name, font_size);
        TrueTypeFontInternal *font_ptr      = nullptr;
        auto                 &word_map      = this->glyph_sets[font_and_size];
        for (auto &word : words)
        {
            int   w, h;
            int   advance = 0;
            float scale   = 0;
            int   y0      = 0;
            auto  iter    = word_map.find(word);
            if (iter == word_map.end())
            {
                if (font_ptr == nullptr)
                {
                    font_ptr = get_font(font_name);
                    if (font_ptr == nullptr)
                        return make_tuple(-1, -1, -1);
                }

                auto pfont = &font_ptr->font;
                int  x0, x1, y1;
                int  glyph = stbtt_FindGlyphIndex(pfont, word);
                scale      = stbtt_ScaleForPixelHeight(pfont, font_size);
                stbtt_GetGlyphHMetrics(pfont, glyph, &advance, nullptr);
                stbtt_GetGlyphBitmapBoxSubpixel(pfont, glyph, scale, scale, 0, 0, &x0, &y0, &x1, &y1);

                w = x1 - x0;
                h = y1 - y0;
            }
            else
            {
                w       = iter->second.x1 - iter->second.x0;
                h       = iter->second.y1 - iter->second.y0;
                advance = iter->second.advance;
                scale   = iter->second.scale;
                y0      = iter->second.y0;
            }

            if (w < 1 || h < 1)
            {
                draw_x += advance * scale * 0.5;
            }
            else
            {
                draw_x += w + std::max(1.0f, advance * scale / 5.0f);
            }

            int y = font_size + y0;
            min_y = min(min_y, y);
            max_b = max(max_b, y + h);
        }
        draw_x -= xadvance;
        return make_tuple(draw_x, max_b - min_y, min_y);
    }

    virtual void add_build_text(const std::vector<unsigned long int> &words, unsigned int font_size,
                                const char *font) override
    {
        auto  font_and_size = concat_font_name_size(font, font_size);
        auto &maps          = build_use_textes[font_and_size];
        auto &glyph_map     = this->glyph_sets[font_and_size];
        for (auto &word : words)
        {
            if (glyph_map.find(word) != glyph_map.end())
                continue;
            maps.insert(maps.end(), word);
            has_new_text_need_build_bitmap = true;
        }
    }

    virtual WordMetaMapper *query(const char *font, int font_size) override
    {
        auto font_and_size = concat_font_name_size(font, font_size);
        auto iter          = this->glyph_sets.find(font_and_size);
        if (iter == this->glyph_sets.end())
            return nullptr;
        return &iter->second;
    }

    virtual void build_bitmap(void *_stream) override
    {
        cudaStream_t stream = (cudaStream_t)_stream;

        // 1. collect all word shape.
        if (!has_new_text_need_build_bitmap)
        {
            has_new_text_need_build_bitmap = false;
            build_use_textes.clear();
            return;
        }

        for (auto &textes : build_use_textes)
        {
            auto  &glyph_map          = this->glyph_sets[textes.first];
            auto  &words              = textes.second;
            string font_name_and_size = textes.first;
            int    p                  = font_name_and_size.rfind(' ');
            font_name_and_size[p]     = 0;
            int         font_size     = std::atoi(font_name_and_size.c_str() + p + 1);
            const char *font_name     = font_name_and_size.c_str();
            auto        font          = get_font(font_name);
            if (font == nullptr)
                continue;

            auto pfont = &font->font;
            int  x0, y0, x1, y1, advance;

            for (auto &word : words)
            {
                if (glyph_map.find(word) != glyph_map.end())
                    continue;

                int   glyph = stbtt_FindGlyphIndex(pfont, word);
                float scale = stbtt_ScaleForPixelHeight(pfont, font_size);
                stbtt_GetGlyphHMetrics(pfont, glyph, &advance, nullptr);
                stbtt_GetGlyphBitmapBoxSubpixel(pfont, glyph, scale, scale, 0, 0, &x0, &y0, &x1, &y1);
                glyph_map.insert(make_pair(word, StbWordMeta(x0, y0, x1, y1, scale, advance, glyph, 0)));
            }

            const char *default_words
                = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789:-&./&^%$#@!+=\\[];,'\"?` ";
            const char *pword = default_words;
            for (; *pword; ++pword)
            {
                unsigned long int word = (unsigned int)*pword | (1ul << 32);
                if (glyph_map.find(word) != glyph_map.end())
                    continue;

                int   glyph = stbtt_FindGlyphIndex(pfont, word);
                float scale = stbtt_ScaleForPixelHeight(pfont, font_size);
                stbtt_GetGlyphHMetrics(pfont, glyph, &advance, nullptr);
                stbtt_GetGlyphBitmapBoxSubpixel(pfont, glyph, scale, scale, 0, 0, &x0, &y0, &x1, &y1);
                glyph_map.insert(make_pair(word, StbWordMeta(x0, y0, x1, y1, scale, advance, glyph, 0)));
            }
        }

        int max_glyph_height  = 0;
        int max_glyph_width   = 0;
        int total_glyph_width = 0;
        for (auto &map : this->glyph_sets)
        {
            for (auto &item : map.second)
            {
                int w = item.second.x1 - item.second.x0, h = item.second.y1 - item.second.y0;
                max_glyph_width  = std::max(max_glyph_width, w);
                max_glyph_height = std::max(max_glyph_height, h);
                total_glyph_width += w;
            }
        }

        if (this->text_bitmap == nullptr)
            this->text_bitmap.reset(new Memory<unsigned char>());
        this->text_bitmap_width  = total_glyph_width;
        this->text_bitmap_height = max_glyph_height;
        this->text_bitmap->alloc_or_resize_to(total_glyph_width * max_glyph_height);
        memset(this->text_bitmap->host(), 0, this->text_bitmap->bytes());

        // Rasterize word to bitmap
        int offset_x = 0;
        for (auto &map : this->glyph_sets)
        {
            string font_name_and_size = map.first;
            int    p                  = font_name_and_size.rfind(' ');
            font_name_and_size[p]     = 0;
            const char *font_name     = font_name_and_size.c_str();
            auto        font          = get_font(font_name);
            if (font == nullptr)
                continue;

            auto pfont = &font->font;
            for (auto &item : map.second)
            {
                auto &glyph = item.second;

                int w = glyph.x1 - glyph.x0, h = glyph.y1 - glyph.y0;
                if (w < 1 || h < 1)
                    continue;

                glyph.offset_x          = offset_x;
                stbtt_vertex *vertices  = nullptr;
                int           num_verts = stbtt_GetGlyphShape(pfont, glyph.glyph, &vertices);
                stbtt__bitmap gbm;
                gbm.pixels = this->text_bitmap->host() + offset_x;
                gbm.w      = w;
                gbm.h      = h;
                gbm.stride = this->text_bitmap_width;
                stbtt_Rasterize(&gbm, 0.35f, vertices, num_verts, glyph.scale, glyph.scale, 0, 0, glyph.x0, glyph.y0, 1,
                                pfont->userdata);
                STBTT_free(vertices, pfont->userdata);
                offset_x += w;
            }
        }
        this->text_bitmap->copy_host_to_device(stream);
        this->has_new_text_need_build_bitmap = false;
        this->build_use_textes.clear();
    }

    virtual unsigned char *bitmap_device_pointer() const override
    {
        if (!this->text_bitmap)
            return nullptr;
        return this->text_bitmap->device();
    }

    virtual int bitmap_width() const override
    {
        return this->text_bitmap_width;
    }

    virtual int compute_y_offset(int max_glyph_height, int h, WordMeta *word, int font_size) const override
    {
        (void)max_glyph_height;
        (void)h;
        return font_size + ((StbWordMeta *)word)->y0;
    }

    virtual int uniform_font_size(int size) const override
    {
        return size * 3;
    }
};

std::shared_ptr<TextBackend> create_stb_backend()
{
    return std::make_shared<StbTrueTypeBackend>();
}

#endif // ENABLE_TEXT_BACKEND_STB
