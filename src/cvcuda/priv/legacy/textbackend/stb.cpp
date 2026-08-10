/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#    include <limits.h>
#    include <stdarg.h>
#    include <sys/stat.h>
#    include <sys/types.h>
#    include <unistd.h>

#    include <algorithm>
#    include <array>
#    include <cstdint>
#    include <cstdlib>
#    include <fstream>
#    include <limits>
#    include <map>
#    include <memory>
#    include <stack>
#    include <string_view>
#    include <unordered_set>
#    include <vector>
#    define strtok_s strtok_r

#    define STB_TRUETYPE_IMPLEMENTATION

#    ifdef STB_TRUETYPE_IMPLEMENTATION
#        define STBTT_STATIC
#        include "stb_truetype.h"
#    endif

using namespace std;

static constexpr size_t kMaxTrustedFontFileBytes = 64U * 1024U * 1024U;

static const vector<string> &system_font_search_paths()
{
    static const vector<string> search_paths = {
        "/usr/share/fonts/truetype",   // Debian/Ubuntu
        "/usr/share/fonts/dejavu",     // RHEL/AlmaLinux
        "/usr/share/fonts/liberation", // RHEL/AlmaLinux alternative
        "/usr/local/share/fonts",
        "/usr/share/fonts" // Generic fallback (will search all subdirectories)
    };
    return search_paths;
}

static string canonical_path(string_view path)
{
    string                pathString(path);
    array<char, PATH_MAX> resolvedPath{};
    if (realpath(pathString.c_str(), resolvedPath.data()) == nullptr)
    {
        return "";
    }
    return string(resolvedPath.data());
}

static bool path_is_under_directory(string_view path, string_view directory)
{
    string root = canonical_path(directory);
    if (root.empty())
    {
        return false;
    }
    if (path == root)
    {
        return true;
    }
    if (root.back() != '/')
    {
        root += "/";
    }
    return path.compare(0, root.size(), root) == 0;
}

static bool is_trusted_font_path(string_view font_path)
{
    string resolvedPath = canonical_path(font_path);
    if (resolvedPath.empty())
    {
        return false;
    }

    const auto &searchPaths = system_font_search_paths();
    return any_of(searchPaths.begin(), searchPaths.end(),
                  [&](const string &root) { return path_is_under_directory(resolvedPath, root); });
}

static int stb_word_xadvance(int width, int height, float scale, int advance, bool empty = false)
{
    if (empty || width < 1 || height < 1)
    {
        return static_cast<int>(static_cast<float>(advance) * scale * 0.5f);
    }
    return static_cast<int>(static_cast<float>(width) + std::max(1.0f, static_cast<float>(advance) * scale / 5.0f));
}

class StbWordMeta : public WordMeta
{
public:
    int   x0;
    int   y0;
    int   x1;
    int   y1;
    int   advance;
    int   glyph;
    int   offset_x;
    float scale;

    int width() const override
    {
        return x1 - x0;
    }

    int height() const override
    {
        return y1 - y0;
    }

    int x_offset_on_bitmap() const override
    {
        return offset_x;
    }

    int xadvance(int font_size, bool empty) const override
    {
        (void)font_size;
        return stb_word_xadvance(width(), height(), this->scale, this->advance, empty);
    }

    StbWordMeta() = default;

    StbWordMeta(int x0, int y0, int x1, int y1, float scale, int advance, int glyph, int offset_x)
        : x0(x0)
        , y0(y0)
        , x1(x1)
        , y1(y1)
        , advance(advance)
        , glyph(glyph)
        , offset_x(offset_x)
        , scale(scale)
    {
    }
};

class StbWordMetaMapperImpl
    : public WordMetaMapper
    , public map<unsigned long int, StbWordMeta>
{
public:
    WordMeta *query(unsigned long int word) override
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

static bool suffix_match(string_view path, string_view suffix)
{
    if (path.size() < suffix.size())
    {
        return false;
    }
    if (suffix.empty())
    {
        return true;
    }
    return path.substr(path.size() - suffix.size()) == suffix;
}

static bool font_name_matches(string_view lower_name, string_view query_name)
{
    if (lower_name == query_name)
    {
        return true;
    }

    size_t dot_pos = lower_name.rfind('.');
    if (dot_pos == string::npos)
    {
        return false;
    }

    return lower_name.substr(0, dot_pos) == query_name;
}

static void warn_font_fallback(bool fallback, const char *requested_family, const string &fallback_name)
{
    if (fallback)
    {
        CUOSD_PRINT_W("Can not find any fonts to match %s, fallback to %s\n", requested_family, fallback_name.c_str());
    }
}

static bool is_current_or_parent_directory(const char *name)
{
    return strcmp(name, ".") == 0 || strcmp(name, "..") == 0;
}

struct DirectoryCloser
{
    void operator()(DIR *dir) const
    {
        if (dir != nullptr)
        {
            closedir(dir);
        }
    }
};

static void collect_files_from_directory(const string &search_path, const string &suffix, bool includeSubDirectory,
                                         stack<string> &pending_paths, vector<tuple<string, string>> &out)
{
    std::unique_ptr<DIR, DirectoryCloser> handle(opendir(search_path.c_str()));
    if (!handle)
    {
        return;
    }

    struct dirent *fileinfo = nullptr;
    while ((fileinfo = readdir(handle.get())) != nullptr)
    {
        if (is_current_or_parent_directory(fileinfo->d_name))
        {
            continue;
        }

        string      path = search_path + fileinfo->d_name;
        struct stat file_stat;
        if (lstat(path.c_str(), &file_stat) < 0)
        {
            continue;
        }

        if (S_ISDIR(file_stat.st_mode))
        {
            if (includeSubDirectory)
            {
                pending_paths.push(path + "/");
            }
            continue;
        }

        if (suffix_match(fileinfo->d_name, suffix))
        {
            out.emplace_back(path, fileinfo->d_name);
        }
    }
}

static vector<tuple<string, string>> find_files(const string &directory, const string &suffix, bool includeSubDirectory)
{
    string realpath = directory;
    if (realpath.empty())
    {
        realpath = "./";
    }

    if (char backchar = realpath.back(); backchar != '\\' && backchar != '/')
    {
        realpath += "/";
    }

    stack<string>                 pending_paths;
    vector<tuple<string, string>> out;
    pending_paths.push(realpath);

    while (!pending_paths.empty())
    {
        string search_path = pending_paths.top();
        pending_paths.pop();
        collect_files_from_directory(search_path, suffix, includeSubDirectory, pending_paths, out);
    }

    return out;
}

static void cuda_font_free(TrueTypeFontInternal *ptr)
{
    std::unique_ptr<TrueTypeFontInternal> font(ptr);
    (void)font;
}

static TrueTypeFontInternal *load_true_type_font(istream &infile, int file_size)
{
    auto output = std::make_unique<TrueTypeFontInternal>();
    output->data.resize(file_size);

    if (!infile.read((char *)output->data.data(), file_size).good())
    {
        CUOSD_PRINT_E("Failed to read %d bytes.\n", file_size);
        return nullptr;
    }

    stbtt_fontinfo *font   = &output->font;
    int             offset = stbtt_GetFontOffsetForIndex(output->data.data(), 0);
    if (offset < 0)
    {
        CUOSD_PRINT_E("Failed to find a TrueType font in file.\n");
        return nullptr;
    }
    if (int ret = stbtt_InitFont(font, output->data.data(), offset); ret == 0)
    {
        CUOSD_PRINT_E("Failed to init font, ret = %d.\n", ret);
        return nullptr;
    }
    return output.release();
}

string get_ttf_path_from_family_name(const char *_font_family, const vector<string> &search_paths);

string get_ttf_path_from_family_name(const char *_font_family, const vector<string> &search_paths)
{
    vector<tuple<string, string>> files;
    unordered_set<string>         seen_paths;
    for (const auto &search_path : search_paths)
    {
        auto found_files = find_files(search_path, ".ttf", true);
        for (const auto &file : found_files)
        {
            if (seen_paths.insert(get<0>(file)).second)
            {
                files.push_back(file);
            }
        }
    }

    if (files.empty())
        return "";

    string font_family = _font_family;
    if (font_family.empty())
        return get<0>(files[0]);
    std::transform(font_family.begin(), font_family.end(), font_family.begin(), ::tolower);

    vector<string> match_list{font_family, "dejavusansmono"};
    for (size_t imatch = 0; imatch < match_list.size(); ++imatch)
    {
        const string &query_name = match_list[imatch];
        for (const auto &file : files)
        {
            string path;
            string lower_name;
            string raw_name;
            tie(path, raw_name) = file;

            lower_name = raw_name;
            std::transform(raw_name.begin(), raw_name.end(), lower_name.begin(), ::tolower);

            if (font_name_matches(lower_name, query_name))
            {
                warn_font_fallback(imatch > 0, _font_family, raw_name);
                return path;
            }
        }
    }

    CUOSD_PRINT_W("Can not find any fonts to match %s, fallback to %s\n", _font_family, get<0>(files[0]).c_str());
    return get<0>(files[0]);
}

static string get_ttf_path_from_family_name(const char *_font_family)
{
    return get_ttf_path_from_family_name(_font_family, system_font_search_paths());
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

    if (!is_trusted_font_path(font_file_or_family))
    {
        CUOSD_PRINT_E("Refusing to parse untrusted font path: %s\n", font_file_or_family);
        return nullptr;
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
    if (file_size > kMaxTrustedFontFileBytes || file_size > static_cast<size_t>(std::numeric_limits<int>::max()))
    {
        CUOSD_PRINT_E("Invalid font file. File is too large. %s\n", font_file_or_family);
        return nullptr;
    }

    return load_true_type_font(infile, static_cast<int>(file_size));
}

static bool decode_utf8_one_byte(const unsigned char *&str, unsigned long int &codepoint)
{
    codepoint = *str++;
    return true;
}

static bool decode_utf8_two_bytes(const unsigned char *&str, unsigned long int &codepoint)
{
    if (*str < 0xc2)
    {
        return false;
    }

    unsigned int c = (*str++ & 0x1f) << 6;
    if ((*str & 0xc0) != 0x80)
    {
        return false;
    }

    codepoint = c + (*str++ & 0x3f);
    return true;
}

static bool decode_utf8_three_bytes(const unsigned char *&str, unsigned long int &codepoint)
{
    if (*str == 0xe0 && (str[1] < 0xa0 || str[1] > 0xbf))
    {
        return false;
    }
    if (*str == 0xed && str[1] > 0x9f)
    {
        return false; // str[1] < 0x80 is checked below
    }

    unsigned int c = (*str++ & 0x0f) << 12;
    if ((*str & 0xc0) != 0x80)
    {
        return false;
    }

    c += (*str++ & 0x3f) << 6;
    if ((*str & 0xc0) != 0x80)
    {
        return false;
    }

    codepoint = c + (*str++ & 0x3f);
    return true;
}

static bool decode_utf8_four_bytes(const unsigned char *&str, unsigned long int &codepoint)
{
    if (*str > 0xf4)
    {
        return false;
    }
    if (*str == 0xf0 && (str[1] < 0x90 || str[1] > 0xbf))
    {
        return false;
    }
    if (*str == 0xf4 && str[1] > 0x8f)
    {
        return false; // str[1] < 0x80 is checked below
    }

    unsigned int c = (*str++ & 0x07) << 18;
    if ((*str & 0xc0) != 0x80)
    {
        return false;
    }

    c += (*str++ & 0x3f) << 12;
    if ((*str & 0xc0) != 0x80)
    {
        return false;
    }

    c += (*str++ & 0x3f) << 6;
    if ((*str & 0xc0) != 0x80)
    {
        return false;
    }

    c += (*str++ & 0x3f);
    if ((c & 0xFFFFF800) == 0xD800)
    {
        return false;
    }

    codepoint = c;
    return true;
}

static bool decode_utf8_codepoint(const unsigned char *&str, unsigned long int &codepoint)
{
    if (!(*str & 0x80))
    {
        return decode_utf8_one_byte(str, codepoint);
    }
    if ((*str & 0xe0) == 0xc0)
    {
        return decode_utf8_two_bytes(str, codepoint);
    }
    if ((*str & 0xf0) == 0xe0)
    {
        return decode_utf8_three_bytes(str, codepoint);
    }
    if ((*str & 0xf8) == 0xf0)
    {
        return decode_utf8_four_bytes(str, codepoint);
    }

    return false;
}

struct FontNameAndSize
{
    string name;
    int    size;
};

static FontNameAndSize parse_font_name_and_size(const string &font_name_and_size)
{
    size_t sep_pos = font_name_and_size.rfind(' ');
    if (sep_pos == string::npos)
    {
        return {font_name_and_size, 0};
    }

    return {font_name_and_size.substr(0, sep_pos), std::atoi(font_name_and_size.c_str() + sep_pos + 1)};
}

class StbTrueTypeBackend : public TextBackend
{
private:
    unique_ptr<Memory<unsigned char>>                     text_bitmap;
    unique_ptr<Memory<unsigned char>>                     single_word_bitmap;
    map<string, StbWordMetaMapperImpl, less<>>            glyph_sets;
    map<string, vector<unsigned long int>, less<>>        build_use_textes;
    int                                                   text_bitmap_width  = 0;
    int                                                   text_bitmap_height = 0;
    int                                                   temp_size          = 0;
    map<string, shared_ptr<TrueTypeFontInternal>, less<>> font_map;
    bool                                                  has_new_text_need_build_bitmap = false;

public:
    StbTrueTypeBackend()
    {
        this->temp_size          = MAX_FONT_SIZE * 2;
        this->single_word_bitmap = std::make_unique<Memory<unsigned char>>();
        this->single_word_bitmap->alloc_or_resize_to(this->temp_size * this->temp_size);
        memset(this->single_word_bitmap->host(), 0, this->single_word_bitmap->bytes());
    }

    ~StbTrueTypeBackend() override = default;

    vector<unsigned long int> split_utf8(const char *utf8_text) override
    {
        vector<unsigned long int> output;
        output.reserve(std::char_traits<char>::length(utf8_text));

        const auto *str = reinterpret_cast<const unsigned char *>(utf8_text);
        while (*str)
        {
            unsigned long int codepoint = 0;
            if (!decode_utf8_codepoint(str, codepoint))
            {
                return {};
            }
            output.emplace_back(codepoint);
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

    struct GlyphMeasure
    {
        int   width;
        int   height;
        int   advance;
        int   y0;
        float scale;
    };

    bool measure_word(StbWordMetaMapperImpl &word_map, TrueTypeFontInternal *&font_ptr, const char *font_name,
                      unsigned long int word, unsigned int font_size, GlyphMeasure &measure)
    {
        if (auto iter = word_map.find(word); iter != word_map.end())
        {
            measure.width   = iter->second.x1 - iter->second.x0;
            measure.height  = iter->second.y1 - iter->second.y0;
            measure.advance = iter->second.advance;
            measure.scale   = iter->second.scale;
            measure.y0      = iter->second.y0;
            return true;
        }

        if (font_ptr == nullptr)
        {
            font_ptr = get_font(font_name);
            if (font_ptr == nullptr)
            {
                return false;
            }
        }

        auto pfont = &font_ptr->font;
        int  x0;
        int  x1;
        int  y1;
        int  glyph    = stbtt_FindGlyphIndex(pfont, static_cast<int>(word));
        measure.scale = stbtt_ScaleForPixelHeight(pfont, static_cast<float>(font_size));
        stbtt_GetGlyphHMetrics(pfont, glyph, &measure.advance, nullptr);
        stbtt_GetGlyphBitmapBoxSubpixel(pfont, glyph, measure.scale, measure.scale, 0, 0, &x0, &measure.y0, &x1, &y1);
        measure.width  = x1 - x0;
        measure.height = y1 - measure.y0;
        return true;
    }

    void add_missing_glyph(StbWordMetaMapperImpl &glyph_map, const stbtt_fontinfo *pfont, unsigned long int word,
                           int font_size) const
    {
        if (glyph_map.find(word) != glyph_map.end())
        {
            return;
        }

        int   x0;
        int   y0;
        int   x1;
        int   y1;
        int   advance;
        int   glyph = stbtt_FindGlyphIndex(pfont, static_cast<int>(word));
        float scale = stbtt_ScaleForPixelHeight(pfont, static_cast<float>(font_size));
        stbtt_GetGlyphHMetrics(pfont, glyph, &advance, nullptr);
        stbtt_GetGlyphBitmapBoxSubpixel(pfont, glyph, scale, scale, 0, 0, &x0, &y0, &x1, &y1);
        glyph_map.insert(make_pair(word, StbWordMeta(x0, y0, x1, y1, scale, advance, glyph, 0)));
    }

    void add_default_glyphs(StbWordMetaMapperImpl &glyph_map, const stbtt_fontinfo *pfont, int font_size) const
    {
        const char *default_words
            = R"stb(ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789:-&./&^%$#@!+=\[];,'"?` )stb";

        for (const char *pword = default_words; *pword; ++pword)
        {
            unsigned long int word = static_cast<unsigned int>(*pword) | (1UL << 32);
            add_missing_glyph(glyph_map, pfont, word, font_size);
        }
    }

    void add_pending_glyphs(const string &font_name_and_size, const vector<unsigned long int> &words)
    {
        auto &glyph_map = this->glyph_sets[font_name_and_size];
        auto  font_key  = parse_font_name_and_size(font_name_and_size);
        auto  font      = get_font(font_key.name.c_str());
        if (font == nullptr)
        {
            return;
        }

        auto pfont = &font->font;
        for (const auto &word : words)
        {
            add_missing_glyph(glyph_map, pfont, word, font_key.size);
        }
        add_default_glyphs(glyph_map, pfont, font_key.size);
    }

    std::tuple<int, int> compute_bitmap_size() const
    {
        int max_glyph_height  = 0;
        int total_glyph_width = 0;

        for (const auto &[fontNameAndSize, glyphMap] : this->glyph_sets)
        {
            for (const auto &[codePoint, glyph] : glyphMap)
            {
                int w            = glyph.x1 - glyph.x0;
                int h            = glyph.y1 - glyph.y0;
                max_glyph_height = std::max(max_glyph_height, h);
                total_glyph_width += w;
            }
        }

        return make_tuple(total_glyph_width, max_glyph_height);
    }

    void resize_text_bitmap(int width, int height)
    {
        if (this->text_bitmap == nullptr)
        {
            this->text_bitmap = std::make_unique<Memory<unsigned char>>();
        }

        this->text_bitmap_width  = width;
        this->text_bitmap_height = height;
        this->text_bitmap->alloc_or_resize_to(width * height);
        memset(this->text_bitmap->host(), 0, this->text_bitmap->bytes());
    }

    void rasterize_glyphs(const string &font_name_and_size, StbWordMetaMapperImpl &glyph_map, int &offset_x)
    {
        auto font_key = parse_font_name_and_size(font_name_and_size);
        auto font     = get_font(font_key.name.c_str());
        if (font == nullptr)
        {
            return;
        }

        auto pfont = &font->font;
        for (auto &[codePoint, glyph] : glyph_map)
        {
            int w = glyph.x1 - glyph.x0;
            int h = glyph.y1 - glyph.y0;
            if (w < 1 || h < 1)
            {
                continue;
            }

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

    std::tuple<int, int, int> measure_text(const std::vector<unsigned long int> &words, unsigned int font_size,
                                           const char *font_name) override
    {
        int                   draw_x        = 0;
        int                   min_y         = font_size;
        int                   max_b         = 0;
        auto                  font_and_size = concat_font_name_size(font_name, font_size);
        TrueTypeFontInternal *font_ptr      = nullptr;
        auto                 &word_map      = this->glyph_sets[font_and_size];
        for (const auto &word : words)
        {
            GlyphMeasure measure{};
            if (!measure_word(word_map, font_ptr, font_name, word, font_size, measure))
            {
                return make_tuple(-1, -1, -1);
            }

            draw_x += stb_word_xadvance(measure.width, measure.height, measure.scale, measure.advance);

            int y = font_size + measure.y0;
            min_y = min(min_y, y);
            max_b = max(max_b, y + measure.height);
        }
        return make_tuple(draw_x, max_b - min_y, min_y);
    }

    void add_build_text(const std::vector<unsigned long int> &words, unsigned int font_size, const char *font) override
    {
        auto  font_and_size = concat_font_name_size(font, font_size);
        auto &maps          = build_use_textes[font_and_size];
        auto &glyph_map     = this->glyph_sets[font_and_size];
        for (const auto &word : words)
        {
            if (glyph_map.find(word) != glyph_map.end())
                continue;
            maps.insert(maps.end(), word);
            has_new_text_need_build_bitmap = true;
        }
    }

    WordMetaMapper *query(const char *font, int font_size) override
    {
        auto font_and_size = concat_font_name_size(font, font_size);
        auto iter          = this->glyph_sets.find(font_and_size);
        if (iter == this->glyph_sets.end())
            return nullptr;
        return &iter->second;
    }

    void build_bitmap(cudaStream_t stream) override
    {
        // 1. collect all word shape.
        if (!has_new_text_need_build_bitmap)
        {
            has_new_text_need_build_bitmap = false;
            build_use_textes.clear();
            return;
        }

        for (const auto &[fontNameAndSize, texts] : build_use_textes)
        {
            add_pending_glyphs(fontNameAndSize, texts);
        }

        auto [total_glyph_width, max_glyph_height] = compute_bitmap_size();
        resize_text_bitmap(total_glyph_width, max_glyph_height);

        // Rasterize word to bitmap
        int offset_x = 0;
        for (auto &[fontNameAndSize, glyphMap] : this->glyph_sets)
        {
            rasterize_glyphs(fontNameAndSize, glyphMap, offset_x);
        }
        this->text_bitmap->copy_host_to_device(stream);
        this->has_new_text_need_build_bitmap = false;
        this->build_use_textes.clear();
    }

    unsigned char *bitmap_device_pointer() const override
    {
        if (!this->text_bitmap)
            return nullptr;
        return this->text_bitmap->device();
    }

    int bitmap_width() const override
    {
        return this->text_bitmap_width;
    }

    int compute_y_offset(int max_glyph_height, int h, WordMeta *word, int font_size) const override
    {
        (void)max_glyph_height;
        (void)h;
        return font_size + ((StbWordMeta *)word)->y0;
    }

    int uniform_font_size(int size) const override
    {
        return size * 3;
    }
};

std::shared_ptr<TextBackend> create_stb_backend()
{
    return std::make_shared<StbTrueTypeBackend>();
}

#endif // ENABLE_TEXT_BACKEND_STB
