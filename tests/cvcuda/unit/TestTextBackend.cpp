/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "Definitions.hpp"

#include <cvcuda/priv/Types.hpp>
#include <cvcuda/priv/legacy/textbackend/backend.hpp>

#include <array>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <tuple>
#include <vector>

std::string get_ttf_path_from_family_name(const char *_font_family, const std::vector<std::string> &search_paths);

namespace {

struct ScopedTempDir
{
    ScopedTempDir()
    {
        auto  dirTemplate = std::to_array("/tmp/cvcuda-textbackend-XXXXXX");
        char *dir         = ::mkdtemp(dirTemplate.data());
        if (dir != nullptr)
        {
            path = dir;
        }
    }

    ~ScopedTempDir()
    {
        std::error_code ec;
        if (!path.empty())
        {
            std::filesystem::remove_all(path, ec);
        }
    }

    ScopedTempDir(const ScopedTempDir &)            = delete;
    ScopedTempDir &operator=(const ScopedTempDir &) = delete;
    ScopedTempDir(ScopedTempDir &&)                 = delete;
    ScopedTempDir &operator=(ScopedTempDir &&)      = delete;

    std::filesystem::path path;
};

void TouchFile(const std::filesystem::path &path)
{
    std::ofstream file(path);
    file << "dummy";
}

} // namespace

TEST(TextBackend, StbMeasureTextMatchesRenderedAdvance)
{
    auto backend = create_text_backend(TextBackendType::StbTrueType);
    ASSERT_NE(backend, nullptr);

    constexpr int kFontSize = 60;
    auto          words     = backend->split_utf8("Hello");
    ASSERT_FALSE(words.empty());

    constexpr auto kDefaultFont = cvcuda::priv::DEFAULT_OSD_FONT;

    int measuredWidth                                        = 0;
    int measuredHeight                                       = 0;
    int measuredYOffset                                      = 0;
    std::tie(measuredWidth, measuredHeight, measuredYOffset) = backend->measure_text(words, kFontSize, kDefaultFont);
    ASSERT_GT(measuredWidth, 0);

    backend->add_build_text(words, kFontSize, kDefaultFont);
    backend->build_bitmap();

    auto glyphMap = backend->query(kDefaultFont, kFontSize);
    ASSERT_NE(glyphMap, nullptr);

    int renderedAdvance = 0;
    for (auto word : words)
    {
        auto meta = glyphMap->query(word);
        ASSERT_NE(meta, nullptr);

        if (meta->width() < 1 || meta->height() < 1)
            renderedAdvance += meta->xadvance(kFontSize, true);
        else
            renderedAdvance += meta->xadvance(kFontSize);
    }

    EXPECT_EQ(measuredWidth, renderedAdvance);
}

TEST(TextBackend, StbSplitUtf8KeepsSupplementaryCodePoint)
{
    auto backend = create_text_backend(TextBackendType::StbTrueType);
    ASSERT_NE(backend, nullptr);

    auto words = backend->split_utf8("\xF0\x9F\x98\x80");

    ASSERT_EQ(words.size(), 1);
    EXPECT_EQ(words[0], 0x1F600UL);
}

TEST(TextBackend, StbSplitUtf8RejectsInvalidSequences)
{
    auto backend = create_text_backend(TextBackendType::StbTrueType);
    ASSERT_NE(backend, nullptr);

    const std::array invalidUtf8 = {
        "\x80",
        "\xC0\x80",
        "\xC2\x41",
        "\xE0\x9F\x80",
        "\xED\xA0\x80",
        "\xE1\x41\x80",
        "\xE1\x80\x41",
        "\xF5\x80\x80\x80",
        "\xF0\x80\x80\x80",
        "\xF4\x90\x80\x80",
        "\xF1\x41\x80\x80",
        "\xF1\x80\x41\x80",
        "\xF1\x80\x80\x41",
    };

    for (const char *text : invalidUtf8)
    {
        EXPECT_TRUE(backend->split_utf8(text).empty());
    }
}

TEST(TextBackend, StbUninitializedAndUntrustedFontPathsFailSafely)
{
    auto backend = create_text_backend(TextBackendType::StbTrueType);
    ASSERT_NE(backend, nullptr);

    EXPECT_EQ(backend->query("not-built", 12), nullptr);
    EXPECT_EQ(backend->bitmap_device_pointer(), nullptr);
    backend->build_bitmap();
    EXPECT_EQ(backend->bitmap_device_pointer(), nullptr);

    ScopedTempDir tempDir;
    ASSERT_FALSE(tempDir.path.empty());
    auto untrustedFont = tempDir.path / "Untrusted.ttf";
    TouchFile(untrustedFont);

    auto words = backend->split_utf8("A");
    EXPECT_EQ(backend->measure_text(words, 12, untrustedFont.c_str()), std::make_tuple(-1, -1, -1));
}

TEST(TextBackend, StbFontFamilySearchContinuesAfterFirstNonEmptyDirectory)
{
    ScopedTempDir tempDir;
    ASSERT_FALSE(tempDir.path.empty());

    std::filesystem::path firstDir  = tempDir.path / "first";
    std::filesystem::path secondDir = tempDir.path / "second";
    std::filesystem::create_directories(firstDir);
    std::filesystem::create_directories(secondDir);

    TouchFile(firstDir / "OtherFont.ttf");
    TouchFile(secondDir / "TargetFont.ttf");

    std::string path = get_ttf_path_from_family_name("TargetFont", {firstDir.string(), secondDir.string()});

    EXPECT_EQ(path, (secondDir / "TargetFont.ttf").string());
}

TEST(TextBackend, StbFontFamilySearchHandlesEmptyAndFallbackCases)
{
    ScopedTempDir tempDir;
    ASSERT_FALSE(tempDir.path.empty());

    std::filesystem::path emptyDir = tempDir.path / "empty";
    std::filesystem::path fontDir  = tempDir.path / "fonts";
    std::filesystem::path otherDir = tempDir.path / "other";
    std::filesystem::create_directories(emptyDir);
    std::filesystem::create_directories(fontDir);
    std::filesystem::create_directories(otherDir);

    EXPECT_TRUE(get_ttf_path_from_family_name("Missing", {emptyDir.string()}).empty());

    TouchFile(fontDir / "x");
    TouchFile(fontDir / "DejaVuSansMono.ttf");

    auto fallback = (fontDir / "DejaVuSansMono.ttf").string();
    EXPECT_EQ(get_ttf_path_from_family_name("", {fontDir.string()}), fallback);
    EXPECT_EQ(get_ttf_path_from_family_name("Missing", {fontDir.string()}), fallback);
    EXPECT_EQ(get_ttf_path_from_family_name("DejaVuSansMono.ttf", {fontDir.string()}), fallback);

    TouchFile(otherDir / "Other.ttf");
    EXPECT_EQ(get_ttf_path_from_family_name("Missing", {otherDir.string()}), (otherDir / "Other.ttf").string());
}
