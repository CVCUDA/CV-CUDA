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

#include "backend.hpp"

#ifdef ENABLE_TEXT_BACKEND_STB
#    include "stb.hpp"
#endif

#include <stdio.h>

#include <sstream>

const char *text_backend_type_name(TextBackendType backend)
{
    switch (backend)
    {
    case TextBackendType::StbTrueType:
        return "StbTrueType";
    default:
        return "Unknow";
    }
}

std::shared_ptr<TextBackend> create_text_backend(TextBackendType backend)
{
    switch (backend)
    {
#ifdef ENABLE_TEXT_BACKEND_STB
    case TextBackendType::StbTrueType:
        return create_stb_backend();
#endif

    default:
        printf("Unsupport text backend: %s\n", text_backend_type_name(backend));
        return nullptr;
    }
}

std::string concat_font_name_size(const char *name, int size)
{
    std::stringstream ss;
    ss << name;
    ss << " ";
    ss << size;
    return ss.str();
}
