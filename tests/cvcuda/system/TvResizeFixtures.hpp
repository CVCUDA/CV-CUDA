/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef CVCUDA_TEST_TVRESIZE_FIXTURES_HPP
#define CVCUDA_TEST_TVRESIZE_FIXTURES_HPP

#include <cuda_fp16.h>
#include <cvcuda/Types.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace tvfixture {

class ExportError : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

template<class T>
void Write(const std::string &path, const T *data, size_t size)
{
    std::string bytes(size, '\0');
    std::memcpy(bytes.data(), data, size);
    if (std::ifstream existing(path, std::ios::binary); existing)
    {
        if (const std::string existingBytes{std::istreambuf_iterator<char>(existing), std::istreambuf_iterator<char>()};
            existingBytes != bytes)
            throw ExportError("Conflicting TvResize fixture key: " + path);
        return;
    }
    std::ofstream file(path, std::ios::binary);
    file.exceptions(std::ios::failbit | std::ios::badbit);
    file.write(bytes.data(), size);
    file.close();
}

template<class T>
const char *DTypeName()
{
    if constexpr (std::is_same_v<T, uint8_t>)
        return "u8";
    else if constexpr (std::is_same_v<T, uint16_t>)
        return "u16";
    else if constexpr (std::is_same_v<T, int16_t>)
        return "s16";
    else if constexpr (std::is_same_v<T, __half>)
        return "f16";
    else
        return "f32";
}

inline const char *ModeName(NVCVInterpolationType mode)
{
    switch (mode)
    {
    case NVCV_INTERP_NEAREST:
        return "NEAREST";
    case NVCV_INTERP_NEAREST_FLOOR:
        return "NEAREST_FLOOR";
    case NVCV_INTERP_LINEAR:
        return "LINEAR";
    default:
        return "CUBIC";
    }
}

template<class T>
void Export(const char *layout, int channels, int sample, int batches, bool padded, int sh, int sw, int dh, int dw,
            NVCVInterpolationType mode, bool aa, int pattern, const std::vector<T> &input, const std::vector<T> &gold,
            const std::vector<T> &actual)
{
    const char *directory = std::getenv("CVCUDA_TVRESIZE_FIXTURE_DIR");
    if (!directory)
        return;
    if (const uint16_t endian = 1; *reinterpret_cast<const std::byte *>(&endian) != std::byte{1})
        throw ExportError("TvResize fixture archives require little-endian storage");
    const char        *dtype    = DTypeName<T>();
    const char        *modeName = ModeName(mode);
    std::ostringstream key;
    key << directory << '/' << dtype << '-' << layout << "-c" << channels << "-b" << sample << "-n" << batches << "-pad"
        << padded << '-' << sh << 'x' << sw << '-' << dh << 'x' << dw << '-' << modeName << "-aa" << aa << "-p"
        << pattern;
    const auto path = key.str();
    Write(path + ".input.bin", input.data(), input.size() * sizeof(T));
    Write(path + ".gold.bin", gold.data(), gold.size() * sizeof(T));
    Write(path + ".actual.bin", actual.data(), actual.size() * sizeof(T));
    std::ostringstream metadata;
    metadata << R"({"dtype":")" << dtype << R"(","mode":")" << modeName << R"(","aa":)" << (aa ? "true" : "false")
             << R"(,"exact":)" << (sh == dh && sw == dw ? "true" : "false") << R"(,"layout":")" << layout
             << R"(","channels":)" << channels << R"(,"sample":)" << sample << R"(,"sh":)" << sh << R"(,"sw":)" << sw
             << R"(,"dh":)" << dh << R"(,"dw":)" << dw << R"(,"pattern":)" << pattern << "}\n";
    const auto json = metadata.str();
    Write(path + ".json", json.data(), json.size());
}

} // namespace tvfixture

#endif // CVCUDA_TEST_TVRESIZE_FIXTURES_HPP
