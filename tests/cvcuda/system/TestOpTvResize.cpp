/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "Definitions.hpp"
#include "TvResizeFixtures.hpp"
#include "TvResizeGold.hpp"

#include <common/TypedTests.hpp>
#include <cvcuda/OpReformat.hpp>
#include <cvcuda/OpTvResize.hpp>
#include <cvcuda/Workspace.hpp>
#include <gtest/gtest.h>
#include <nvcv/Image.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorData.hpp>

#include <array>
#include <climits>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <tuple>

namespace {

struct Family
{
    NVCVInterpolationType mode;
    bool                  aa;
};

constexpr std::array<Family, 6> kFamilies{
    {{NVCV_INTERP_NEAREST, false},
     {NVCV_INTERP_NEAREST_FLOOR, false},
     {NVCV_INTERP_LINEAR, false},
     {NVCV_INTERP_CUBIC, false},
     {NVCV_INTERP_LINEAR, true},
     {NVCV_INTERP_CUBIC, true}}
};

constexpr std::array<Family, 8> kQueryFamilies{
    {{NVCV_INTERP_NEAREST, false},
     {NVCV_INTERP_NEAREST, true},
     {NVCV_INTERP_NEAREST_FLOOR, false},
     {NVCV_INTERP_NEAREST_FLOOR, true},
     {NVCV_INTERP_LINEAR, false},
     {NVCV_INTERP_LINEAR, true},
     {NVCV_INTERP_CUBIC, false},
     {NVCV_INTERP_CUBIC, true}}
};

constexpr std::array<const char *, 4> kLayouts{
    {"HWC", "NHWC", "CHW", "NCHW"}
};

template<class T>
nvcv::DataType Dtype()
{
    if constexpr (std::is_same_v<T, uint8_t>)
        return nvcv::TYPE_U8;
    if constexpr (std::is_same_v<T, uint16_t>)
        return nvcv::TYPE_U16;
    if constexpr (std::is_same_v<T, int16_t>)
        return nvcv::TYPE_S16;
    if constexpr (std::is_same_v<T, __half>)
        return nvcv::TYPE_F16;
    return nvcv::TYPE_F32;
}

nvcv::ImageFormat Format(nvcv::DataType dtype, int channels)
{
    const std::array<nvcv::Swizzle, 4> swizzle{nvcv::Swizzle::S_X000, nvcv::Swizzle::S_XY00, nvcv::Swizzle::S_XYZ0,
                                               nvcv::Swizzle::S_XYZW};
    const std::array<nvcv::Packing, 4> p8{nvcv::Packing::X8, nvcv::Packing::X8_Y8, nvcv::Packing::X8_Y8_Z8,
                                          nvcv::Packing::X8_Y8_Z8_W8};
    const std::array<nvcv::Packing, 4> p16{nvcv::Packing::X16, nvcv::Packing::X16_Y16, nvcv::Packing::X16_Y16_Z16,
                                           nvcv::Packing::X16_Y16_Z16_W16};
    const std::array<nvcv::Packing, 4> p32{nvcv::Packing::X32, nvcv::Packing::X32_Y32, nvcv::Packing::X32_Y32_Z32,
                                           nvcv::Packing::X32_Y32_Z32_W32};
    auto                               packing = p32[channels - 1];
    if (dtype.strideBytes() == 1)
        packing = p8[channels - 1];
    else if (dtype.strideBytes() == 2)
        packing = p16[channels - 1];
    return nvcv::ImageFormat(nvcv::MemLayout::PL, dtype.dataKind(), swizzle[channels - 1], packing);
}

struct Stream
{
    cudaStream_t value{};

    Stream()
    {
        EXPECT_EQ(cudaSuccess, cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking));
    }

    Stream(const Stream &)            = delete;
    Stream &operator=(const Stream &) = delete;

    ~Stream()
    {
        EXPECT_EQ(cudaSuccess, cudaStreamDestroy(value));
    }
};

int64_t AxisExtent(char axis, int64_t n, int64_t h, int64_t w, int64_t c)
{
    switch (axis)
    {
    case 'N':
        return n;
    case 'H':
        return h;
    case 'W':
        return w;
    default:
        return c;
    }
}

// Owning guards surround the complete strided allocation, including row/plane/sample padding.
struct GuardedTensor
{
    static constexpr size_t  kGuard    = 256;
    static constexpr uint8_t kSentinel = 0xa5;
    nvcv::Tensor             tensor;
    uint8_t                 *allocation = nullptr;
    std::vector<uint8_t>     host;
    std::vector<bool>        logical;
    int                      n;
    int                      h;
    int                      w;
    int                      c;
    int                      element;
    int64_t                  row;
    int64_t                  plane;
    int64_t                  sample;
    bool                     planar;
    size_t                   baseOffset;

    GuardedTensor(const char *layout, nvcv::DataType dtype, int samples, int height, int width, int channels,
                  bool padded = true, int byteOffset = 0, int rowPadding = 0)
        : n(samples)
        , h(height)
        , w(width)
        , c(channels)
        , element(dtype.strideBytes())
        , planar(std::string(layout).find("CHW") != std::string::npos)
        , baseOffset(kGuard + byteOffset)
    {
        row = int64_t(w) * element * (planar ? 1 : c);
        if (padded)
            row = ((row + 63) / 64) * 64 + 64;
        row += rowPadding;
        plane  = row * h + (padded ? 128 : 0);
        sample = plane * (planar ? c : 1) + (padded ? 256 : 0);
        host.assign(baseOffset + size_t(sample) * std::max(n, 1) + kGuard, kSentinel);
        logical.assign(host.size(), false);
        EXPECT_EQ(cudaSuccess, cudaMalloc(reinterpret_cast<void **>(&allocation), host.size()));
        nvcv::TensorLayout      l(layout);
        std::array<int64_t, 4>  shape{};
        NVCVTensorBufferStrided buffer{};
        buffer.basePtr = reinterpret_cast<NVCVByte *>(allocation + baseOffset);
        for (int i = 0; i < l.rank(); ++i)
        {
            char axis         = l[i];
            shape[i]          = AxisExtent(axis, n, h, w, c);
            buffer.strides[i] = AxisExtent(axis, sample, row, element * (planar ? 1 : c), planar ? plane : element);
        }
        tensor = nvcv::TensorWrapData(
            nvcv::TensorDataStridedCuda(nvcv::TensorShape(shape.data(), l.rank(), l), dtype, buffer));
        Upload();
    }

    ~GuardedTensor()
    {
        EXPECT_EQ(cudaSuccess, cudaFree(allocation));
    }

    GuardedTensor(const GuardedTensor &)            = delete;
    GuardedTensor &operator=(const GuardedTensor &) = delete;

    size_t Offset(int batch, int y, int x, int channel) const
    {
        return baseOffset + batch * sample + y * row
             + (planar ? channel * plane + x * element : (x * c + channel) * element);
    }

    template<class T>
    void Set(int batch, const std::vector<T> &values)
    {
        ASSERT_EQ(size_t(h) * w * c, values.size());
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x)
                for (int channel = 0; channel < c; ++channel)
                {
                    size_t offset = Offset(batch, y, x, channel);
                    std::memcpy(host.data() + offset, &values[(size_t(y) * w + x) * c + channel], sizeof(T));
                    std::fill_n(logical.begin() + offset, sizeof(T), true);
                }
    }

    template<class T>
    std::vector<T> Get(int batch) const
    {
        std::vector<T> result(size_t(h) * w * c);
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x)
                for (int channel = 0; channel < c; ++channel)
                    std::memcpy(&result[(size_t(y) * w + x) * c + channel], host.data() + Offset(batch, y, x, channel),
                                sizeof(T));
        return result;
    }

    void Upload() const
    {
        ASSERT_EQ(cudaSuccess, cudaMemcpy(allocation, host.data(), host.size(), cudaMemcpyHostToDevice));
        // Pageable H2D can return after staging; nonblocking test streams do not wait on the default stream.
        ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(nullptr));
    }

    void Download()
    {
        ASSERT_EQ(cudaSuccess, cudaMemcpy(host.data(), allocation, host.size(), cudaMemcpyDeviceToHost));
    }

    void CheckPadding() const
    {
        for (size_t i = 0; i < host.size(); ++i)
            if (!logical[i])
            {
                ASSERT_EQ(kSentinel, host[i]) << "guard/padding byte " << i;
            }
    }

    nvcv::Image AsImage(nvcv::ImageFormat format) const
    {
        nvcv::ImageDataStridedCuda::Buffer buffer{};
        buffer.numPlanes           = 1;
        buffer.planes[0].basePtr   = reinterpret_cast<NVCVByte *>(allocation + baseOffset);
        buffer.planes[0].width     = w;
        buffer.planes[0].height    = h;
        buffer.planes[0].rowStride = static_cast<int32_t>(row);
        return nvcv::ImageWrapData(nvcv::ImageDataStridedCuda(format, buffer));
    }
};

enum class Pattern
{
    Random,
    Ramp,
    Constant,
    Checkerboard,
    Step,
    Subnormal,
    Cancellation,
    Impulse
};

template<class T>
float RandomValue(std::mt19937 &generator) // NOSONAR: deterministic test data, not security-sensitive.
{
    auto bits = static_cast<uint32_t>(generator());
    if constexpr (std::is_integral_v<T>)
    {
        T raw;
        std::memcpy(&raw, &bits, sizeof(T));
        return tvgold::Float(raw);
    }
    else
        return float(bits >> 8) / float(0x1000000) * (std::is_same_v<T, __half> ? 255.f : 1.f);
}

template<class T>
float PatternValue(Pattern pattern, size_t i, int h, int w, int channels, unsigned seed, float randomValue)
{
    auto x = int(i / channels % w);
    auto y = int(i / channels / w);
    switch (pattern)
    {
    case Pattern::Ramp:
        return float((i * 17 + seed * 13) % 251);
    case Pattern::Constant:
        return 7.f;
    case Pattern::Checkerboard:
        return ((x + y) & 1) ? 0.f : 255.f;
    case Pattern::Step:
        return x < w / 2 ? float(std::numeric_limits<T>::lowest()) : float(std::numeric_limits<T>::max());
    case Pattern::Subnormal:
        return float(int(i % 17) - 8) * 0x1p-24f;
    case Pattern::Cancellation:
        return ((i & 1) ? -1.f : 1.f) * float(1 + i % 31) * 0x1p12f;
    case Pattern::Impulse:
        return (x == int(seed % 3) * (w - 1) / 2 && y == int(seed / 3 % 3) * (h - 1) / 2) ? 200.f : 0.f;
    case Pattern::Random:
        return randomValue;
    }
    return randomValue;
}

template<class T>
std::vector<T> Input(int h, int w, int channels, unsigned seed, Pattern pattern = Pattern::Random)
{
    std::mt19937   generator(seed);
    std::vector<T> values(size_t(h) * w * channels);
    size_t         i = 0;
    for (T &value : values)
        value = tvgold::Store<T>(PatternValue<T>(pattern, i++, h, w, channels, seed, RandomValue<T>(generator)));
    return values;
}

template<class T>
std::string DifferenceContext(const std::vector<T> &gold, const std::vector<T> &actual)
{
    if (gold.size() != actual.size())
        return "buffer size mismatch";
    size_t worst = gold.size();
    float  error = -1.f;
    for (size_t i = 0; i < gold.size(); ++i)
        if (std::memcmp(&gold[i], &actual[i], sizeof(T)) != 0)
        {
            float difference = std::abs(tvgold::Float(gold[i]) - tvgold::Float(actual[i]));
            if (worst == gold.size() || difference > error)
            {
                worst = i;
                error = difference;
            }
        }
    if (worst == gold.size())
        return "buffers are bit-exact";
    return (::testing::Message() << "largest differing lane=" << worst << " gold=" << tvgold::Float(gold[worst])
                                 << " actual=" << tvgold::Float(actual[worst]) << " abs_error=" << error)
        .GetString();
}

template<class T>
void RunTensor(const char *layout, int channels, Family family, int sh = 13, int sw = 17, int dh = 7, int dw = 11,
               Pattern pattern = Pattern::Random, int batches = 0, bool padded = true)
{
    int n = batches;
    if (!n)
        n = layout[0] == 'N' ? 2 : 1;
    SCOPED_TRACE(::testing::Message() << layout << " C=" << channels << " N=" << n << " mode=" << family.mode
                                      << " AA=" << family.aa << " " << sw << "x" << sh << "->" << dw << "x" << dh
                                      << " pattern=" << int(pattern) << " dtype=" << Dtype<T>());
    Stream                      stream;
    GuardedTensor               input(layout, Dtype<T>(), n, sh, sw, channels, padded);
    GuardedTensor               output(layout, Dtype<T>(), n, dh, dw, channels, padded);
    std::vector<std::vector<T>> inputs(n);
    for (int b = 0; b < n; ++b)
    {
        inputs[b] = Input<T>(sh, sw, channels, 1234 + b, pattern);
        input.Set(b, inputs[b]);
        output.Set(b, std::vector<T>(size_t(dh) * dw * channels));
    }
    input.Upload();
    cvcuda::TvResize op;
    auto             workspace
        = cvcuda::AllocateWorkspace(op.getWorkspaceRequirements(n, sh, sw, dh, dw, channels, family.mode, family.aa));
    op(stream.value, workspace.get(), input.tensor, output.tensor, family.mode, family.aa);
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream.value));
    output.Download();
    for (int b = 0; b < n; ++b)
    {
        auto gold   = tvgold::Resize(inputs[b], sh, sw, dh, dw, channels, family.mode, family.aa);
        auto actual = output.Get<T>(b);
        tvfixture::Export(layout, channels, b, n, padded, sh, sw, dh, dw, family.mode, family.aa, int(pattern),
                          inputs[b], gold, actual);
        EXPECT_TRUE(tvgold::WithinTolerance(gold, actual, family.mode, family.aa, sh == dh && sw == dw))
            << "sample=" << b << " " << DifferenceContext(gold, actual);
    }
    output.CheckPadding();
    input.Download();
    input.CheckPadding();
    for (int b = 0; b < n; ++b)
        EXPECT_TRUE(tvgold::WithinTolerance(inputs[b], input.Get<T>(b), family.mode, family.aa, true));
}

template<class T>
void RunVarShape(int channels, Family family, const std::vector<std::array<int, 4>> &sizes,
                 Pattern pattern = Pattern::Random, int misaligned = -1)
{
    auto n = static_cast<int>(sizes.size());
    SCOPED_TRACE(::testing::Message() << "VarShape C=" << channels << " mode=" << family.mode << " AA=" << family.aa);
    Stream                                      stream;
    nvcv::ImageBatchVarShape                    input(n);
    nvcv::ImageBatchVarShape                    output(n);
    std::vector<std::unique_ptr<GuardedTensor>> sources;
    std::vector<std::unique_ptr<GuardedTensor>> destinations;
    std::vector<std::vector<T>>                 inputs(n);
    std::vector<int32_t>                        sh(n);
    std::vector<int32_t>                        sw(n);
    std::vector<int32_t>                        dh(n);
    std::vector<int32_t>                        dw(n);
    for (int b = 0; b < n; ++b)
    {
        sh[b] = sizes[b][0];
        sw[b] = sizes[b][1];
        dh[b] = sizes[b][2];
        dw[b] = sizes[b][3];
        sources.emplace_back(
            std::make_unique<GuardedTensor>("HWC", Dtype<T>(), 1, sh[b], sw[b], channels, true, b == misaligned, 0));
        destinations.emplace_back(std::make_unique<GuardedTensor>("HWC", Dtype<T>(), 1, dh[b], dw[b], channels));
        inputs[b] = Input<T>(sh[b], sw[b], channels, 1234 + b, pattern);
        sources[b]->Set(0, inputs[b]);
        sources[b]->Upload();
        destinations[b]->Set(0, std::vector<T>(size_t(dh[b]) * dw[b] * channels));
        input.pushBack(sources[b]->AsImage(Format(Dtype<T>(), channels)));
        output.pushBack(destinations[b]->AsImage(Format(Dtype<T>(), channels)));
    }
    cvcuda::TvResize op;
    auto             workspace = cvcuda::AllocateWorkspace(
                    op.getWorkspaceRequirements(n, sh.data(), sw.data(), dh.data(), dw.data(), channels, family.mode, family.aa));
    op(stream.value, workspace.get(), input, output, family.mode, family.aa);
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream.value));
    for (int b = 0; b < n; ++b)
    {
        SCOPED_TRACE(::testing::Message() << "sample=" << b << " " << sw[b] << "x" << sh[b] << "->" << dw[b] << "x"
                                          << dh[b] << " pattern=" << int(pattern) << " dtype=" << Dtype<T>());
        destinations[b]->Download();
        auto gold   = tvgold::Resize(inputs[b], sh[b], sw[b], dh[b], dw[b], channels, family.mode, family.aa);
        auto actual = destinations[b]->Get<T>(0);
        tvfixture::Export("VarShape", channels, b, n, true, sh[b], sw[b], dh[b], dw[b], family.mode, family.aa,
                          int(pattern), inputs[b], gold, actual);
        EXPECT_TRUE(tvgold::WithinTolerance(gold, actual, family.mode, family.aa, sh[b] == dh[b] && sw[b] == dw[b]))
            << DifferenceContext(gold, actual);
        destinations[b]->CheckPadding();
        sources[b]->Download();
        sources[b]->CheckPadding();
        EXPECT_TRUE(tvgold::WithinTolerance(inputs[b], sources[b]->Get<T>(0), family.mode, family.aa, true));
    }
}

template<class T>
void RunLayoutChannels(const char *layout, Family family, bool padded)
{
    for (int channels : {1, 2, 3, 4, 5, 17})
        if (channels <= 4 || std::string(layout).find("CHW") != std::string::npos)
            RunTensor<T>(layout, channels, family, 13, 17, 7, 11, Pattern::Random, 0, padded);
}

NVCV_TYPED_TEST_SUITE(OpTvResize, ::testing::Types<uint8_t, uint16_t, int16_t, __half, float>);

TEST(OpTvResize, linear_interleaved_u8_general_randomized)
{
    for (int channels : {1, 3, 4})
    {
        for (auto dims : {
                 std::array<int, 4>{1080, 1920,  720, 1280},
                 {2160, 3840, 1080, 1920},
                 {1080, 1920,  224,  224},
                 {1081, 1921,  719, 1281},
                 { 720, 1280, 1080, 1920},
                 {1080, 1920,  720, 1920},
                 {1080, 1920, 1080, 1280}
        })
            RunTensor<uint8_t>("NHWC", channels, {NVCV_INTERP_LINEAR, false}, dims[0], dims[1], dims[2], dims[3],
                               Pattern::Random, 1, true);
    }
    for (int channels : {3, 4})
        RunVarShape<uint8_t>(channels,
                             {
                                 NVCV_INTERP_LINEAR, false
        },
                             {{1081, 1921, 719, 1281}, {720, 1280, 1080, 1920}, {1080, 1920, 720, 1920}});
}

TYPED_TEST(OpTvResize, independent_gold_all_families_layouts_channels)
{
    for (auto family : kFamilies)
        for (auto layout : kLayouts) RunLayoutChannels<TypeParam>(layout, family, true);
}

TYPED_TEST(OpTvResize, varshape_correct_output_all_families_channels)
{
    for (auto family : kFamilies)
        for (int channels : {1, 3, 4})
            RunVarShape<TypeParam>(channels, family,
                                   {
                                       {13, 17,  7, 11},
                                       { 9, 21, 14, 10},
                                       { 1,  7,  4,  3}
            });
}

TYPED_TEST(OpTvResize, exact_identity_and_geometry)
{
    const std::vector<std::array<int, 4>> sizes{
        {48, 64, 48, 64},
        { 1,  1,  3,  5},
        { 1,  7,  4, 11},
        { 7,  1, 11,  4},
        { 1,  7,  4,  3},
        { 7,  1,  3,  4},
        { 9,  7,  4,  3},
        { 7,  9,  7, 13},
        { 7,  9, 11,  9},
        { 7,  9,  7,  5},
        { 7,  9,  3,  9},
        { 8, 12,  4,  6},
        { 4,  6,  8, 12},
        {13, 17,  7, 23},
        {17, 13, 23,  7}
    };
    for (auto family : kFamilies)
    {
        for (auto size : sizes)
            for (auto layout : kLayouts) RunTensor<TypeParam>(layout, 1, family, size[0], size[1], size[2], size[3]);
        RunVarShape<TypeParam>(1, family, sizes);
    }
}

template<class PixelType>
void CheckVarShapeBatchAgainstSingles(int channels, Family family, const std::vector<std::array<int, 4>> &sizes,
                                      bool exportBeforeSubmit = false, bool capture = false)
{
    int                                         count = static_cast<int>(sizes.size());
    Stream                                      stream;
    Stream                                      producer;
    cvcuda::TvResize                            operation;
    nvcv::ImageBatchVarShape                    input(count);
    nvcv::ImageBatchVarShape                    output(count);
    std::vector<std::unique_ptr<GuardedTensor>> sources;
    std::vector<std::unique_ptr<GuardedTensor>> destinations;
    std::vector<std::unique_ptr<GuardedTensor>> singles;
    std::vector<std::vector<PixelType>>         values(count);
    std::vector<int32_t>                        sourceHeights(count);
    std::vector<int32_t>                        sourceWidths(count);
    std::vector<int32_t>                        destinationHeights(count);
    std::vector<int32_t>                        destinationWidths(count);
    for (int sample = 0; sample < count; ++sample)
    {
        auto size                  = sizes[sample];
        sourceHeights[sample]      = size[0];
        sourceWidths[sample]       = size[1];
        destinationHeights[sample] = size[2];
        destinationWidths[sample]  = size[3];
        sources.emplace_back(std::make_unique<GuardedTensor>("HWC", Dtype<PixelType>(), 1, size[0], size[1], channels));
        destinations.emplace_back(
            std::make_unique<GuardedTensor>("HWC", Dtype<PixelType>(), 1, size[2], size[3], channels));
        singles.emplace_back(std::make_unique<GuardedTensor>("HWC", Dtype<PixelType>(), 1, size[2], size[3], channels));
        values[sample] = Input<PixelType>(size[0], size[1], channels, 789 + sample);
        if (family.mode == NVCV_INTERP_NEAREST || family.mode == NVCV_INTERP_NEAREST_FLOOR)
        {
            constexpr std::array<uint32_t, 8> patterns{0x00000000, 0x80008000, 0x7fc17e35, 0xffc1fe71,
                                                       0x00010001, 0x3f803c00, 0xffff7fff, 0x1234abcd};
            for (size_t scalar = 0; scalar < values[sample].size(); ++scalar)
            {
                uint32_t bits = patterns[(scalar + sample) % patterns.size()];
                std::memcpy(static_cast<void *>(&values[sample][scalar]), &bits, sizeof(PixelType));
            }
        }
        sources[sample]->Set(0, values[sample]);
        sources[sample]->Upload();
        destinations[sample]->Set(0, std::vector<PixelType>(size_t(size[2]) * size[3] * channels));
        singles[sample]->Set(0, std::vector<PixelType>(size_t(size[2]) * size[3] * channels));
        input.pushBack(sources[sample]->AsImage(Format(Dtype<PixelType>(), channels)));
        output.pushBack(destinations[sample]->AsImage(Format(Dtype<PixelType>(), channels)));
    }
    auto workspace = cvcuda::AllocateWorkspace(
        operation.getWorkspaceRequirements(count, sourceHeights.data(), sourceWidths.data(), destinationHeights.data(),
                                           destinationWidths.data(), channels, family.mode, family.aa));
    for (int sample = 0; sample < count; ++sample)
    {
        nvcv::ImageBatchVarShape singleInput(1);
        nvcv::ImageBatchVarShape singleOutput(1);
        singleInput.pushBack(input[sample]);
        singleOutput.pushBack(singles[sample]->AsImage(Format(Dtype<PixelType>(), channels)));
        operation(stream.value, workspace.get(), singleInput, singleOutput, family.mode, family.aa);
    }
    if (exportBeforeSubmit)
    {
        ASSERT_TRUE(input.exportData<nvcv::ImageBatchVarShapeDataStridedCuda>(producer.value));
        ASSERT_TRUE(output.exportData<nvcv::ImageBatchVarShapeDataStridedCuda>(producer.value));
    }
    if (capture)
    {
        ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream.value));
        ASSERT_EQ(cudaSuccess, cudaStreamBeginCapture(stream.value, cudaStreamCaptureModeThreadLocal));
    }
    operation(stream.value, workspace.get(), input, output, family.mode, family.aa);
    if (capture)
    {
        cudaGraph_t     graph;
        cudaGraphExec_t executable;
        ASSERT_EQ(cudaSuccess, cudaStreamEndCapture(stream.value, &graph));
        ASSERT_EQ(cudaSuccess, cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
        ASSERT_EQ(cudaSuccess, cudaGraphLaunch(executable, stream.value));
        ASSERT_EQ(cudaSuccess, cudaGraphLaunch(executable, stream.value));
        ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream.value));
        ASSERT_EQ(cudaSuccess, cudaGraphExecDestroy(executable));
        ASSERT_EQ(cudaSuccess, cudaGraphDestroy(graph));
    }
    if (!capture)
    {
        operation(stream.value, workspace.get(), input, output, family.mode, family.aa);
    }
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream.value));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(producer.value));
    for (int sample = 0; sample < count; ++sample)
    {
        SCOPED_TRACE(::testing::Message()
                     << "sample=" << sample << " channels=" << channels << " mode=" << family.mode);
        destinations[sample]->Download();
        singles[sample]->Download();
        auto actual = destinations[sample]->Get<PixelType>(0);
        auto single = singles[sample]->Get<PixelType>(0);
        EXPECT_EQ(0, std::memcmp(actual.data(), single.data(), actual.size() * sizeof(PixelType)));
        auto gold = tvgold::Resize(values[sample], sizes[sample][0], sizes[sample][1], sizes[sample][2],
                                   sizes[sample][3], channels, family.mode, family.aa);
        EXPECT_TRUE(tvgold::WithinTolerance(gold, actual, family.mode, family.aa));
        destinations[sample]->CheckPadding();
        singles[sample]->CheckPadding();
        sources[sample]->Download();
        auto unchanged = sources[sample]->Get<PixelType>(0);
        EXPECT_EQ(0, std::memcmp(values[sample].data(), unchanged.data(), unchanged.size() * sizeof(PixelType)));
        sources[sample]->CheckPadding();
    }
}

TYPED_TEST(OpTvResize, varshape_nearest_batch_exact_bits_geometry_and_clean_exports)
{
    const std::vector<std::array<int, 4>> sizes{
        {13, 257,  7, 129},
        {31,   5, 17,  19},
        { 1,   1,  3,   5},
        { 1,   7,  4,   7},
        { 7,   1,  7,  11},
        { 9,  17,  9,  17},
        { 3,   5, 11,   9}
    };
    for (int channels : {1, 3, 4})
        for (auto mode : {NVCV_INTERP_NEAREST, NVCV_INTERP_NEAREST_FLOOR})
            for (bool antialias : {false, true})
                ASSERT_NO_FATAL_FAILURE(
                    CheckVarShapeBatchAgainstSingles<TypeParam>(channels, {mode, antialias}, sizes, true));
}

TEST(OpTvResize, varshape_cubic_batch_exact_heterogeneous_and_mixed_fallback)
{
    std::vector<std::array<int, 4>> sizes{
        {13, 257,  7, 129},
        {31,   5, 17,  19},
        { 3,   7,  9,  13},
        {17,   9,  7,   5},
        { 1,   1,  3,   5}
    };
    for (int channels : {1, 3, 4})
        ASSERT_NO_FATAL_FAILURE(CheckVarShapeBatchAgainstSingles<uint8_t>(channels, {NVCV_INTERP_CUBIC, false}, sizes));
    sizes.push_back({7, 9, 7, 9});
    sizes.push_back({7, 9, 7, 13});
    for (int channels : {1, 3, 4})
    {
        ASSERT_NO_FATAL_FAILURE(CheckVarShapeBatchAgainstSingles<uint8_t>(channels, {NVCV_INTERP_CUBIC, false}, sizes));
        std::reverse(sizes.begin(), sizes.end());
        ASSERT_NO_FATAL_FAILURE(CheckVarShapeBatchAgainstSingles<uint8_t>(channels, {NVCV_INTERP_CUBIC, false}, sizes));
    }
}

TEST(OpTvResize, varshape_batched_capture_replay_and_count_tails)
{
    for (int count : {1, 2, 31, 32, 33, 69})
    {
        std::vector<std::array<int, 4>> sizes(count, {3, 5, 7, 9});
        sizes.back() = {7, 257, 5, 129};
        ASSERT_NO_FATAL_FAILURE(
            CheckVarShapeBatchAgainstSingles<uint16_t>(4, {NVCV_INTERP_NEAREST_FLOOR, true}, sizes));
    }
    const std::vector<std::array<int, 4>> sizes{
        {7, 11, 3,  5},
        {5,  9, 9, 13}
    };
    for (bool clean : {false, true})
    {
        ASSERT_NO_FATAL_FAILURE(
            CheckVarShapeBatchAgainstSingles<float>(3, {NVCV_INTERP_NEAREST, false}, sizes, clean, true));
        ASSERT_NO_FATAL_FAILURE(
            CheckVarShapeBatchAgainstSingles<uint8_t>(4, {NVCV_INTERP_CUBIC, false}, sizes, clean, true));
    }
}

TEST(OpTvResize_Negative, varshape_batched_late_invalid_storage_preserves_all_outputs)
{
    Stream           stream;
    cvcuda::TvResize operation;
    GuardedTensor    source("HWC", nvcv::TYPE_U16, 1, 9, 13, 3);
    GuardedTensor    first("HWC", nvcv::TYPE_U16, 1, 5, 7, 3);
    GuardedTensor    last("HWC", nvcv::TYPE_U16, 1, 5, 7, 3);
    const auto       untouched = first.host;
    auto             format    = Format(nvcv::TYPE_U16, 3);
    for (bool badInput : {false, true})
    {
        const auto                        &storage = badInput ? source : last;
        nvcv::ImageDataStridedCuda::Buffer buffer{};
        buffer.numPlanes                 = 1;
        buffer.planes[0].basePtr         = reinterpret_cast<NVCVByte *>(storage.allocation + storage.baseOffset);
        buffer.planes[0].width           = storage.w;
        buffer.planes[0].height          = storage.h;
        buffer.planes[0].rowStride       = static_cast<int32_t>(storage.row + 1);
        auto                     invalid = nvcv::ImageWrapData(nvcv::ImageDataStridedCuda(format, buffer));
        nvcv::ImageBatchVarShape input(2);
        nvcv::ImageBatchVarShape output(2);
        input.pushBack(source.AsImage(format));
        output.pushBack(first.AsImage(format));
        input.pushBack(badInput ? invalid : source.AsImage(format));
        output.pushBack(badInput ? last.AsImage(format) : invalid);
        NVCVWorkspace workspace{};
        for (auto mode : {NVCV_INTERP_NEAREST, NVCV_INTERP_NEAREST_FLOOR})
        {
            EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
                      cvcudaTvResizeImageBatchSubmit(operation.handle(), stream.value, &workspace, input.handle(),
                                                     output.handle(), mode, false));
            ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream.value));
            first.Download();
            last.Download();
            EXPECT_EQ(untouched, first.host);
            EXPECT_EQ(untouched, last.host);
        }
    }
}

template<class TypeParam>
void CheckReformatParity(Family family, bool batched, int channels, int n = 0, int sh = 97, int sw = 113, int dh = 71,
                         int dw = 53)
{
    Stream stream;
    if (!n)
        n = batched ? 2 : 1;
    auto il = batched ? "NHWC" : "HWC";
    auto pl = batched ? "NCHW" : "CHW";
    SCOPED_TRACE(::testing::Message() << il << "->" << pl << " C=" << channels << " N=" << n << " mode=" << family.mode
                                      << " AA=" << family.aa << " " << sw << "x" << sh << "->" << dw << "x" << dh
                                      << " dtype=" << Dtype<TypeParam>());
    GuardedTensor source(il, Dtype<TypeParam>(), n, sh, sw, channels);
    GuardedTensor planarSource(pl, Dtype<TypeParam>(), n, sh, sw, channels);
    GuardedTensor direct(pl, Dtype<TypeParam>(), n, dh, dw, channels);
    GuardedTensor interleavedResult(il, Dtype<TypeParam>(), n, dh, dw, channels);
    GuardedTensor reformatted(pl, Dtype<TypeParam>(), n, dh, dw, channels);
    for (int b = 0; b < n; ++b) source.Set(b, Input<TypeParam>(sh, sw, channels, b + 61));
    source.Upload();
    cvcuda::Reformat reformat;
    reformat(stream.value, source.tensor, planarSource.tensor);
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream.value));
    planarSource.Download();
    for (int b = 0; b < n; ++b)
    {
        auto expected = source.Get<TypeParam>(b);
        auto actual   = planarSource.Get<TypeParam>(b);
        ASSERT_TRUE(tvgold::WithinTolerance(expected, actual, family.mode, family.aa, true))
            << "input reformat sample=" << b << " " << DifferenceContext(expected, actual);
    }
    cvcuda::TvResize op;
    auto             ws1
        = cvcuda::AllocateWorkspace(op.getWorkspaceRequirements(n, sh, sw, dh, dw, channels, family.mode, family.aa));
    auto ws2
        = cvcuda::AllocateWorkspace(op.getWorkspaceRequirements(n, sh, sw, dh, dw, channels, family.mode, family.aa));
    op(stream.value, ws1.get(), planarSource.tensor, direct.tensor, family.mode, family.aa);
    op(stream.value, ws2.get(), source.tensor, interleavedResult.tensor, family.mode, family.aa);
    reformat(stream.value, interleavedResult.tensor, reformatted.tensor);
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream.value));
    direct.Download();
    reformatted.Download();
    for (int b = 0; b < n; ++b)
    {
        auto expected = direct.Get<TypeParam>(b);
        auto actual   = reformatted.Get<TypeParam>(b);
        EXPECT_TRUE(tvgold::WithinTolerance(expected, actual, family.mode, family.aa, true))
            << "output parity sample=" << b << " " << DifferenceContext(expected, actual);
    }
}

TYPED_TEST(OpTvResize, planar_matches_interleaved_reformat_exactly)
{
    for (auto family : kFamilies)
        for (bool batched : {false, true})
            for (int channels : {1, 2, 3, 4})
                ASSERT_NO_FATAL_FAILURE(CheckReformatParity<TypeParam>(family, batched, channels));
}

// NHWC takes the shared-tap and vectored kernels while NCHW keeps the per-plane kernels, so byte parity on both
// sides of each dispatch threshold pins the fast paths to the previous arithmetic.
TEST(OpTvResize, interleaved_fast_paths_match_planar_bitexactly)
{
    for (auto mode : {NVCV_INTERP_LINEAR, NVCV_INTERP_CUBIC})
    {
        for (int channels : {3, 4})
            for (auto [aa, n, sh, sw, dh, dw] : {
                     std::array{0, 1, 1080, 1920,  720, 1280},
                     std::array{0, 1,  720, 1280, 1080, 1920},
                     std::array{1, 4, 1080, 1920,  720, 1280}
            })
                ASSERT_NO_FATAL_FAILURE(
                    CheckReformatParity<uint8_t>({mode, aa != 0}, true, channels, n, sh, sw, dh, dw));
        // Planar plane-count thresholds: vectored at >= 4 planes, 8192-block grids at >= 16 planes.
        for (int n : {1, 2, 5, 6})
            ASSERT_NO_FATAL_FAILURE(CheckReformatParity<uint8_t>({mode, false}, true, 3, n, 1080, 1920, 720, 1280));
    }
    // Shared-tap linear kernel starts at 256x256 outputs.
    for (int dh : {256, 255})
        ASSERT_NO_FATAL_FAILURE(
            CheckReformatParity<uint8_t>({NVCV_INTERP_LINEAR, false}, true, 3, 1, 300, 300, dh, 256));
    // AA horizontal ratios around 3.5.
    for (int dw : {200, 199})
        ASSERT_NO_FATAL_FAILURE(
            CheckReformatParity<uint8_t>({NVCV_INTERP_LINEAR, true}, true, 3, 4, 1080, 700, 720, dw));
    // Strong vertical AA reductions.
    for (int sh : {768, 960})
        ASSERT_NO_FATAL_FAILURE(
            CheckReformatParity<uint8_t>({NVCV_INTERP_LINEAR, true}, true, 3, 4, sh, 1024, 64, 1000));
    ASSERT_NO_FATAL_FAILURE(CheckReformatParity<uint8_t>({NVCV_INTERP_CUBIC, true}, true, 3, 4, 448, 1024, 64, 1000));
    ASSERT_NO_FATAL_FAILURE(CheckReformatParity<float>({NVCV_INTERP_LINEAR, true}, true, 3, 4, 1080, 1920, 720, 1280));
}

TEST(OpTvResizeGold, rational_fixture_cross_check)
{
    // Independently derived exact fractions for resizing [[0,8,0,8]] to (1,3).
    const std::array<std::array<float, 3>, 4> rational{
        {{4.f / 3, 4, 20.f / 3}, {139.f / 108, 4, 725.f / 108}, {12.f / 5, 4, 28.f / 5}, {56.f / 23, 4, 128.f / 23}}
    };
    for (int i = 0; i < 4; ++i)
    {
        auto family = kFamilies[i + 2];
        auto gold   = tvgold::Resize(std::vector<float>{0, 8, 0, 8}, 1, 4, 1, 3, 1, family.mode, family.aa);
        EXPECT_TRUE(tvgold::WithinTolerance(std::vector<float>(rational[i].begin(), rational[i].end()), gold,
                                            family.mode, family.aa));
    }
    EXPECT_EQ(51, tvgold::NearestIndex(104, 97, 48, NVCV_INTERP_NEAREST));
    EXPECT_EQ(117, tvgold::NearestIndex(143, 33, 27, NVCV_INTERP_NEAREST_FLOOR));
    EXPECT_EQ(0, tvgold::Store<uint8_t>(0.5f));
    EXPECT_EQ(2, tvgold::Store<uint8_t>(1.5f));
    EXPECT_EQ(-2, tvgold::Store<int16_t>(-1.5f));
    EXPECT_EQ(65535, tvgold::Store<uint16_t>(70000.f));
    EXPECT_EQ(-32768, tvgold::Store<int16_t>(-40000.f));
}

TEST(OpTvResizeGold, frozen_comparator_boundaries)
{
    auto half = [](float x)
    {
        return std::vector<__half>{__float2half_rn(x)};
    };
    for (bool aa : {false, true})
    {
        EXPECT_TRUE(tvgold::WithinTolerance(half(0), half(0x1p-22f), NVCV_INTERP_LINEAR, aa));
        EXPECT_FALSE(tvgold::WithinTolerance(half(0), half(5 * 0x1p-24f), NVCV_INTERP_LINEAR, aa));
        EXPECT_TRUE(tvgold::WithinTolerance(half(1), half(1 + 4 * 0x1p-10f), NVCV_INTERP_CUBIC, aa));
        EXPECT_FALSE(tvgold::WithinTolerance(half(1), half(1 + 5 * 0x1p-10f), NVCV_INTERP_CUBIC, aa));
        float bound = (aa ? 5e-6f : 1e-5f) * 8.f;
        EXPECT_TRUE(
            tvgold::WithinTolerance(std::vector<float>{0, 8}, std::vector<float>{bound, 8}, NVCV_INTERP_LINEAR, aa));
        EXPECT_FALSE(tvgold::WithinTolerance(
            std::vector<float>{0, 8}, std::vector<float>{std::nextafter(bound, INFINITY), 8}, NVCV_INTERP_LINEAR, aa));
    }
    for (auto mode : {NVCV_INTERP_NEAREST, NVCV_INTERP_NEAREST_FLOOR})
    {
        EXPECT_FALSE(tvgold::WithinTolerance(half(1), half(1 + 0x1p-10f), mode, false));
        EXPECT_FALSE(tvgold::WithinTolerance(std::vector<float>{0.f}, std::vector<float>{-0.f}, mode, false));
    }
    EXPECT_TRUE(
        tvgold::WithinTolerance(std::vector<uint16_t>{255}, std::vector<uint16_t>{256}, NVCV_INTERP_LINEAR, true));
    EXPECT_FALSE(
        tvgold::WithinTolerance(std::vector<int16_t>{-32768}, std::vector<int16_t>{32767}, NVCV_INTERP_LINEAR, true));
}

template<class T>
void CheckFullRangeStepOvershoot()
{
    constexpr float low  = std::numeric_limits<T>::lowest();
    constexpr float high = std::numeric_limits<T>::max();
    SCOPED_TRACE(::testing::Message() << "CPU CUBIC AA 97x1->71x1 step@48 range=[" << low << "," << high << "]");
    std::vector<float> step(97);
    for (int x = 0; x < 97; ++x)
    {
        step[x] = x < 48 ? low : high;
        ASSERT_EQ(x < 48 ? low : high, step[x]) << "input store x=" << x;
        ASSERT_EQ(step[x], tvgold::Float(tvgold::Store<T>(step[x]))) << "integer input store x=" << x;
    }
    auto raw = tvgold::Resize(step, 1, 97, 1, 71, 1, NVCV_INTERP_CUBIC, true);
    EXPECT_LT(*std::min_element(raw.begin(), raw.end()), low);
    EXPECT_GT(*std::max_element(raw.begin(), raw.end()), high);
    ASSERT_LT(raw[34], low) << "lower lobe x=34";
    ASSERT_GT(raw[36], high) << "upper lobe x=36";
    EXPECT_EQ(std::numeric_limits<T>::lowest(), tvgold::Store<T>(raw[34])) << "lower saturated store";
    EXPECT_EQ(std::numeric_limits<T>::max(), tvgold::Store<T>(raw[36])) << "upper saturated store";
}

TEST(OpTvResizeGold, full_range_16bit_aa_cubic_step_overshoots_and_saturates)
{
    CheckFullRangeStepOvershoot<uint16_t>();
    CheckFullRangeStepOvershoot<int16_t>();
}

TEST(OpTvResize, full_range_16bit_aa_cubic_varshape_overshoot)
{
    const Family family{NVCV_INTERP_CUBIC, true};
    for (int channels : {1, 3, 4})
    {
        RunVarShape<uint16_t>(channels, family,
                              {
                                  {113, 97, 53, 71}
        },
                              Pattern::Step);
        RunVarShape<int16_t>(channels, family,
                             {
                                 {113, 97, 53, 71}
        },
                             Pattern::Step);
    }
    CheckFullRangeStepOvershoot<uint16_t>();
    CheckFullRangeStepOvershoot<int16_t>();
}

TYPED_TEST(OpTvResize, deterministic_patterns_and_nearest_rounding)
{
    for (auto family : kFamilies)
    {
        for (auto pattern : {Pattern::Ramp, Pattern::Constant, Pattern::Checkerboard, Pattern::Impulse})
            RunTensor<TypeParam>("NHWC", 3, family, 13, 17, 7, 11, pattern);
        RunTensor<TypeParam>("NHWC", 3, family, 97, 104, 71, 97);
        RunTensor<TypeParam>("NHWC", 3, family, 97, 143, 71, 33);
    }
}

TEST(OpTvResize, finite_float_stress_keeps_frozen_thresholds)
{
    for (auto family : kFamilies)
    {
        RunTensor<__half>("NCHW", 3, family, 13, 17, 7, 11, Pattern::Subnormal);
        RunTensor<float>("NHWC", 3, family, 13, 17, 7, 11, Pattern::Cancellation);
    }
}

TEST(OpTvResize, nchw_non_aa_ramp_unpadded_and_odd_channels)
{
    for (auto geometry : {
             std::array<int, 4>{1080, 1920,  720, 1280},
             {2160, 3840, 1080, 1920}
    })
        for (int channels : {1, 3})
            RunTensor<uint8_t>("NCHW", channels, {NVCV_INTERP_CUBIC, false}, geometry[0], geometry[1], geometry[2],
                               geometry[3], Pattern::Ramp, 1, false);
    RunTensor<uint8_t>("NCHW", 5, {NVCV_INTERP_CUBIC, false}, 37, 53, 19, 31, Pattern::Random, 1, true);
    RunTensor<uint8_t>("NCHW", 3, {NVCV_INTERP_LINEAR, false}, 1080, 1920, 720, 1280, Pattern::Ramp, 1, false);
}

TEST(OpTvResize, planar_non_aa_realistic)
{
    for (auto [sh, sw, dh, dw, n] : {
             std::array{1080, 1920,  720, 1280, 2},
             std::array{2160, 3840, 1080, 1920, 1},
             std::array{ 720, 1280, 1080, 1920, 1},
             std::array{1081, 1921,  719, 1283, 2},
             std::array{1080, 1920, 1080, 1280, 1},
             std::array{1080, 1920,  720, 1920, 1}
    })
        for (auto interp : {NVCV_INTERP_LINEAR, NVCV_INTERP_CUBIC})
        {
            RunTensor<uint8_t>("NCHW", 3, {interp, false}, sh, sw, dh, dw, Pattern::Random, n, true);
            RunTensor<uint16_t>("NCHW", 1, {interp, false}, sh, sw, dh, dw, Pattern::Random, 1, true);
            RunTensor<float>("NCHW", 3, {interp, false}, sh, sw, dh, dw, Pattern::Random, 1, false);
        }
}

TEST(OpTvResize, linear_aa_interleaved_realistic)
{
    for (auto [sh, sw, dh, dw, n] : {
             std::array{1080, 1920,  720, 1280, 4},
             std::array{2160, 3840, 1080, 1920, 2},
             std::array{1081, 1921,  719, 1281, 3},
             std::array{1080, 1920, 1080, 1280, 4},
             std::array{1080, 1920,  224,  224, 4}
    })
    {
        for (int channels : {1, 3, 4})
            RunTensor<uint8_t>("NHWC", channels, {NVCV_INTERP_LINEAR, true}, sh, sw, dh, dw, Pattern::Random, n, true);
        RunTensor<float>("NHWC", 3, {NVCV_INTERP_LINEAR, true}, sh, sw, dh, dw, Pattern::Random, n, true);
    }
}

TEST(OpTvResize, nearest_realistic_sizes_layouts_channels)
{
    for (auto mode : {NVCV_INTERP_NEAREST, NVCV_INTERP_NEAREST_FLOOR})
    {
        for (auto [sh, sw, dh, dw, n] : {
                 std::array{1080, 1920,  720, 1280, 5},
                 std::array{1081, 1921,  719, 1283, 2},
                 std::array{ 480,  641, 1080, 1920, 1}
        })
        {
            for (int channels : {1, 3, 4})
            {
                RunTensor<uint8_t>("NHWC", channels, {mode, false}, sh, sw, dh, dw, Pattern::Random, n, true);
                RunTensor<uint8_t>("NCHW", channels, {mode, false}, sh, sw, dh, dw, Pattern::Random, n, true);
            }
            RunTensor<uint16_t>("NHWC", 1, {mode, false}, sh, sw, dh, dw, Pattern::Random, n, true);
            RunTensor<uint16_t>("NCHW", 3, {mode, false}, sh, sw, dh, dw, Pattern::Random, n, true);
            RunTensor<float>("NHWC", 3, {mode, false}, sh, sw, dh, dw, Pattern::Random, n, true);
            RunTensor<float>("NCHW", 3, {mode, false}, sh, sw, dh, dw, Pattern::Random, n, true);
        }
        const std::vector<std::array<int, 4>> sizes{
            {1080, 1920,  720, 1280},
            {1081, 1921,  719, 1283},
            { 480,  641, 1080, 1920}
        };
        RunVarShape<uint8_t>(3, {mode, false}, sizes);
        RunVarShape<uint16_t>(1, {mode, false}, sizes);
        RunVarShape<float>(4, {mode, false}, sizes);
        // More than 65535 * 8 output rows runs the row grid-stride loop.
        RunTensor<uint8_t>("NHWC", 1, {mode, false}, 7, 1, 524289, 2, Pattern::Random, 1, true);
        RunVarShape<uint8_t>(1,
                             {
                                 mode, false
        },
                             {{7, 1, 524289, 2}, {5, 3, 7, 5}});
    }
}

template<class T>
void RunInterleavedRealistic(bool cubic = true)
{
    for (auto family : {
             Family{NVCV_INTERP_LINEAR, false},
             Family{ NVCV_INTERP_CUBIC, false}
    })
        for (int channels : {3, 4})
        {
            if (!cubic && family.mode == NVCV_INTERP_CUBIC)
                continue;
            for (auto [sh, sw, dh, dw, n] : {
                     std::array{1080, 1920,  720, 1280, 2},
                     std::array{ 720, 1280, 1080, 1920, 1},
                     std::array{1081, 1921,  719, 1281, 1},
                     std::array{1080, 1920, 1080, 1280, 1}
            })
                RunTensor<T>("NHWC", channels, family, sh, sw, dh, dw, Pattern::Random, n, true);
            RunVarShape<T>(channels, family,
                           {
                               {1081, 1921,  719, 1281},
                               { 720, 1280, 1080, 1920},
                               { 480,  640,  360,  500}
            });
        }
}

TEST(OpTvResize, interleaved_shared_taps_non_u8_realistic)
{
    RunInterleavedRealistic<uint16_t>();
    RunInterleavedRealistic<int16_t>();
    // F16 cubic at realistic magnitudes exceeds the frozen gold tolerance on main too; parity below covers it.
    RunInterleavedRealistic<__half>(false);
    RunInterleavedRealistic<float>();
}

// The per-pixel NHWC kernels must match the per-scalar planar path bit-for-bit, inf/NaN and identity axes included.
template<class T>
void CheckInterleavedMatchesPlanar(Family family, int channels, int sh, int sw, int dh, int dw)
{
    SCOPED_TRACE(::testing::Message() << "mode=" << family.mode << " C=" << channels << " " << sw << "x" << sh << "->"
                                      << dw << "x" << dh << " dtype=" << Dtype<T>());
    Stream        stream;
    GuardedTensor src("NHWC", Dtype<T>(), 1, sh, sw, channels);
    GuardedTensor planarSrc("NCHW", Dtype<T>(), 1, sh, sw, channels);
    GuardedTensor fast("NHWC", Dtype<T>(), 1, dh, dw, channels);
    GuardedTensor planar("NCHW", Dtype<T>(), 1, dh, dw, channels);
    GuardedTensor planarInterleaved("NHWC", Dtype<T>(), 1, dh, dw, channels);
    auto          input      = Input<T>(sh, sw, channels, 77);
    const float   specials[] = {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity(),
                                -std::numeric_limits<float>::infinity()};
    const size_t  marks      = (input.size() + 996) / 997;
    for (size_t k = 0; k < marks; ++k) input[k * 997] = tvgold::Store<T>(specials[k % 3]);
    src.Set(0, input);
    src.Upload();
    cvcuda::Reformat reformat;
    reformat(stream.value, src.tensor, planarSrc.tensor);
    cvcuda::TvResize op;
    auto             req = op.getWorkspaceRequirements(1, sh, sw, dh, dw, channels, family.mode, false);
    auto             ws1 = cvcuda::AllocateWorkspace(req);
    auto             ws2 = cvcuda::AllocateWorkspace(req);
    op(stream.value, ws1.get(), src.tensor, fast.tensor, family.mode, false);
    op(stream.value, ws2.get(), planarSrc.tensor, planar.tensor, family.mode, false);
    reformat(stream.value, planar.tensor, planarInterleaved.tensor);
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream.value));
    fast.Download();
    planarInterleaved.Download();
    auto a = fast.Get<T>(0);
    auto b = planarInterleaved.Get<T>(0);
    ASSERT_EQ(a.size(), b.size());
    EXPECT_EQ(0, std::memcmp(a.data(), b.data(), a.size() * sizeof(T)));
}

TEST(OpTvResize, interleaved_float_nonfinite_matches_planar_bitexactly)
{
    for (auto family : {
             Family{NVCV_INTERP_LINEAR, false},
             Family{ NVCV_INTERP_CUBIC, false}
    })
        for (int channels : {3, 4})
            for (auto [sh, sw, dh, dw] : {
                     std::array{1080, 1920,  720, 1280},
                     std::array{ 720, 1280, 1080, 1920},
                     std::array{ 480,  640,  480,  960},
                     std::array{ 480,  640,  720,  640}
            })
            {
                CheckInterleavedMatchesPlanar<float>(family, channels, sh, sw, dh, dw);
                CheckInterleavedMatchesPlanar<__half>(family, channels, sh, sw, dh, dw);
            }
}

TEST(OpTvResize, aa_interleaved_wide_and_small_batches)
{
    for (auto mode : {NVCV_INTERP_LINEAR, NVCV_INTERP_CUBIC})
        for (auto [sh, sw, dh, dw, n] : {
                 std::array{1080, 1920,  720, 1280, 1},
                 std::array{2160, 3840,  720, 1280, 2},
                 std::array{1080, 1920,  640,  640, 2},
                 std::array{2160, 3840, 1080, 1920, 1},
                 std::array{1080, 1920,  720, 1280, 4}
        })
        {
            for (int channels : {1, 3, 4})
                RunTensor<uint8_t>("NHWC", channels, {mode, true}, sh, sw, dh, dw, Pattern::Random, n, true);
            RunTensor<float>("NHWC", 3, {mode, true}, sh, sw, dh, dw, Pattern::Random, n, true);
        }
}

TEST(OpTvResize, two_pass_aa_realistic)
{
    for (auto [sh, sw, dh, dw, n] : {
             std::array{1080, 1920,  720, 1280, 2},
             std::array{2160, 3840, 1080, 1920, 1},
             std::array{1080, 1920,  224,  224, 3},
             std::array{1081, 1921,  127,  129, 2},
             std::array{2160, 3840,  128,  128, 1}
    })
        for (auto mode : {NVCV_INTERP_LINEAR, NVCV_INTERP_CUBIC})
        {
            for (const char *layout : {"NCHW", "NHWC"})
            {
                RunTensor<uint8_t>(layout, 3, {mode, true}, sh, sw, dh, dw, Pattern::Random, n, true);
                RunTensor<float>(layout, 3, {mode, true}, sh, sw, dh, dw, Pattern::Random, 1, true);
                RunTensor<uint16_t>(layout, 3, {mode, true}, sh, sw, dh, dw, Pattern::Random, 1, true);
            }
            RunTensor<uint8_t>("NHWC", 1, {mode, true}, sh, sw, dh, dw, Pattern::Random, n, true);
        }
}

TEST(OpTvResize, combined_dispatch_boundaries)
{
    RunTensor<uint8_t>("NHWC", 3, {NVCV_INTERP_LINEAR, false}, 48, 66, 32, 44, Pattern::Random, 48, true);
    RunTensor<uint8_t>("NHWC", 3, {NVCV_INTERP_CUBIC, false}, 37, 53, 61, 27, Pattern::Random, 2, true);
    RunTensor<uint8_t>("NHWC", 3, {NVCV_INTERP_CUBIC, false}, 37, 53, 17, 67, Pattern::Random, 2, true);
    RunTensor<uint8_t>("NCHW", 3, {NVCV_INTERP_CUBIC, false}, 37, 53, 61, 67, Pattern::Random, 2, true);
    RunTensor<uint16_t>("NHWC", 3, {NVCV_INTERP_LINEAR, false}, 48, 66, 32, 44, Pattern::Random, 2, true);
}

TEST(OpTvResize, extreme_reduction)
{
    for (auto family : kFamilies)
    {
        RunTensor<uint8_t>("HWC", 1, family, 4096, 4096, 1, 1, Pattern::Checkerboard, 1, false);
        RunTensor<float>("CHW", 1, family, 4096, 4096, 1, 1, Pattern::Constant, 1, false);
    }
}

void RunCubicAntialiasPlanarReuseGeometry(const char *layout, int channels, bool padded)
{
    for (auto geometry : {
             std::array<int, 4>{ 73,  97, 31,   41},
             { 37, 259, 17,   67},
             {  7,  11, 29,   37},
             { 16,  32, 16,    9},
             { 17,  31,  9,   31},
             {257, 259,  4,    7},
             {112, 112, 16,   16},
             {113, 113, 16,   16},
             {  8,   8, 16, 4096},
             {  8,   8, 16, 4097},
             { 73,  97, 31,   44},
             { 37, 259, 17,   68}
    })
        RunTensor<uint8_t>(layout, channels, {NVCV_INTERP_CUBIC, true}, geometry[0], geometry[1], geometry[2],
                           geometry[3], Pattern::Random, 0, padded);
}

TEST(OpTvResize, cubic_antialias_planar_reuse_geometry)
{
    for (const char *layout : {"CHW", "NCHW"})
        for (int channels : {1, 3, 5})
            for (bool padded : {false, true}) RunCubicAntialiasPlanarReuseGeometry(layout, channels, padded);
}

TYPED_TEST(OpTvResize, antialias_packed_paired_rows_match_independent_gold)
{
    for (auto mode : {NVCV_INTERP_LINEAR, NVCV_INTERP_CUBIC})
    {
        Family family{mode, true};
        for (int channels : {1, 3, 4})
        {
            for (const char *layout : {"NHWC", "NCHW"})
            {
                RunTensor<TypeParam>(layout, channels, family, 73, 97, 31, 44, Pattern::Random, 2, false);
                RunTensor<TypeParam>(layout, channels, family, 37, 259, 17, 68, Pattern::Random, 2, true);
            }
            RunVarShape<TypeParam>(channels, family,
                                   {
                                       {73,  97, 31, 44},
                                       {37, 259, 17, 68}
            });
        }
        for (int channels : {5, 17})
            RunTensor<TypeParam>("NCHW", channels, family, 73, 97, 31, 44, Pattern::Random, 2, true);
    }
}

TEST(OpTvResize, cubic_antialias_scalar_multiblock_tail)
{
    RunTensor<uint8_t>("NCHW", 3, {NVCV_INTERP_CUBIC, true}, 73, 97, 31, 513, Pattern::Random, 2, true);
}

TEST(OpTvResize, antialias_varshape_workspace_includes_shared_coefficients)
{
    cvcuda::TvResize             operation;
    const Family                 family{NVCV_INTERP_CUBIC, true};
    const std::array<int32_t, 2> sourceHeights{73, 37};
    const std::array<int32_t, 2> sourceWidths{97, 259};
    const std::array<int32_t, 2> destinationHeights{31, 17};
    const std::array<int32_t, 2> destinationWidths{44, 68};
    for (int channels : {1, 3, 4})
    {
        SCOPED_TRACE(::testing::Message() << "C=" << channels);
        auto   cubic        = operation.getWorkspaceRequirements(2, sourceHeights.data(), sourceWidths.data(),
                                                                 destinationHeights.data(), destinationWidths.data(), channels,
                                                                 NVCV_INTERP_CUBIC, true);
        auto   linear       = operation.getWorkspaceRequirements(2, sourceHeights.data(), sourceWidths.data(),
                                                                 destinationHeights.data(), destinationWidths.data(), channels,
                                                                 NVCV_INTERP_LINEAR, true);
        size_t intermediate = static_cast<size_t>(sourceHeights[0]) * destinationWidths[0] * channels * sizeof(float);
        size_t coefficients = static_cast<size_t>(destinationHeights[0] + destinationWidths[0]) * 34 * sizeof(float);
        EXPECT_EQ(cubic.cudaMem.size, intermediate + coefficients);
        EXPECT_EQ(cubic.cudaMem.size, linear.cudaMem.size);
        auto tensor = operation.getWorkspaceRequirements(1, 73, 97, 31, 44, channels, NVCV_INTERP_CUBIC, true);
        EXPECT_EQ(tensor.cudaMem.size, linear.cudaMem.size);
        RunVarShape<uint8_t>(channels, family,
                             {
                                 {73,  97, 31, 44},
                                 {37, 259, 17, 68}
        });
    }
}

template<class PixelType>
void RunAntialiasEligibleTensorLayouts(Family family)
{
    for (const char *layout : kLayouts)
        for (int channels : {1, 2, 3, 4})
            for (bool padded : {false, true})
                RunTensor<PixelType>(layout, channels, family, 73, 97, 31, 44, Pattern::Random, 0, padded);
}

TYPED_TEST(OpTvResize, antialias_shared_coefficients_eligible_gold)
{
    for (auto mode : {NVCV_INTERP_LINEAR, NVCV_INTERP_CUBIC})
    {
        Family family{mode, true};
        RunAntialiasEligibleTensorLayouts<TypeParam>(family);
        RunTensor<TypeParam>("CHW", 17, family, 37, 259, 17, 68, Pattern::Random);
        RunTensor<TypeParam>("HWC", 3, family, 37, 259, 17, 67, Pattern::Random);
        for (int channels : {1, 3, 4})
            RunVarShape<TypeParam>(channels, family,
                                   {
                                       {73,  97, 31, 44},
                                       {37, 259, 17, 68},
                                       { 8,  11, 29, 37},
                                       {13,  17,  7, 11},
                                       {17,  31,  9, 31},
                                       {23,  19, 23, 19}
            });
    }
}

template<class PixelType>
constexpr Pattern FilteredNumericPattern()
{
    if constexpr (std::is_same_v<PixelType, __half>)
        return Pattern::Subnormal;
    if constexpr (std::is_same_v<PixelType, float>)
        return Pattern::Cancellation;
    return Pattern::Step;
}

TYPED_TEST(OpTvResize, antialias_shared_coefficients_numeric_patterns)
{
    Pattern pattern = FilteredNumericPattern<TypeParam>();
    for (auto mode : {NVCV_INTERP_LINEAR, NVCV_INTERP_CUBIC})
        for (const char *layout : {"NHWC", "NCHW"})
            RunTensor<TypeParam>(layout, 3, {mode, true}, 73, 97, 31, 44, pattern);
}

TYPED_TEST(OpTvResize, non_antialias_fixed_taps_gold)
{
    Pattern pattern = FilteredNumericPattern<TypeParam>();
    for (auto mode : {NVCV_INTERP_LINEAR, NVCV_INTERP_CUBIC})
    {
        Family family{mode, false};
        for (const char *layout : kLayouts)
            for (int channels : {2, 3, 4})
            {
                RunTensor<TypeParam>(layout, channels, family, 73, 97, 31, 44, pattern);
                RunTensor<TypeParam>(layout, channels, family, 19, 23, 37, 53, Pattern::Random);
                RunTensor<TypeParam>(layout, channels, family, 17, 31, 17, 257, Pattern::Random);
                RunTensor<TypeParam>(layout, channels, family, 17, 31, 37, 31, Pattern::Random);
            }
        RunTensor<TypeParam>("CHW", 17, family, 37, 259, 17, 129, pattern);
        for (int channels : {1, 3, 4})
            RunVarShape<TypeParam>(channels, family,
                                   {
                                       {73, 97, 31, 44},
                                       {19, 23, 37, 53},
                                       {17, 31, 17, 37}
            });
    }
}

template<class Scalar>
void CheckNonAAExactIntegerFixture(const char *layout, NVCVInterpolationType mode)
{
    constexpr Scalar    low   = std::numeric_limits<Scalar>::lowest();
    constexpr Scalar    high  = std::numeric_limits<Scalar>::max();
    bool                cubic = mode == NVCV_INTERP_CUBIC;
    Stream              stream;
    cvcuda::TvResize    operation;
    GuardedTensor       input(layout, Dtype<Scalar>(), 1, 1, cubic ? 4 : 2, 3);
    GuardedTensor       output(layout, Dtype<Scalar>(), 1, 1, cubic ? 8 : 1, 3);
    std::vector<Scalar> values
        = cubic ? std::vector<Scalar>{low, low, low, low, low, low, high, high, high, high, high, high}
                : std::vector<Scalar>{0, 1, 2, 1, 2, 3};
    input.Set(0, values);
    output.Set(0, std::vector<Scalar>(output.w * 3));
    input.Upload();
    output.Upload();
    auto workspace
        = cvcuda::AllocateWorkspace(operation.getWorkspaceRequirements(1, 1, input.w, 1, output.w, 3, mode, false));
    operation(stream.value, workspace.get(), input.tensor, output.tensor, mode, false);
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream.value));
    output.Download();
    auto actual = output.Get<Scalar>(0);
    if (cubic)
    {
        for (int channel = 0; channel < 3; ++channel)
        {
            EXPECT_EQ(actual[3 + channel], low);
            EXPECT_EQ(actual[18 + channel], high);
        }
    }
    else
    {
        EXPECT_EQ(actual, (std::vector<Scalar>{0, 2, 2}));
        if constexpr (std::is_signed_v<Scalar>)
        {
            input.Set(0, std::vector<Scalar>{0, 1, -1, 1, 2, 0});
            input.Upload();
            operation(stream.value, workspace.get(), input.tensor, output.tensor, mode, false);
            ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream.value));
            output.Download();
            EXPECT_EQ(output.Get<Scalar>(0), (std::vector<Scalar>{0, 2, 0}));
        }
    }
    output.CheckPadding();
    GuardedTensor twoAxisInput(layout, Dtype<Scalar>(), 1, 2, 2, 3);
    GuardedTensor twoAxisOutput(layout, Dtype<Scalar>(), 1, 1, 1, 3);
    twoAxisInput.Set(0, std::vector<Scalar>{0, 1, 2, 1, 2, 3, 1, 2, 3, 1, 2, 3});
    twoAxisOutput.Set(0, std::vector<Scalar>(3));
    twoAxisInput.Upload();
    twoAxisOutput.Upload();
    auto twoAxisWorkspace
        = cvcuda::AllocateWorkspace(operation.getWorkspaceRequirements(1, 2, 2, 1, 1, 3, mode, false));
    operation(stream.value, twoAxisWorkspace.get(), twoAxisInput.tensor, twoAxisOutput.tensor, mode, false);
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream.value));
    twoAxisOutput.Download();
    EXPECT_EQ(twoAxisOutput.Get<Scalar>(0), (std::vector<Scalar>{1, 2, 3}));
    twoAxisOutput.CheckPadding();
}

template<class Scalar>
void CheckNonAAExactIntegerFixtures()
{
    for (const char *layout : {"NHWC", "NCHW"})
        for (auto mode : {NVCV_INTERP_LINEAR, NVCV_INTERP_CUBIC})
            ASSERT_NO_FATAL_FAILURE(CheckNonAAExactIntegerFixture<Scalar>(layout, mode));
}

TEST(OpTvResizeNonAA, exact_integer_ties_and_cubic_saturation)
{
    CheckNonAAExactIntegerFixtures<uint8_t>();
    CheckNonAAExactIntegerFixtures<uint16_t>();
    CheckNonAAExactIntegerFixtures<int16_t>();
}

template<class PixelType>
void RunFilteredSingletonChannelPixelGaps(const char *layout, Family family)
{
    auto             mode    = family.mode;
    int              samples = layout[0] == 'N' ? 2 : 1;
    Stream           stream;
    cvcuda::TvResize operation;
    GuardedTensor    input(layout, Dtype<PixelType>(), samples, 73, 97, 1);
    GuardedTensor    output(layout, Dtype<PixelType>(), samples, 31, 44, 1);
    GuardedTensor    gappedInput(layout, Dtype<PixelType>(), samples, 73, 194, 1);
    GuardedTensor    gappedOutput(layout, Dtype<PixelType>(), samples, 31, 88, 1);
    auto             wrap = [&](GuardedTensor &storage, int height, int width)
    {
        NVCVTensorBufferStrided buffer{};
        buffer.basePtr = reinterpret_cast<NVCVByte *>(storage.allocation + storage.baseOffset);
        nvcv::TensorLayout     tensorLayout(layout);
        std::array<int64_t, 4> shape{};
        for (int axis = 0; axis < tensorLayout.rank(); ++axis)
        {
            shape[axis]          = AxisExtent(tensorLayout[axis], samples, height, width, 1);
            buffer.strides[axis] = AxisExtent(tensorLayout[axis], storage.sample, storage.row, 2 * sizeof(PixelType),
                                              2 * sizeof(PixelType));
        }
        return nvcv::TensorWrapData(nvcv::TensorDataStridedCuda(
            nvcv::TensorShape(shape.data(), tensorLayout.rank(), tensorLayout), Dtype<PixelType>(), buffer));
    };
    auto source      = wrap(gappedInput, 73, 97);
    auto destination = wrap(gappedOutput, 31, 44);
    for (int sample = 0; sample < samples; ++sample)
    {
        auto values = Input<PixelType>(73, 97, 1, 1234 + sample);
        input.Set(sample, values);
        output.Set(sample, std::vector<PixelType>(31 * 44));
        for (int row = 0; row < 73; ++row)
            for (int column = 0; column < 97; ++column)
            {
                auto offset = gappedInput.Offset(sample, row, 2 * column, 0);
                std::memcpy(gappedInput.host.data() + offset, &values[row * 97 + column], sizeof(PixelType));
                std::fill_n(gappedInput.logical.begin() + offset, sizeof(PixelType), true);
            }
        for (int row = 0; row < 31; ++row)
            for (int column = 0; column < 44; ++column)
            {
                auto offset = gappedOutput.Offset(sample, row, 2 * column, 0);
                std::fill_n(gappedOutput.logical.begin() + offset, sizeof(PixelType), true);
            }
    }
    input.Upload();
    output.Upload();
    gappedInput.Upload();
    const auto originalInput = gappedInput.host;
    auto       requirements  = operation.getWorkspaceRequirements(samples, 73, 97, 31, 44, 1, mode, family.aa);
    requirements.cudaMem.size += sizeof(float);
    auto workspace = cvcuda::AllocateWorkspace(requirements);
    operation(stream.value, workspace.get(), input.tensor, output.tensor, mode, family.aa);
    auto scalarWorkspace         = workspace.get();
    scalarWorkspace.cudaMem.data = static_cast<unsigned char *>(scalarWorkspace.cudaMem.data) + sizeof(float);
    scalarWorkspace.cudaMem.req.size -= sizeof(float);
    scalarWorkspace.cudaMem.req.alignment = alignof(float);
    operation(stream.value, scalarWorkspace, source, destination, mode, family.aa);
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream.value));
    output.Download();
    gappedInput.Download();
    gappedOutput.Download();
    EXPECT_EQ(originalInput, gappedInput.host);
    for (int sample = 0; sample < samples; ++sample)
    {
        auto expected = output.Get<PixelType>(sample);
        auto physical = gappedOutput.Get<PixelType>(sample);
        for (size_t index = 0; index < expected.size(); ++index)
            EXPECT_EQ(0, std::memcmp(&expected[index], &physical[2 * index], sizeof(PixelType)));
    }
    output.CheckPadding();
    gappedInput.CheckPadding();
    gappedOutput.CheckPadding();
}

TYPED_TEST(OpTvResize, filtered_singleton_channel_pixel_gaps_preserve_sentinels)
{
    for (const char *layout : {"HWC", "NHWC"})
        for (Family family : {
                 Family{NVCV_INTERP_LINEAR, false},
                 Family{ NVCV_INTERP_CUBIC, false},
                 Family{NVCV_INTERP_LINEAR,  true},
                 Family{ NVCV_INTERP_CUBIC,  true}
        })
            ASSERT_NO_FATAL_FAILURE(RunFilteredSingletonChannelPixelGaps<TypeParam>(layout, family));
}

TEST(OpTvResize, cubic_antialias_small_or_thin_planes_do_not_reserve_coefficients)
{
    cvcuda::TvResize operation;
    for (auto geometry : {
             std::array<int32_t, 4>{ 1,  1,   2,   2},
             { 7,  8,  16,  16},
             { 8,  7,  16,  16},
             { 8,  8,  15,  16},
             { 8,  8,  16,  15},
             { 1, 64,   2, 128},
             {64,  1, 128,   2},
             { 2, 64,   4, 128},
             {64,  2, 128,   4}
    })
    {
        SCOPED_TRACE(::testing::Message()
                     << geometry[0] << "x" << geometry[1] << "->" << geometry[2] << "x" << geometry[3]);
        for (int batch : {1, 65535})
        {
            auto requirements = operation.getWorkspaceRequirements(batch, geometry[0], geometry[1], geometry[2],
                                                                   geometry[3], 1, NVCV_INTERP_CUBIC, true);
            EXPECT_EQ(static_cast<size_t>(batch) * geometry[0] * geometry[3] * sizeof(float),
                      requirements.cudaMem.size);
        }
        auto requirements = operation.getWorkspaceRequirements(1, &geometry[0], &geometry[1], &geometry[2],
                                                               &geometry[3], 1, NVCV_INTERP_CUBIC, true);
        EXPECT_EQ(static_cast<size_t>(geometry[0]) * geometry[3] * sizeof(float), requirements.cudaMem.size);
    }
    auto requirements = operation.getWorkspaceRequirements(1, 8, 8, 16, 16, 1, NVCV_INTERP_CUBIC, true);
    EXPECT_GT(requirements.cudaMem.size, 8 * 16 * sizeof(float));
    RunTensor<uint8_t>("NHWC", 1, {NVCV_INTERP_CUBIC, true}, 1, 64, 2, 128, Pattern::Random, 2, true);
    RunTensor<uint8_t>("NHWC", 1, {NVCV_INTERP_CUBIC, true}, 64, 1, 128, 2, Pattern::Random, 2, true);
    RunTensor<uint8_t>("NCHW", 3, {NVCV_INTERP_CUBIC, true}, 7, 7, 14, 14, Pattern::Random, 2, true);
    RunTensor<uint8_t>("NCHW", 3, {NVCV_INTERP_CUBIC, true}, 8, 8, 16, 16, Pattern::Random, 2, true);
}

void ExpectRejectedUnchanged(cvcuda::TvResize &op, Stream &stream, GuardedTensor &source, GuardedTensor &destination,
                             Family family, bool varshape)
{
    const auto before = destination.host;
    auto ws = cvcuda::AllocateWorkspace(op.getWorkspaceRequirements(1, source.h, source.w, destination.h, destination.w,
                                                                    source.c, family.mode, family.aa));
    NVCVStatus status;
    if (varshape)
    {
        nvcv::ImageBatchVarShape input(1);
        nvcv::ImageBatchVarShape output(1);
        input.pushBack(source.AsImage(Format(source.tensor.dtype(), source.c)));
        output.pushBack(destination.AsImage(Format(destination.tensor.dtype(), destination.c)));
        status = cvcudaTvResizeImageBatchSubmit(op.handle(), stream.value, &ws.get(), input.handle(), output.handle(),
                                                family.mode, family.aa);
    }
    else
        status = cvcudaTvResizeSubmit(op.handle(), stream.value, &ws.get(), source.tensor.handle(),
                                      destination.tensor.handle(), family.mode, family.aa);
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, status);
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream.value));
    destination.Download();
    EXPECT_EQ(before, destination.host);
}

TEST(OpTvResize, cubic_antialias_workspace_alignment_paths_match_exactly)
{
    Stream           stream;
    cvcuda::TvResize operation;
    GuardedTensor    input("NCHW", nvcv::TYPE_U8, 2, 73, 97, 3);
    GuardedTensor    alignedOutput("NCHW", nvcv::TYPE_U8, 2, 31, 44, 3);
    GuardedTensor    scalarOutput("NCHW", nvcv::TYPE_U8, 2, 31, 44, 3);
    for (int sample = 0; sample < 2; ++sample)
    {
        input.Set(sample, Input<uint8_t>(73, 97, 3, 1234 + sample, Pattern::Random));
        alignedOutput.Set(sample, std::vector<uint8_t>(31 * 44 * 3));
        scalarOutput.Set(sample, std::vector<uint8_t>(31 * 44 * 3));
    }
    input.Upload();
    alignedOutput.Upload();
    scalarOutput.Upload();
    auto requirements = operation.getWorkspaceRequirements(2, 73, 97, 31, 44, 3, NVCV_INTERP_CUBIC, true);
    requirements.cudaMem.size += 16;
    auto storage = cvcuda::AllocateWorkspace(requirements);
    operation(stream.value, storage.get(), input.tensor, alignedOutput.tensor, NVCV_INTERP_CUBIC, true);
    auto misaligned         = storage.get();
    misaligned.cudaMem.data = static_cast<unsigned char *>(misaligned.cudaMem.data) + sizeof(float);
    misaligned.cudaMem.req.size -= sizeof(float);
    misaligned.cudaMem.req.alignment = alignof(float);
    operation(stream.value, misaligned, input.tensor, scalarOutput.tensor, NVCV_INTERP_CUBIC, true);
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream.value));
    alignedOutput.Download();
    scalarOutput.Download();
    for (int sample = 0; sample < 2; ++sample)
        EXPECT_EQ(alignedOutput.Get<uint8_t>(sample), scalarOutput.Get<uint8_t>(sample));
    alignedOutput.CheckPadding();
    scalarOutput.CheckPadding();
}

TEST(OpTvResize, cubic_antialias_byte_aligned_destination_matches_gold)
{
    Stream           stream;
    cvcuda::TvResize operation;
    GuardedTensor    input("NCHW", nvcv::TYPE_U8, 2, 73, 97, 3);
    for (int sample = 0; sample < 2; ++sample)
        input.Set(sample, Input<uint8_t>(73, 97, 3, 1234 + sample, Pattern::Random));
    input.Upload();
    auto storage
        = cvcuda::AllocateWorkspace(operation.getWorkspaceRequirements(2, 73, 97, 31, 44, 3, NVCV_INTERP_CUBIC, true));
    ASSERT_EQ(0u, reinterpret_cast<uintptr_t>(storage.get().cudaMem.data) % alignof(float4));
    for (auto alignment : {
             std::array<int, 2>{1, 0},
             {0, 1}
    })
    {
        SCOPED_TRACE(::testing::Message() << "byte offset=" << alignment[0] << " row padding=" << alignment[1]);
        GuardedTensor output("NCHW", nvcv::TYPE_U8, 2, 31, 44, 3, true, alignment[0], alignment[1]);
        for (int sample = 0; sample < 2; ++sample) output.Set(sample, std::vector<uint8_t>(31 * 44 * 3));
        operation(stream.value, storage.get(), input.tensor, output.tensor, NVCV_INTERP_CUBIC, true);
        ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream.value));
        output.Download();
        for (int sample = 0; sample < 2; ++sample)
        {
            auto gold   = tvgold::Resize(input.Get<uint8_t>(sample), 73, 97, 31, 44, 3, NVCV_INTERP_CUBIC, true);
            auto actual = output.Get<uint8_t>(sample);
            EXPECT_TRUE(tvgold::WithinTolerance(gold, actual, NVCV_INTERP_CUBIC, true))
                << "sample=" << sample << " " << DifferenceContext(gold, actual);
        }
        output.CheckPadding();
    }
}

TEST(OpTvResize_Negative, dtype_mismatch_rejected_before_any_write)
{
    Stream                                                         stream;
    cvcuda::TvResize                                               op;
    const std::array<std::pair<nvcv::DataType, nvcv::DataType>, 3> pairs{
        {{nvcv::TYPE_U8, nvcv::TYPE_F32}, {nvcv::TYPE_F32, nvcv::TYPE_U8}, {nvcv::TYPE_U16, nvcv::TYPE_S16}}
    };
    for (auto [sourceType, destinationType] : pairs)
        for (auto family : kFamilies)
            for (bool varshape : {false, true})
            {
                SCOPED_TRACE(::testing::Message()
                             << "mode=" << family.mode << " AA=" << family.aa << " varshape=" << varshape);
                GuardedTensor source("HWC", sourceType, 1, 113, 97, 3);
                GuardedTensor destination("HWC", destinationType, 1, 53, 71, 3);
                ExpectRejectedUnchanged(op, stream, source, destination, family, varshape);
            }
}

void CheckInvalidTensorMetadata(cvcuda::TvResize &op, Stream &stream, bool badInput, int fault)
{
    auto                                badLayout = fault == 1 ? "NCHW" : "NHWC";
    const std::array<nvcv::DataType, 7> types{nvcv::TYPE_U8,  nvcv::TYPE_U8,  nvcv::TYPE_U8, nvcv::TYPE_U8,
                                              nvcv::TYPE_S32, nvcv::TYPE_3U8, nvcv::TYPE_U8};
    const std::array<int, 7>            channels{3, 3, 4, 5, 3, 3, 3};
    auto                                badType = types[fault];
    int                                 badC    = channels[fault];
    int                                 badN    = fault == 0 ? 2 : 1;
    GuardedTensor                       normal("NHWC", nvcv::TYPE_U8, 1, 7, 11, 3);
    GuardedTensor                       invalid(badLayout, badType, badN, 7, 11, badC);
    if (fault == 6)
    {
        auto                    data = invalid.tensor.exportData<nvcv::TensorDataStridedCuda>();
        NVCVTensorBufferStrided buffer{};
        buffer.basePtr = reinterpret_cast<NVCVByte *>(data->basePtr());
        for (int i = 0; i < 4; ++i) buffer.strides[i] = data->stride(i);
        invalid.tensor = nvcv::TensorWrapData(nvcv::TensorDataStridedCuda(
            nvcv::TensorShape{
                {1, 7, 11, 3},
                "NWHC"
        },
            nvcv::TYPE_U8, buffer));
    }
    auto &src = badInput ? invalid : normal;
    auto &dst = badInput ? normal : invalid;
    ExpectRejectedUnchanged(op, stream, src, dst, {NVCV_INTERP_LINEAR, true}, false);
}

TEST(OpTvResize_Negative, tensor_metadata_branch_basis_both_operands)
{
    Stream           stream;
    cvcuda::TvResize op;
    for (bool badInput : {false, true})
        for (int fault = 0; fault < 7; ++fault)
            ASSERT_NO_FATAL_FAILURE(CheckInvalidTensorMetadata(op, stream, badInput, fault));
    for (auto dtype : {nvcv::TYPE_2F16, nvcv::TYPE_3U8, nvcv::TYPE_4F32})
        for (auto layout : kLayouts)
        {
            GuardedTensor src(layout, dtype, 1, 9, 13, 1);
            GuardedTensor dst(layout, dtype, 1, 5, 7, 1);
            ExpectRejectedUnchanged(op, stream, src, dst, {NVCV_INTERP_NEAREST, false}, false);
            ExpectRejectedUnchanged(op, stream, src, dst, {NVCV_INTERP_CUBIC, true}, false);
        }
}

void CheckInvalidVarShapeMetadata(cvcuda::TvResize &op, Stream &stream, bool badInput, int fault)
{
    nvcv::ImageBatchVarShape               input(2);
    nvcv::ImageBatchVarShape               output(2);
    auto                                   good = nvcv::FMT_RGB8;
    const std::array<nvcv::ImageFormat, 5> badFormats{nvcv::FMT_2F32, nvcv::FMT_RGB8p, nvcv::FMT_S32, nvcv::FMT_RGBA8,
                                                      nvcv::FMT_RGBA8};
    auto                                   bad = badFormats[fault];
    input.pushBack(nvcv::Image({17, 13}, badInput && fault < 3 ? bad : good));
    output.pushBack(nvcv::Image({11, 7}, !badInput && fault < 3 ? bad : good));
    if (fault >= 3)
    {
        if (badInput)
            input.pushBack(nvcv::Image({9, 1}, bad));
        else
            output.pushBack(nvcv::Image({5, 4}, bad));
        if (fault == 3)
        {
            if (badInput)
                output.pushBack(nvcv::Image({5, 4}, good));
            else
                input.pushBack(nvcv::Image({9, 1}, good));
        }
    }
    for (bool aa : {false, true})
    {
        auto ws = cvcuda::AllocateWorkspace(op.getWorkspaceRequirements(2, 13, 17, 7, 11, 3, NVCV_INTERP_LINEAR, aa));
        EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
                  cvcudaTvResizeImageBatchSubmit(op.handle(), stream.value, &ws.get(), input.handle(), output.handle(),
                                                 NVCV_INTERP_LINEAR, aa));
        ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream.value));
    }
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream.value));
}

TEST(OpTvResize_Negative, varshape_formats_counts_multiplane_both_operands)
{
    Stream           stream;
    cvcuda::TvResize op;
    for (bool badInput : {false, true})
        for (int fault = 0; fault < 5; ++fault)
            ASSERT_NO_FATAL_FAILURE(CheckInvalidVarShapeMetadata(op, stream, badInput, fault));
}

void CheckInvalidWorkspaceExtents(cvcuda::TvResize &op, NVCVInterpolationType mode, bool aa, int n)
{
    NVCVWorkspaceRequirements req{};
    for (int axis = 0; axis < 5; ++axis)
        for (int bad : {0, -1})
        {
            std::array<int32_t, 5> dims{2, 2, 2, 2, 3};
            dims[axis] = bad;
            EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
                      cvcudaTvResizeTensorGetWorkspaceRequirements(op.handle(), n, dims[0], dims[1], dims[2], dims[3],
                                                                   dims[4], mode, aa, &req));
            if (n || axis == 4)
            {
                EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
                          cvcudaTvResizeImageBatchGetWorkspaceRequirements(op.handle(), n, &dims[0], &dims[1], &dims[2],
                                                                           &dims[3], dims[4], mode, aa, &req));
            }
        }
}

void CheckWorkspaceQueryDimensions(cvcuda::TvResize &op, NVCVInterpolationType mode, bool aa,
                                   const std::vector<int32_t> &extents)
{
    NVCVWorkspaceRequirements req{};
    for (int n : {-1, 65536})
    {
        EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
                  cvcudaTvResizeTensorGetWorkspaceRequirements(op.handle(), n, 2, 2, 2, 2, 3, mode, aa, &req));
        EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
                  cvcudaTvResizeImageBatchGetWorkspaceRequirements(op.handle(), n, extents.data(), extents.data(),
                                                                   extents.data(), extents.data(), 3, mode, aa, &req));
    }
    for (int n : {0, 1}) CheckInvalidWorkspaceExtents(op, mode, aa, n);
    for (int n : {0, 1})
        for (int axis = 0; axis < 4; ++axis)
        {
            std::array<const int32_t *, 4> arrays{extents.data(), extents.data(), extents.data(), extents.data()};
            arrays[axis] = nullptr;
            EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
                      cvcudaTvResizeImageBatchGetWorkspaceRequirements(op.handle(), n, arrays[0], arrays[1], arrays[2],
                                                                       arrays[3], 3, mode, aa, &req));
        }
    EXPECT_EQ(NVCV_SUCCESS,
              cvcudaTvResizeTensorGetWorkspaceRequirements(op.handle(), 65535, 2, 2, 2, 2, 4, mode, aa, &req));
    EXPECT_EQ(NVCV_SUCCESS,
              cvcudaTvResizeImageBatchGetWorkspaceRequirements(op.handle(), 65535, extents.data(), extents.data(),
                                                               extents.data(), extents.data(), 4, mode, aa, &req));
}

TEST(OpTvResize_Negative, queries_validate_before_shortcuts)
{
    cvcuda::TvResize           op;
    NVCVWorkspaceRequirements  req{};
    const std::vector<int32_t> extents(65536, 2);
    for (auto mode : {NVCV_INTERP_NEAREST, NVCV_INTERP_NEAREST_FLOOR, NVCV_INTERP_LINEAR, NVCV_INTERP_CUBIC})
        for (bool aa : {false, true}) ASSERT_NO_FATAL_FAILURE(CheckWorkspaceQueryDimensions(op, mode, aa, extents));
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              cvcudaTvResizeTensorGetWorkspaceRequirements(op.handle(), 65535, INT_MAX, INT_MAX, INT_MAX, INT_MAX,
                                                           INT_MAX, NVCV_INTERP_CUBIC, true, &req));
}

void CheckWorkspaceQueryChannels(cvcuda::TvResize &op, NVCVInterpolationType mode, bool aa, int n, int channels)
{
    const int32_t srcExtent = 3;
    const int32_t dstExtent = 2;
    SCOPED_TRACE(::testing::Message() << "mode=" << mode << " aa=" << aa << " n=" << n << " channels=" << channels);
    NVCVWorkspaceRequirements tensorReq{};
    NVCVWorkspaceRequirements batchReq{};
    EXPECT_EQ(NVCV_SUCCESS,
              cvcudaTvResizeTensorGetWorkspaceRequirements(op.handle(), n, srcExtent, srcExtent, dstExtent, dstExtent,
                                                           channels, mode, aa, &tensorReq));
    auto expected = channels == 2 || channels == 5 ? NVCV_ERROR_INVALID_ARGUMENT : NVCV_SUCCESS;
    EXPECT_EQ(expected,
              cvcudaTvResizeImageBatchGetWorkspaceRequirements(op.handle(), n, &srcExtent, &srcExtent, &dstExtent,
                                                               &dstExtent, channels, mode, aa, &batchReq));
    if (n == 0)
    {
        EXPECT_EQ(0u, tensorReq.hostMem.size + tensorReq.pinnedMem.size + tensorReq.cudaMem.size);
        if (expected == NVCV_SUCCESS)
        {
            EXPECT_EQ(0u, batchReq.hostMem.size + batchReq.pinnedMem.size + batchReq.cudaMem.size);
        }
    }
}

TEST(OpTvResize_Negative, workspace_queries_enforce_channel_contracts)
{
    cvcuda::TvResize op;
    for (auto [mode, aa] : kQueryFamilies)
        for (int n : {0, 1})
            for (int channels : {1, 2, 3, 4, 5})
                ASSERT_NO_FATAL_FAILURE(CheckWorkspaceQueryChannels(op, mode, aa, n, channels));
}

TEST(OpTvResize_Negative, nulls_wrong_operator_and_unsupported_modes)
{
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, cvcudaTvResizeCreate(nullptr));
    cvcuda::TvResize          op;
    cvcuda::Reformat          wrong;
    int32_t                   extent = 3;
    NVCVWorkspaceRequirements req{};
    for (auto handle : {NVCVOperatorHandle(nullptr), wrong.handle()})
    {
        SCOPED_TRACE(handle ? "wrong operator: ToDynamicRef" : "null operator: ToDynamicRef");
        auto expected = handle ? NVCV_ERROR_NOT_COMPATIBLE : NVCV_ERROR_INVALID_ARGUMENT;
        EXPECT_EQ(expected, cvcudaTvResizeTensorGetWorkspaceRequirements(handle, 1, 3, 3, 2, 2, 1, NVCV_INTERP_LINEAR,
                                                                         false, &req));
        EXPECT_EQ(expected, cvcudaTvResizeImageBatchGetWorkspaceRequirements(
                                handle, 1, &extent, &extent, &extent, &extent, 1, NVCV_INTERP_LINEAR, false, &req));
    }
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, cvcudaTvResizeTensorGetWorkspaceRequirements(
                                               op.handle(), 1, 3, 3, 2, 2, 1, NVCV_INTERP_LINEAR, false, nullptr));
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              cvcudaTvResizeImageBatchGetWorkspaceRequirements(op.handle(), 1, &extent, &extent, &extent, &extent, 1,
                                                               NVCV_INTERP_LINEAR, false, nullptr));
    GuardedTensor src("HWC", nvcv::TYPE_U8, 1, 3, 3, 1);
    GuardedTensor dst("HWC", nvcv::TYPE_U8, 1, 2, 2, 1);
    for (bool nullInput : {false, true})
    {
        SCOPED_TRACE(nullInput ? "null input: Tensor::exportData" : "null output: Tensor::exportData");
        EXPECT_EQ(NVCV_ERROR_INVALID_OPERATION,
                  cvcudaTvResizeSubmit(op.handle(), nullptr, nullptr, nullInput ? nullptr : src.tensor.handle(),
                                       nullInput ? dst.tensor.handle() : nullptr, NVCV_INTERP_NEAREST, false));
    }
    for (auto mode : {NVCV_INTERP_AREA, NVCV_INTERP_LANCZOS, NVCV_INTERP_GAUSSIAN, NVCV_INTERP_MAX})
    {
        SCOPED_TRACE(::testing::Message() << "unsupported mode=" << mode);
        EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
                  cvcudaTvResizeTensorGetWorkspaceRequirements(op.handle(), 0, 3, 3, 2, 2, 1, mode, false, &req));
        EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, cvcudaTvResizeSubmit(op.handle(), nullptr, nullptr, src.tensor.handle(),
                                                                    dst.tensor.handle(), mode, false));
    }
    dst.Download();
    EXPECT_TRUE(std::all_of(dst.host.begin(), dst.host.end(), [](uint8_t value) { return value == 0xa5; }));
}

TEST(OpTvResize, matched_empty_has_zero_workspace_and_no_writes)
{
    Stream           stream;
    cvcuda::TvResize op;
    int32_t          sentinel = 1;
    for (auto family : kFamilies)
    {
        auto req = op.getWorkspaceRequirements(0, 13, 17, 7, 11, 3, family.mode, family.aa);
        EXPECT_EQ(0u, req.hostMem.size + req.pinnedMem.size + req.cudaMem.size);
        auto batchReq
            = op.getWorkspaceRequirements(0, &sentinel, &sentinel, &sentinel, &sentinel, 3, family.mode, family.aa);
        EXPECT_EQ(0u, batchReq.hostMem.size + batchReq.pinnedMem.size + batchReq.cudaMem.size);
        nvcv::ImageBatchVarShape src(1);
        nvcv::ImageBatchVarShape dst(1);
        EXPECT_EQ(NVCV_SUCCESS, cvcudaTvResizeImageBatchSubmit(op.handle(), stream.value, nullptr, src.handle(),
                                                               dst.handle(), family.mode, family.aa));
        ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream.value));
    }
}

TEST(OpTvResize, actual_grid_boundary_launches_write_last_sample_and_plane)
{
    for (auto family : kFamilies)
        for (bool planar : {false, true})
            RunTensor<uint8_t>(planar ? "NCHW" : "NHWC", planar ? 3 : 1, family, 1, 1, 2, 2, Pattern::Ramp,
                               planar ? 21845 : 65535, false);
}

nvcv::Tensor MetadataTensor(const char *layout, int64_t n, int64_t h, int64_t w, int64_t c, uint8_t *pointer,
                            int64_t badInnerStride = 0)
{
    nvcv::TensorLayout      l(layout);
    std::array<int64_t, 4>  dims{};
    NVCVTensorBufferStrided buffer{};
    buffer.basePtr = reinterpret_cast<NVCVByte *>(pointer);
    int64_t stride = 1;
    for (int i = l.rank() - 1; i >= 0; --i)
    {
        dims[i]           = AxisExtent(l[i], n, h, w, c);
        buffer.strides[i] = stride;
        stride *= dims[i];
    }
    if (badInnerStride)
        buffer.strides[l.rank() - 2] = badInnerStride;
    return nvcv::TensorWrapData(
        nvcv::TensorDataStridedCuda(nvcv::TensorShape(dims.data(), l.rank(), l), nvcv::TYPE_U8, buffer));
}

TEST(OpTvResize_Negative, wide_grid_metadata_rejected_before_narrowing)
{
    cvcuda::TvResize op;
    GuardedTensor    storage("NHWC", nvcv::TYPE_U8, 1, 1, 1, 1);
    for (auto dims : {
             std::array<int64_t, 2>{              65536,     1},
             {INT64_C(4294967297),     1},
             {              16384,     4},
             {                  1, 65536}
    })
        for (bool badInput : {false, true})
        {
            auto        invalid = MetadataTensor("NCHW", dims[0], 1, 1, dims[1], storage.allocation + 256);
            const auto &src     = badInput ? invalid : storage.tensor;
            const auto &dst     = badInput ? storage.tensor : invalid;
            EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, cvcudaTvResizeSubmit(op.handle(), nullptr, nullptr, src.handle(),
                                                                        dst.handle(), NVCV_INTERP_NEAREST, false));
        }
    storage.Download();
    EXPECT_TRUE(std::all_of(storage.host.begin(), storage.host.end(), [](uint8_t x) { return x == 0xa5; }));
}

bool CorruptWorkspaceMemory(NVCVWorkspaceMem &memory, int fault)
{
    if (!memory.req.size)
        return false;
    switch (fault)
    {
    case 0:
        --memory.req.size;
        return true;
    case 1:
        memory.data = nullptr;
        return true;
    case 2:
        if (memory.req.alignment <= 1)
            return false;
        memory.data = static_cast<uint8_t *>(memory.data) + 1;
        return true;
    default:
        return false;
    }
}

void CheckWorkspaceStorage(cvcuda::TvResize &op, Stream &stream, Family family, bool varshape)
{
    GuardedTensor src("HWC", nvcv::TYPE_U8, 1, 13, 17, 3);
    GuardedTensor dst("HWC", nvcv::TYPE_U8, 1, 7, 11, 3);
    src.Set(0, Input<uint8_t>(13, 17, 3, 0));
    src.Upload();
    int32_t sh  = 13;
    int32_t sw  = 17;
    int32_t dh  = 7;
    int32_t dw  = 11;
    auto    req = varshape ? op.getWorkspaceRequirements(1, &sh, &sw, &dh, &dw, 3, family.mode, family.aa)
                           : op.getWorkspaceRequirements(1, sh, sw, dh, dw, 3, family.mode, family.aa);
    auto    ws  = cvcuda::AllocateWorkspace(req);
    nvcv::ImageBatchVarShape input(1);
    nvcv::ImageBatchVarShape output(1);
    input.pushBack(src.AsImage(nvcv::FMT_RGB8));
    output.pushBack(dst.AsImage(nvcv::FMT_RGB8));
    auto submit = [&op, &stream, &src, &dst, &input, &output, family, varshape](const NVCVWorkspace *space)
    {
        return varshape ? cvcudaTvResizeImageBatchSubmit(op.handle(), stream.value, space, input.handle(),
                                                         output.handle(), family.mode, family.aa)
                        : cvcudaTvResizeSubmit(op.handle(), stream.value, space, src.tensor.handle(),
                                               dst.tensor.handle(), family.mode, family.aa);
    };
    bool required = req.hostMem.size || req.pinnedMem.size || req.cudaMem.size;
    EXPECT_EQ(required ? NVCV_ERROR_INVALID_ARGUMENT : NVCV_SUCCESS, submit(nullptr));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream.value));
    if (required)
    {
        dst.Download();
        EXPECT_TRUE(std::all_of(dst.host.begin(), dst.host.end(), [](uint8_t x) { return x == 0xa5; }));
    }
    for (int kind = 0; kind < 3; ++kind)
        for (int fault = 0; fault < 3; ++fault)
        {
            auto                              broken = ws.get();
            std::array<NVCVWorkspaceMem *, 3> memories{&broken.hostMem, &broken.pinnedMem, &broken.cudaMem};
            if (auto &memory = *memories[kind]; !CorruptWorkspaceMemory(memory, fault))
                continue;
            EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, submit(&broken));
            ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream.value));
        }
    EXPECT_EQ(NVCV_SUCCESS, submit(&ws.get()));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream.value));
    dst.Download();
    auto gold = tvgold::Resize(src.Get<uint8_t>(0), 13, 17, 7, 11, 3, family.mode, family.aa);
    EXPECT_TRUE(tvgold::WithinTolerance(gold, dst.Get<uint8_t>(0), family.mode, family.aa));
}

TEST(OpTvResize, workspace_query_submit_agreement_and_required_storage)
{
    Stream           stream;
    cvcuda::TvResize op;
    for (auto family : kFamilies)
        for (bool varshape : {false, true})
            ASSERT_NO_FATAL_FAILURE(CheckWorkspaceStorage(op, stream, family, varshape));
}

TEST(OpTvResize_Negative, coefficient_tail_workspace_truncation_rejected_before_write)
{
    Stream           stream;
    cvcuda::TvResize operation;
    for (auto mode : {NVCV_INTERP_LINEAR, NVCV_INTERP_CUBIC})
        for (int channels : {1, 3, 4})
            for (bool varshape : {false, true})
            {
                SCOPED_TRACE(::testing::Message() << "VarShape=" << varshape);
                int32_t       sourceHeight      = 73;
                int32_t       sourceWidth       = 97;
                int32_t       destinationHeight = 31;
                int32_t       destinationWidth  = 44;
                GuardedTensor source("HWC", nvcv::TYPE_U8, 1, sourceHeight, sourceWidth, channels);
                GuardedTensor destination("HWC", nvcv::TYPE_U8, 1, destinationHeight, destinationWidth, channels);
                source.Set(0, Input<uint8_t>(sourceHeight, sourceWidth, channels, 1234, Pattern::Random));
                destination.Set(0, std::vector<uint8_t>(destinationHeight * destinationWidth * channels, 0x5a));
                source.Upload();
                destination.Upload();
                const auto before = destination.host;
                auto       requirements
                    = varshape ? operation.getWorkspaceRequirements(1, &sourceHeight, &sourceWidth, &destinationHeight,
                                                                    &destinationWidth, channels, mode, true)
                               : operation.getWorkspaceRequirements(1, sourceHeight, sourceWidth, destinationHeight,
                                                                    destinationWidth, channels, mode, true);
                size_t intermediate = static_cast<size_t>(sourceHeight) * destinationWidth * channels * sizeof(float);
                ASSERT_GT(requirements.cudaMem.size, intermediate);
                auto storage           = cvcuda::AllocateWorkspace(requirements);
                auto exact             = storage.get();
                exact.cudaMem.req.size = requirements.cudaMem.size;
                auto truncated         = exact;
                --truncated.cudaMem.req.size;
                ASSERT_GE(truncated.cudaMem.req.size, intermediate);
                nvcv::ImageBatchVarShape input(1);
                nvcv::ImageBatchVarShape output(1);
                input.pushBack(source.AsImage(Format(nvcv::TYPE_U8, channels)));
                output.pushBack(destination.AsImage(Format(nvcv::TYPE_U8, channels)));
                auto submit = [&](const NVCVWorkspace &workspace)
                {
                    return varshape
                             ? cvcudaTvResizeImageBatchSubmit(operation.handle(), stream.value, &workspace,
                                                              input.handle(), output.handle(), mode, true)
                             : cvcudaTvResizeSubmit(operation.handle(), stream.value, &workspace,
                                                    source.tensor.handle(), destination.tensor.handle(), mode, true);
                };
                EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, submit(truncated));
                ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream.value));
                destination.Download();
                EXPECT_EQ(before, destination.host);
                EXPECT_EQ(NVCV_SUCCESS, submit(exact));
                ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream.value));
                destination.Download();
                auto gold = tvgold::Resize(source.Get<uint8_t>(0), sourceHeight, sourceWidth, destinationHeight,
                                           destinationWidth, channels, mode, true);
                EXPECT_TRUE(tvgold::WithinTolerance(gold, destination.Get<uint8_t>(0), mode, true));
                destination.CheckPadding();
            }
}

template<class T>
void RunFrozenValidGeometryParameterizations(bool padded)
{
    // Geometry/mode/N authority: frozen1848 TestOpTvResize.cpp at dc3e476fb0023ab7f09f80ab2c90312ef2301570.
    // Expanded across all scalar types/channels, without importing its non-independent gold or production code.
    struct Case
    {
        int sw;
        int sh;
        int dw;
        int dh;
        int family;
        int n;
    };

    const std::vector<Case> cases{
        { 42,  48,  23,  24, 0, 1},
        { 23,  24,  42,  48, 0, 2},
        { 42,  48,  23,  24, 1, 1},
        { 23,  24,  42,  48, 1, 1},
        {104,  97,  97,  71, 0, 1},
        {143,  97,  33,  71, 1, 1},
        {113,  97,  53,  71, 2, 1},
        { 53,  71, 113,  97, 2, 2},
        {113,  97,  53,  71, 3, 1},
        { 53,  71, 113,  97, 3, 1},
        {113,  97,  53,  71, 4, 1},
        { 53,  71, 113,  97, 4, 1},
        {113,  97,  53,  71, 5, 1},
        { 53,  71, 113,  97, 5, 2},
        {240, 320, 120, 160, 4, 1},
        {113,  97,  53, 130, 4, 2},
        {113,  97,  53, 130, 5, 2},
        { 97, 113, 130,  53, 4, 2},
        { 97, 113, 130,  53, 5, 2},
        { 53,  71, 113,  97, 5, 1},
        {113,  97,  53,  71, 0, 2},
        { 53,  71, 113,  97, 1, 2},
        {113,  97,  53,  71, 2, 2},
        {113,  97,  53,  71, 3, 2},
        {113,  97,  53,  71, 4, 2},
        {113,  97,  53,  71, 5, 2},
        { 53,  71, 113,  97, 2, 1},
        {480, 360, 240, 180, 3, 1},
        {113,  97,  53,  71, 0, 1},
        { 64,  48,  64,  48, 4, 1}
    };
    for (auto fixture : cases)
    {
        std::vector<std::array<int, 4>> sizes;
        for (int b = 0; b < fixture.n; ++b)
            sizes.push_back({fixture.sh - b, fixture.sw - b, fixture.dh - b, fixture.dw - b});
        for (int channels : {1, 2, 3, 4})
        {
            auto family = kFamilies[fixture.family];
            RunTensor<T>("NHWC", channels, family, fixture.sh, fixture.sw, fixture.dh, fixture.dw, Pattern::Random,
                         fixture.n, padded);
            if (channels != 2)
                RunVarShape<T>(channels, family, sizes);
        }
    }
    for (auto family : kFamilies) RunTensor<T>("NCHW", 5, family, 23, 31, 29, 17, Pattern::Random, 0, padded);
}

TYPED_TEST(OpTvResize, frozen_valid_geometry_parameterizations)
{
    RunFrozenValidGeometryParameterizations<TypeParam>(true);
}

TEST(OpTvResize, kf_case09_u8_nhwc_c2_cubic_no_antialias)
{
    RunTensor<uint8_t>("NHWC", 2, {NVCV_INTERP_CUBIC, false}, 7, 9, 13, 17, Pattern::Random, 1, true);
}

TEST(OpTvResize, cubic_u8_interleaved_matches_planar_bitexactly)
{
    Stream        stream;
    GuardedTensor src("NHWC", nvcv::TYPE_U8, 1, 31, 47, 3);
    GuardedTensor planarSrc("NCHW", nvcv::TYPE_U8, 1, 31, 47, 3);
    GuardedTensor fast("NHWC", nvcv::TYPE_U8, 1, 23, 19, 3);
    GuardedTensor fallback("NCHW", nvcv::TYPE_U8, 1, 23, 19, 3);
    GuardedTensor fallbackInterleaved("NHWC", nvcv::TYPE_U8, 1, 23, 19, 3);
    src.Set(0, Input<uint8_t>(31, 47, 3, 1234, Pattern::Step));
    src.Upload();
    cvcuda::Reformat reformat;
    reformat(stream.value, src.tensor, planarSrc.tensor);
    cvcuda::TvResize op;
    auto             req               = op.getWorkspaceRequirements(1, 31, 47, 23, 19, 3, NVCV_INTERP_CUBIC, false);
    auto             fastWorkspace     = cvcuda::AllocateWorkspace(req);
    auto             fallbackWorkspace = cvcuda::AllocateWorkspace(req);
    op(stream.value, fastWorkspace.get(), src.tensor, fast.tensor, NVCV_INTERP_CUBIC, false);
    op(stream.value, fallbackWorkspace.get(), planarSrc.tensor, fallback.tensor, NVCV_INTERP_CUBIC, false);
    reformat(stream.value, fallback.tensor, fallbackInterleaved.tensor);
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream.value));
    fast.Download();
    fallbackInterleaved.Download();
    EXPECT_EQ(fast.Get<uint8_t>(0), fallbackInterleaved.Get<uint8_t>(0));
}

TEST(OpTvResize, cubic_u8_interleaved_general_ratio_frozen)
{
    for (int channels : {1, 2, 3, 4})
    {
        for (auto [sh, sw, dh, dw] : {
                 std::array{ 720, 1280, 1080, 1920},
                 std::array{ 720, 1280, 2160, 3840},
                 std::array{1080, 1920,  720, 1280},
                 std::array{2160, 3840, 1080, 1920},
                 std::array{ 131,  237,   79,  113},
                 std::array{  53,   71,   67,   41},
                 std::array{  53,   71,   37,   97},
                 std::array{  37,   53,   37,   53}
        })
            for (bool padded : {true, false})
                RunTensor<uint8_t>("NHWC", channels, {NVCV_INTERP_CUBIC, false}, sh, sw, dh, dw, Pattern::Random, 1,
                                   padded);
        if (channels != 2)
            RunVarShape<uint8_t>(channels,
                                 {
                                     NVCV_INTERP_CUBIC, false
            },
                                 {{37, 53, 61, 71}, {83, 97, 57, 67}});
    }
    // One misaligned image must force the whole 4-channel batch off the uchar4 path.
    for (int misaligned : {0, 1})
        RunVarShape<uint8_t>(4,
                             {
                                 NVCV_INTERP_CUBIC, false
        },
                             {{37, 53, 61, 71}, {83, 97, 57, 67}, {41, 43, 29, 31}}, Pattern::Random, misaligned);
}

void CheckInterleavedU8LargeUnaligned(NVCVInterpolationType mode, int channels, int sh, int sw, int dh, int dw)
{
    SCOPED_TRACE(::testing::Message() << "mode=" << mode << " C=" << channels << " " << sh << "x" << sw << "->" << dh
                                      << "x" << dw);
    constexpr int                     n = 2;
    Stream                            stream;
    GuardedTensor                     input("NHWC", nvcv::TYPE_U8, n, sh, sw, channels, true, 1, 1);
    GuardedTensor                     output("NHWC", nvcv::TYPE_U8, n, dh, dw, channels, true, 3, 1);
    std::vector<std::vector<uint8_t>> inputs(n);
    for (int b = 0; b < n; ++b)
    {
        inputs[b] = Input<uint8_t>(sh, sw, channels, 4321 + b, Pattern::Random);
        input.Set(b, inputs[b]);
        output.Set(b, std::vector<uint8_t>(size_t(dh) * dw * channels));
    }
    input.Upload();
    output.Upload();
    cvcuda::TvResize op;
    auto workspace = cvcuda::AllocateWorkspace(op.getWorkspaceRequirements(n, sh, sw, dh, dw, channels, mode, false));
    op(stream.value, workspace.get(), input.tensor, output.tensor, mode, false);
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream.value));
    output.Download();
    for (int b = 0; b < n; ++b)
    {
        auto gold   = tvgold::Resize(inputs[b], sh, sw, dh, dw, channels, mode, false);
        auto actual = output.Get<uint8_t>(b);
        tvfixture::Export("NHWC", channels, b, n, true, sh, sw, dh, dw, mode, false, int(Pattern::Random), inputs[b],
                          gold, actual);
        EXPECT_TRUE(tvgold::WithinTolerance(gold, actual, mode, false))
            << "sample=" << b << " " << DifferenceContext(gold, actual);
    }
    output.CheckPadding();
}

TEST(OpTvResize, interleaved_u8_large_unaligned_and_varshape)
{
    // Byte offset and odd row padding force the scalar fallback of the 4-channel vector path.
    for (auto mode : {NVCV_INTERP_LINEAR, NVCV_INTERP_CUBIC})
        for (int channels : {3, 4})
            for (auto [sh, sw, dh, dw] : {
                     std::array{1081, 1921,  719, 1281},
                     std::array{ 720, 1280, 1080, 1920}
            })
                ASSERT_NO_FATAL_FAILURE(CheckInterleavedU8LargeUnaligned(mode, channels, sh, sw, dh, dw));
    for (int channels : {1, 3, 4})
        RunVarShape<uint8_t>(channels,
                             {
                                 NVCV_INTERP_CUBIC, false
        },
                             {{1081, 1921, 719, 1281}, {720, 1280, 1080, 1920}, {2160, 3840, 1080, 1920}});
}

TEST(OpTvResize, cubic_unchanged_large_width_preserves_source_column)
{
    constexpr int        width = 8388612;
    Stream               stream;
    GuardedTensor        src("NHWC", nvcv::TYPE_U8, 1, 1, width, 1, false);
    GuardedTensor        dst("NHWC", nvcv::TYPE_U8, 1, 2, width, 1, false);
    std::vector<uint8_t> input(width);
    input[8388609] = 255;
    src.Set(0, input);
    src.Upload();
    cvcuda::TvResize op;
    auto             workspace
        = cvcuda::AllocateWorkspace(op.getWorkspaceRequirements(1, 1, width, 2, width, 1, NVCV_INTERP_CUBIC, false));
    op(stream.value, workspace.get(), src.tensor, dst.tensor, NVCV_INTERP_CUBIC, false);
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream.value));
    dst.Download();
    auto result = dst.Get<uint8_t>(0);
    EXPECT_EQ(result[8388609], 255);
    EXPECT_EQ(result[width + 8388609], 255);
}

TYPED_TEST(OpTvResize, paired_frozen_supported_storage)
{
    // The frozen implementation requires packed planar samples; candidate padding coverage stays separate.
    for (auto family : kFamilies)
        for (auto layout : kLayouts) RunLayoutChannels<TypeParam>(layout, family, false);

    const std::vector<std::array<int, 4>> sizes{
        {48, 64, 48, 64},
        { 1,  1,  3,  5},
        { 1,  7,  4, 11},
        { 7,  1, 11,  4},
        { 1,  7,  4,  3},
        { 7,  1,  3,  4},
        { 9,  7,  4,  3},
        { 7,  9,  7, 13},
        { 7,  9, 11,  9},
        { 7,  9,  7,  5},
        { 7,  9,  3,  9},
        { 8, 12,  4,  6},
        { 4,  6,  8, 12},
        {13, 17,  7, 23},
        {17, 13, 23,  7}
    };
    for (auto family : kFamilies)
    {
        for (auto size : sizes)
            for (auto layout : kLayouts)
                RunTensor<TypeParam>(layout, 1, family, size[0], size[1], size[2], size[3], Pattern::Random, 0, false);
        RunVarShape<TypeParam>(1, family, sizes);
    }
    RunFrozenValidGeometryParameterizations<TypeParam>(false);
}

template<class T, class Bits>
void CheckNearestBits(const std::vector<Bits> &patterns, NVCVInterpolationType mode, bool aa, const char *layout)
{
    Stream         stream;
    GuardedTensor  src(layout, Dtype<T>(), 1, 3, 7, 1);
    GuardedTensor  dst(layout, Dtype<T>(), 1, 5, 11, 1);
    std::vector<T> input(21);
    for (size_t i = 0; i < input.size(); ++i)
    {
        Bits bits = patterns[i % patterns.size()];
        std::memcpy(static_cast<void *>(&input[i]), &bits, sizeof(T));
    }
    src.Set(0, input);
    src.Upload();
    cvcuda::TvResize op;
    auto             ws = cvcuda::AllocateWorkspace(op.getWorkspaceRequirements(1, 3, 7, 5, 11, 1, mode, aa));
    op(stream.value, ws.get(), src.tensor, dst.tensor, mode, aa);
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream.value));
    dst.Download();
    EXPECT_TRUE(tvgold::WithinTolerance(tvgold::Resize(input, 3, 7, 5, 11, 1, mode, aa), dst.Get<T>(0), mode, aa));
    ASSERT_EQ(cudaSuccess, cudaMemsetAsync(dst.allocation, GuardedTensor::kSentinel, dst.host.size(), stream.value));
    nvcv::ImageBatchVarShape inputBatch(1);
    nvcv::ImageBatchVarShape outputBatch(1);
    // A tensor's channel-major and channel-last storage coincide for C=1.
    inputBatch.pushBack(src.AsImage(Format(Dtype<T>(), 1)));
    outputBatch.pushBack(dst.AsImage(Format(Dtype<T>(), 1)));
    int32_t sh      = 3;
    int32_t sw      = 7;
    int32_t dh      = 5;
    int32_t dw      = 11;
    auto    batchWs = cvcuda::AllocateWorkspace(op.getWorkspaceRequirements(1, &sh, &sw, &dh, &dw, 1, mode, aa));
    op(stream.value, batchWs.get(), inputBatch, outputBatch, mode, aa);
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream.value));
    dst.Download();
    EXPECT_TRUE(tvgold::WithinTolerance(tvgold::Resize(input, 3, 7, 5, 11, 1, mode, aa), dst.Get<T>(0), mode, aa));
}

template<class T, class Bits>
void NearestBits(const std::vector<Bits> &patterns)
{
    for (auto mode : {NVCV_INTERP_NEAREST, NVCV_INTERP_NEAREST_FLOOR})
        for (bool aa : {false, true})
            for (auto layout : kLayouts)
                ASSERT_NO_FATAL_FAILURE((CheckNearestBits<T, Bits>(patterns, mode, aa, layout)));
}

TEST(OpTvResize, nearest_preserves_signed_zero_and_nan_payload_bits_aa_ignored)
{
    NearestBits<float, uint32_t>({0x00000000, 0x80000000, 0x7fc12345, 0xffc54321, 0x3f800000, 0x00000001});
    NearestBits<__half, uint16_t>({0x0000, 0x8000, 0x7e35, 0xfe71, 0x3c00, 0x0001});
}

TEST(OpTvResize_Negative, matched_wide_metadata_cannot_wrap_into_valid_launch)
{
    cvcuda::TvResize op;
    Stream           stream;
    GuardedTensor    source("NHWC", nvcv::TYPE_U8, 1, 1, 1, 1);
    GuardedTensor    destination("NHWC", nvcv::TYPE_U8, 1, 1, 1, 1);
    for (auto shape : {
             std::array<int64_t, 4>{INT64_C(4294967297),                   1,                   1,     1},
             {              65536,                   1,                   1,     1},
             {              16384,                   1,                   1,     4},
             {                  1,                   1,                   1, 65536},
             {                  1, INT64_C(4294967297),                   1,     1},
             {                  1,                   1, INT64_C(4294967297),     1}
    })
    {
        auto src = MetadataTensor("NCHW", shape[0], shape[1], shape[2], shape[3], source.allocation + 256);
        auto dst = MetadataTensor("NCHW", shape[0], shape[1], shape[2], shape[3], destination.allocation + 256);
        for (auto family : kFamilies)
            EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
                      cvcudaTvResizeSubmit(op.handle(), stream.value, nullptr, src.handle(), dst.handle(), family.mode,
                                           family.aa));
        ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream.value));
        destination.Download();
        EXPECT_TRUE(
            std::all_of(destination.host.begin(), destination.host.end(), [](uint8_t value) { return value == 0xa5; }));
    }
}

TEST(OpTvResize, independent_workspaces_on_two_caller_streams)
{
    Stream           first;
    Stream           second;
    cvcuda::TvResize op;
    GuardedTensor    src("NHWC", nvcv::TYPE_F32, 1, 13, 17, 3);
    GuardedTensor    dst1("NHWC", nvcv::TYPE_F32, 1, 7, 11, 3);
    GuardedTensor    dst2("NHWC", nvcv::TYPE_F32, 1, 7, 11, 3);
    auto             input = Input<float>(13, 17, 3, 73);
    src.Set(0, input);
    src.Upload();
    auto        requirements = op.getWorkspaceRequirements(1, 13, 17, 7, 11, 3, NVCV_INTERP_CUBIC, true);
    auto        ws1          = cvcuda::AllocateWorkspace(requirements);
    auto        ws2          = cvcuda::AllocateWorkspace(requirements);
    cudaEvent_t done1{};
    cudaEvent_t done2{};
    ASSERT_EQ(cudaSuccess, cudaEventCreate(&done1));
    ASSERT_EQ(cudaSuccess, cudaEventCreate(&done2));
    op(first.value, ws1.get(), src.tensor, dst1.tensor, NVCV_INTERP_CUBIC, true);
    ASSERT_EQ(cudaSuccess, cudaEventRecord(done1, first.value));
    op(second.value, ws2.get(), src.tensor, dst2.tensor, NVCV_INTERP_CUBIC, true);
    ASSERT_EQ(cudaSuccess, cudaEventRecord(done2, second.value));
    ASSERT_EQ(cudaSuccess, cudaEventSynchronize(done1));
    ASSERT_EQ(cudaSuccess, cudaEventSynchronize(done2));
    dst1.Download();
    dst2.Download();
    auto gold = tvgold::Resize(input, 13, 17, 7, 11, 3, NVCV_INTERP_CUBIC, true);
    EXPECT_TRUE(tvgold::WithinTolerance(gold, dst1.Get<float>(0), NVCV_INTERP_CUBIC, true));
    EXPECT_TRUE(tvgold::WithinTolerance(gold, dst2.Get<float>(0), NVCV_INTERP_CUBIC, true));
    EXPECT_TRUE(tvgold::WithinTolerance(dst1.Get<float>(0), dst2.Get<float>(0), NVCV_INTERP_CUBIC, true, true));
    ASSERT_EQ(cudaSuccess, cudaEventDestroy(done1));
    ASSERT_EQ(cudaSuccess, cudaEventDestroy(done2));
}

TEST(OpTvResize_Negative, singleton_negative_strides_rejected_before_identity)
{
    Stream           stream;
    cvcuda::TvResize op;
    GuardedTensor    srcStorage("NHWC", nvcv::TYPE_U8, 1, 1, 1, 1);
    GuardedTensor    dstStorage("NHWC", nvcv::TYPE_U8, 1, 1, 1, 1);
    for (bool badInput : {false, true})
        for (int axis = 0; axis < 4; ++axis)
        {
            NVCVTensorBufferStrided buffer{};
            buffer.basePtr
                = reinterpret_cast<NVCVByte *>((badInput ? srcStorage.allocation : dstStorage.allocation) + 256);
            for (int i = 0; i < 4; ++i) buffer.strides[i] = 1;
            buffer.strides[axis] = -1;
            // nvcv ignores singleton strides on wrapping; TvResize must still validate them.
            auto invalid = nvcv::TensorWrapData(nvcv::TensorDataStridedCuda(
                nvcv::TensorShape{
                    {1, 1, 1, 1},
                    "NHWC"
            },
                nvcv::TYPE_U8, buffer));
            EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
                      cvcudaTvResizeSubmit(
                          op.handle(), stream.value, nullptr, badInput ? invalid.handle() : srcStorage.tensor.handle(),
                          badInput ? dstStorage.tensor.handle() : invalid.handle(), NVCV_INTERP_NEAREST, false));
            ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream.value));
            dstStorage.Download();
            EXPECT_TRUE(
                std::all_of(dstStorage.host.begin(), dstStorage.host.end(), [](uint8_t x) { return x == 0xa5; }));
        }
}

TEST(OpTvResize, positive_singleton_pitches_above_int_max_are_valid)
{
    Stream           stream;
    cvcuda::TvResize op;
    GuardedTensor    srcStorage("NHWC", nvcv::TYPE_U8, 1, 1, 1, 1);
    GuardedTensor    dstStorage("NHWC", nvcv::TYPE_U8, 1, 1, 1, 1);
    srcStorage.Set(0, std::vector<uint8_t>{73});
    srcStorage.Upload();
    auto wrap = [](uint8_t *pointer)
    {
        NVCVTensorBufferStrided buffer{};
        buffer.basePtr = reinterpret_cast<NVCVByte *>(pointer);
        for (int i = 0; i < 4; ++i) buffer.strides[i] = INT64_C(4294967296);
        return nvcv::TensorWrapData(nvcv::TensorDataStridedCuda(
            nvcv::TensorShape{
                {1, 1, 1, 1},
                "NHWC"
        },
            nvcv::TYPE_U8, buffer));
    };
    auto src = wrap(srcStorage.allocation + 256);
    auto dst = wrap(dstStorage.allocation + 256);
    for (auto family : kFamilies)
    {
        auto ws = cvcuda::AllocateWorkspace(op.getWorkspaceRequirements(1, 1, 1, 1, 1, 1, family.mode, family.aa));
        op(stream.value, ws.get(), src, dst, family.mode, family.aa);
        ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream.value));
        dstStorage.Download();
        EXPECT_EQ(std::vector<uint8_t>{73}, dstStorage.Get<uint8_t>(0));
    }
}

TEST(OpTvResize_Negative, empty_tensor_is_rejected_by_shared_constructor)
{
    for (auto layout : {"NHWC", "NCHW"})
    {
        const nvcv::TensorShape shape = std::string(layout) == "NHWC" ? nvcv::TensorShape({0, 13, 17, 3}, layout)
                                                                      : nvcv::TensorShape({0, 3, 13, 17}, layout);
        std::array<NVCVByte, 13 * 17 * 3> storage{};
        NVCVTensorBufferStrided           buffer{};
        buffer.basePtr    = storage.data();
        buffer.strides[0] = 13 * 17 * 3;
        buffer.strides[1] = std::string(layout) == "NHWC" ? 17 * 3 : 13 * 17;
        buffer.strides[2] = std::string(layout) == "NHWC" ? 3 : 17;
        buffer.strides[3] = 1;
        // Zero extents are rejected before the wrapper can use the supplied pointer.
        EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
                  nvcv::ProtectCall(
                      [&shape, &buffer] {
                          auto tensor = nvcv::TensorWrapData(nvcv::TensorDataStridedCuda(shape, nvcv::TYPE_U8, buffer));
                      }));
    }
}

template<class TypeParam>
void CheckPlanarChannelsIndependently(Family family, int channels)
{
    Stream        stream;
    GuardedTensor src("NCHW", Dtype<TypeParam>(), 2, 23, 31, channels);
    GuardedTensor dst("NCHW", Dtype<TypeParam>(), 2, 29, 17, channels);
    GuardedTensor singleSrc("NHWC", Dtype<TypeParam>(), 2 * channels, 23, 31, 1);
    GuardedTensor singleDst("NHWC", Dtype<TypeParam>(), 2 * channels, 29, 17, 1);
    for (int b = 0; b < 2; ++b)
    {
        auto input = Input<TypeParam>(23, 31, channels, 941 + b);
        src.Set(b, input);
        for (int c = 0; c < channels; ++c)
        {
            std::vector<TypeParam> plane(23 * 31);
            size_t                 i = 0;
            for (auto &value : plane) value = input[i++ * channels + c];
            singleSrc.Set(b * channels + c, plane);
        }
    }
    src.Upload();
    singleSrc.Upload();
    cvcuda::TvResize op;
    auto             ws1
        = cvcuda::AllocateWorkspace(op.getWorkspaceRequirements(2, 23, 31, 29, 17, channels, family.mode, family.aa));
    auto ws2 = cvcuda::AllocateWorkspace(
        op.getWorkspaceRequirements(2 * channels, 23, 31, 29, 17, 1, family.mode, family.aa));
    op(stream.value, ws1.get(), src.tensor, dst.tensor, family.mode, family.aa);
    op(stream.value, ws2.get(), singleSrc.tensor, singleDst.tensor, family.mode, family.aa);
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream.value));
    dst.Download();
    singleDst.Download();
    for (int b = 0; b < 2; ++b)
    {
        auto actual = dst.Get<TypeParam>(b);
        for (int c = 0; c < channels; ++c)
        {
            std::vector<TypeParam> plane(29 * 17);
            size_t                 i = 0;
            for (auto &value : plane) value = actual[i++ * channels + c];
            EXPECT_TRUE(tvgold::WithinTolerance(singleDst.Get<TypeParam>(b * channels + c), plane, family.mode,
                                                family.aa, true));
        }
    }
}

TYPED_TEST(OpTvResize, planar_many_channels_match_independent_single_channel_images_exactly)
{
    for (auto family : kFamilies)
        for (int channels : {5, 17})
            ASSERT_NO_FATAL_FAILURE(CheckPlanarChannelsIndependently<TypeParam>(family, channels));
}

TEST(OpTvResizeGold, large_float32_coordinates_preserve_distinct_taps)
{
    const std::array<Family, 4> families{
        {{NVCV_INTERP_LINEAR, false},
         {NVCV_INTERP_LINEAR, true},
         {NVCV_INTERP_CUBIC, false},
         {NVCV_INTERP_CUBIC, true}}
    };
    for (auto [mode, aa] : families)
    {
        auto  taps    = tvgold::Weights(33554433, 33554432, 16777216, mode, aa);
        float sum     = 0;
        int   nonzero = 0;
        for (auto tap : taps)
        {
            sum = tvgold::F32(sum + tap.weight);
            if (tap.weight != 0.f)
            {
                ++nonzero;
                EXPECT_EQ(16777216, tap.index);
                EXPECT_EQ(1.f, tap.weight);
            }
        }
        EXPECT_EQ(1, nonzero);
        EXPECT_EQ(1.f, sum);
    }
}

TEST(OpTvResizeGold, cubic_polynomial_preserves_upstream_float32_evaluation)
{
    // Pinned ATen CUDA polynomial PTX: contracted Horner stages, with q*x*x left-associated.
    EXPECT_EQ(0x1.4p-22f, tvgold::Kernel(0x1.fffff6p-1f, true, false));
    EXPECT_EQ(0x1.800004p-23f, tvgold::Kernel(0x1.fffff6p-1f, true, true));
    EXPECT_EQ(-0x1.8f1c72p-4f, tvgold::Kernel(0x1.7abcdep0f, true, false));
    EXPECT_EQ(-0x1.0a1316p-4f, tvgold::Kernel(0x1.7abcdep0f, true, true));
}

TEST(OpTvResizeGold, int_max_nearest_coordinates_clamp_before_narrowing)
{
    for (auto mode : {NVCV_INTERP_NEAREST, NVCV_INTERP_NEAREST_FLOOR})
        EXPECT_EQ(INT_MAX - 1, tvgold::NearestIndex(INT_MAX, INT_MAX - 1, INT_MAX - 2, mode));
}

TEST(OpTvResizeGold, int_max_coordinates_have_finite_normalized_constant_weights)
{
    for (auto mode : {NVCV_INTERP_LINEAR, NVCV_INTERP_CUBIC})
        for (bool aa : {false, true})
        {
            SCOPED_TRACE(::testing::Message() << "mode=" << mode << " AA=" << aa);
            auto taps = tvgold::Weights(INT_MAX, INT_MAX - 1, INT_MAX - 2, mode, aa);
            ASSERT_FALSE(taps.empty());
            float sum      = 0.f;
            float constant = 0.f;
            for (auto tap : taps)
            {
                EXPECT_GE(tap.index, 0);
                EXPECT_LT(tap.index, INT_MAX);
                ASSERT_TRUE(std::isfinite(tap.weight));
                sum      = tvgold::F32(sum + tap.weight);
                constant = tvgold::F32(constant + tvgold::F32(7.f * tap.weight));
            }
            EXPECT_EQ(1.f, sum);
            EXPECT_EQ(7.f, constant);
        }
}

TEST(OpTvResize, scalar_two_channel_tensor_formats)
{
    EXPECT_EQ(nvcv::TYPE_F16, nvcv::FMT_2F16.planeDataType(0).channelType(0));
    EXPECT_EQ(nvcv::TYPE_F32, nvcv::FMT_2F32.planeDataType(0).channelType(0));
    for (auto family : kFamilies)
    {
        RunTensor<__half>("HWC", nvcv::FMT_2F16.numChannels(), family);
        RunTensor<float>("CHW", nvcv::FMT_2F32.numChannels(), family);
    }
}

} // namespace
