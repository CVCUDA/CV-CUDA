/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef CVCUDA_BENCH_CHANNEL_COPY_HPP
#define CVCUDA_BENCH_CHANNEL_COPY_HPP

#include "../CppBenchUtils.hpp"

#include <cvcuda/OpMerge.hpp>
#include <cvcuda/OpReformat.hpp>
#include <cvcuda/OpSplit.hpp>

#include <nvbench/nvbench.cuh>

#include <optional>
#include <type_traits>
#include <vector>

namespace channelbench {

inline nvcv::Tensor MakeTensor(int3 shape, int channels, nvcv::DataType dtype, bool planar)
{
    const auto tensorShape = planar ? nvcv::TensorShape({shape.x, channels, shape.y, shape.z}, "NCHW")
                                    : nvcv::TensorShape({shape.x, shape.y, shape.z, channels}, "NHWC");
    return nvcv::Tensor(tensorShape, dtype);
}

template<typename Scalar>
nvcv::ImageFormat ImageFormat(int channels, bool planar)
{
    if (channels == 1)
        return benchutils::GetFormat<Scalar>();
    if (channels == 3)
    {
        using RGB = nvcv::cuda::MakeType<Scalar, 3>;
        if (planar)
            return benchutils::GetPlanarFormat<RGB>();
        if constexpr (std::is_same_v<Scalar,
                                     uint8_t> || std::is_same_v<Scalar, float> || std::is_same_v<Scalar, __half>)
            return benchutils::GetRGBFormat<RGB>();
        return benchutils::GetFormat<RGB>();
    }
    if (channels == 4)
    {
        if constexpr (std::is_same_v<Scalar, int8_t>)
            if (planar)
                return nvcv::FMT_RGBA8p.dataKind(nvcv::DataKind::SIGNED);
        using RGBA = nvcv::cuda::MakeType<Scalar, 4>;
        return planar ? benchutils::GetPlanarFormat<RGBA>() : benchutils::GetFormat<RGBA>();
    }
    const auto dtype = benchutils::GetDataType<Scalar>();
    if (planar)
        return nvcv::ImageFormat{nvcv::MemLayout::PL, dtype.dataKind(), nvcv::Swizzle::S_XY00, dtype.packing(),
                                 dtype.packing()};
    auto packing = nvcv::Packing::X64_Y64;
    if constexpr (sizeof(Scalar) == 1)
        packing = nvcv::Packing::X8_Y8;
    else if constexpr (sizeof(Scalar) == 2)
        packing = nvcv::Packing::X16_Y16;
    else if constexpr (sizeof(Scalar) == 4)
        packing = nvcv::Packing::X32_Y32;
    return nvcv::ImageFormat{nvcv::MemLayout::PL, dtype.dataKind(), nvcv::Swizzle::S_XY00, packing};
}

template<typename Scalar>
void FillImage(const nvcv::Image &image, bool checker)
{
    auto data = image.exportData<nvcv::ImageDataStridedCuda>();
    CVCUDA_CHECK_DATA(data);
    for (int plane = 0; plane < data->numPlanes(); ++plane)
    {
        const auto &buffer = data->plane(plane);
        if (checker)
            benchutils::launchCheckerboardImageTyped<Scalar>(
                reinterpret_cast<Scalar *>(buffer.basePtr), buffer.rowStride * buffer.height / sizeof(Scalar),
                buffer.rowStride / sizeof(Scalar), image.format().planeNumChannels(plane),
                benchutils::DefaultRangeMax<Scalar>(), static_cast<Scalar>(0), nullptr);
        else
            CUDA_CHECK_ERROR(cudaMemset2D(buffer.basePtr, buffer.rowStride, 0, buffer.rowStride, buffer.height));
    }
}

template<typename Scalar, bool Split>
void RunVarShape(nvbench::state &state, int warmup, int3 shape, int channels, bool planar, int variation)
{
    if (shape.x <= 0 || shape.y <= 0 || shape.z <= 0 || channels < 1 || channels > 4 || variation < 0)
    {
        state.skip("VarShape channel-copy benchmarks require positive shapes, C1..4 and nonnegative sizeVariation");
        return;
    }
    const auto                            imageFormat = ImageFormat<Scalar>(channels, planar);
    const auto                            bandFormat  = ImageFormat<Scalar>(1, false);
    nvcv::ImageBatchVarShape              image(shape.x);
    std::vector<nvcv::ImageBatchVarShape> bands;
    bands.reserve(channels);
    int64_t bytes = 0;
    for (int sample = 0; sample < shape.x; ++sample)
    {
        const int          offset = (sample < shape.x / 2 ? sample : -(sample - shape.x / 2)) * variation;
        const nvcv::Size2D size{std::max(1, shape.z + offset), std::max(1, shape.y + offset)};
        nvcv::Image        sampleImage(size, imageFormat);
        FillImage<Scalar>(sampleImage, Split);
        image.pushBack(sampleImage);
        bytes += static_cast<int64_t>(size.w) * size.h * channels * sizeof(Scalar);
    }
    for (int channel = 0; channel < channels; ++channel)
    {
        bands.emplace_back(shape.x);
        for (int sample = 0; sample < shape.x; ++sample)
        {
            nvcv::Image sampleBand(image[sample].size(), bandFormat);
            FillImage<Scalar>(sampleBand, !Split);
            bands.back().pushBack(sampleBand);
        }
    }
    CUDA_CHECK_ERROR(cudaStreamSynchronize(nullptr));
    state.add_global_memory_reads(bytes);
    state.add_global_memory_writes(bytes);
    std::conditional_t<Split, cvcuda::Split, cvcuda::Merge> op;
    benchutils::warmup_and_exec(state, warmup,
                                [&op, &image, &bands, channels](cudaStream_t stream)
                                {
                                    if constexpr (Split)
                                        op(stream, image, bands.data(), channels);
                                    else
                                        op(stream, bands.data(), channels, image);
                                });
}

template<typename Scalar, bool Split>
void FillTensorBands(int3 shape, int channels, bool planar, bool fake, nvcv::TensorBatch &bands,
                     nvcv::TensorBatch &interBands)
{
    const auto dtype = benchutils::GetDataType<Scalar>();
    for (int channel = 0; channel < channels; ++channel)
    {
        auto band = MakeTensor(shape, 1, dtype, planar);
        if constexpr (!Split)
            benchutils::FillTensor<Scalar>(band, benchutils::CheckerboardValues<Scalar>());
        bands.pushBack(band);
        if (fake)
            interBands.pushBack(MakeTensor(shape, 1, dtype, false));
    }
}

inline void ReformatBands(cudaStream_t stream, cvcuda::Reformat &reformat, const nvcv::TensorBatch &src,
                          const nvcv::TensorBatch &dst, int channels)
{
    for (int channel = 0; channel < channels; ++channel) reformat(stream, src[channel], dst[channel]);
}

template<typename Scalar, bool Split>
void RunTensor(nvbench::state &state, int warmup, int3 shape, int channels, bool planar, bool fake)
{
    const auto dtype = benchutils::GetDataType<Scalar>();
    if (fake && dtype == nvcv::TYPE_U32)
    {
        state.skip("Reformat does not support U32; benchmark native layouts for this dtype");
        return;
    }
    const int64_t bytes = static_cast<int64_t>(shape.x) * shape.y * shape.z * channels * sizeof(Scalar);
    state.add_global_memory_reads(bytes * (fake ? 3 : 1));
    state.add_global_memory_writes(bytes * (fake ? 3 : 1));
    auto              image       = MakeTensor(shape, channels, dtype, planar);
    auto              interleaved = fake ? MakeTensor(shape, channels, dtype, false) : image;
    nvcv::TensorBatch bands(channels);
    auto              interBands = fake ? nvcv::TensorBatch(channels) : bands;
    if constexpr (Split)
        benchutils::FillTensor<Scalar>(image, benchutils::CheckerboardValues<Scalar>());
    FillTensorBands<Scalar, Split>(shape, channels, planar, fake, bands, interBands);
    std::conditional_t<Split, cvcuda::Split, cvcuda::Merge> op;
    std::optional<cvcuda::Reformat>                         reformat;
    if (fake)
        reformat.emplace();
    benchutils::warmup_and_exec(
        state, warmup,
        [&op, &reformat, &image, &interleaved, &bands, &interBands, fake, channels](cudaStream_t stream)
        {
            if constexpr (Split)
            {
                if (fake)
                    (*reformat)(stream, image, interleaved);
                op(stream, interleaved, interBands);
                if (fake)
                    ReformatBands(stream, *reformat, interBands, bands, channels);
            }
            else
            {
                if (fake)
                    ReformatBands(stream, *reformat, bands, interBands, channels);
                op(stream, interBands, interleaved);
                if (fake)
                    (*reformat)(stream, interleaved, image);
            }
        });
}

template<typename Pixel, bool Split>
void Run(nvbench::state &state, int warmup)
{
    using Scalar         = typename nvcv::cuda::BaseType<Pixel>;
    const auto shapeText = state.get_string("shape");
    const auto separator = shapeText.find('@');
    const auto shape     = benchutils::GetShape<3, int3>(shapeText.substr(0, separator));
    const auto layout    = state.get_string("layout");
    const auto inputKind = state.get_string("inputKind");
    const auto channels  = static_cast<int>(state.get_int64("channels"));
    if ((inputKind != "Tensor" && inputKind != "VarShape")
        || (layout != "NHWC" && layout != "NCHW" && layout != "NCHW_FAKE"))
    {
        state.skip("Channel-copy benchmarks require Tensor or VarShape and NHWC, NCHW or NCHW_FAKE");
        return;
    }
    const bool fake   = layout == "NCHW_FAKE";
    const bool planar = layout != "NHWC";
    if (inputKind == "VarShape")
    {
        if (fake)
            state.skip("Fake-planar (NCHW_FAKE) channel-copy benchmarks are Tensor-only");
        else
            RunVarShape<Scalar, Split>(state, warmup, shape, channels, planar,
                                       separator == std::string::npos ? 0 : std::stoi(shapeText.substr(separator + 1)));
        return;
    }
    if (separator != std::string::npos)
    {
        state.skip("Ragged shape suffixes are VarShape-only");
        return;
    }
    RunTensor<Scalar, Split>(state, warmup, shape, channels, planar, fake);
}

} // namespace channelbench

#endif
