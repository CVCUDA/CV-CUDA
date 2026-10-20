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

#include <common/TensorDataUtils.hpp>
#include <cvcuda/OpRemap.hpp>
#include <cvcuda/OpWarpAffine.hpp>
#include <cvcuda/OpWarpPerspective.hpp>
#include <cvcuda/cuda_tools/InterpolationWrap.hpp>
#include <cvcuda/cuda_tools/MathOps.hpp>
#include <nvcv/TensorDataAccess.hpp>

#include <array>
#include <cmath>

namespace {
namespace cuda = nvcv::cuda;

// Independent Keys cubic reference (a = -0.5), including the negative lobes.
double CubicWeight(double distance)
{
    const double x = std::abs(distance);
    if (x <= 1)
        return (1.5 * x - 2.5) * x * x + 1;
    if (x < 2)
        return ((-.5 * x + 2.5) * x - 4) * x + 2;
    return 0;
}

void ExpectPixel(const std::array<float, 3> &actual, int channels, float expected)
{
    for (int channel = 0; channel < channels; ++channel) EXPECT_NEAR(actual[channel], expected, 1e-4f);
}

double CoverageWeight(NVCVInterpolationType interpolation, double distance)
{
    if (interpolation == NVCV_INTERP_LINEAR)
        return std::max(1 - std::abs(distance), 0.);
    return CubicWeight(distance);
}

float CoverageGold(float x, float y, float border, NVCVInterpolationType interpolation)
{
    double coverage = 0;
    if (interpolation == NVCV_INTERP_NEAREST)
    {
        const auto ix = static_cast<int>(std::floor(x + .5));
        const auto iy = static_cast<int>(std::floor(y + .5));
        coverage      = 0 <= ix && ix < 8 && 0 <= iy && iy < 8;
    }
    else
    {
        for (int iy = 0; iy < 8; ++iy)
            for (int ix = 0; ix < 8; ++ix)
                coverage += CoverageWeight(interpolation, x - static_cast<float>(ix))
                          * CoverageWeight(interpolation, y - static_cast<float>(iy));
    }
    if (coverage == 1)
        return 100.f;
    return static_cast<float>((100 * coverage - border) * coverage + border);
}
} // namespace

using CoverageParam = std::tuple<nvcv::ImageFormat, NVCVInterpolationType>;

class OpConstantCoverage : public testing::TestWithParam<CoverageParam>
{
};

TEST_P(OpConstantCoverage, tensor_operators)
{
    const auto [format, interpolation] = GetParam();
    const int    planes                = format.numPlanes();
    const int    channelsPerPlane      = format.numChannels() / planes;
    nvcv::Tensor input(1, {8, 8}, format);
    nvcv::Tensor output(1, {1, 1}, format);
    nvcv::Tensor map(1, {1, 1}, nvcv::FMT_2F32);
    auto in  = nvcv::TensorDataAccessStridedImagePlanar::Create(*input.exportData<nvcv::TensorDataStridedCuda>());
    auto out = nvcv::TensorDataAccessStridedImagePlanar::Create(*output.exportData<nvcv::TensorDataStridedCuda>());
    auto uv  = nvcv::TensorDataAccessStridedImagePlanar::Create(*map.exportData<nvcv::TensorDataStridedCuda>());
    ASSERT_TRUE(in && out && uv);
    std::array<float, 192> pixels;
    pixels.fill(100.f);
    for (int plane = 0; plane < planes; ++plane)
    {
        const auto rowBytes = 8 * channelsPerPlane * sizeof(float);
        ASSERT_EQ(cudaSuccess, cudaMemcpy2D(in->sampleData(0) + plane * in->planeStride(), in->rowStride(),
                                            pixels.data(), rowBytes, rowBytes, 8, cudaMemcpyHostToDevice));
    }
    cvcuda::WarpAffine      affine(1);
    cvcuda::WarpPerspective perspective(1);
    cvcuda::Remap           remap;
    for (float2 point : {make_float2(-5, 3), make_float2(-.5f, 3), make_float2(-.5f, -.5f), make_float2(.25f, 3),
                         make_float2(7.5f, 7.5f), make_float2(3.25f, 3.5f), make_float2(7, 3), make_float2(0, 3)})
    {
        SCOPED_TRACE(testing::Message() << "coordinate " << point.x << "," << point.y);
        const float              border      = point.x == 7 || point.x == 0 ? 1e20f : 20.5f;
        const float4             borderValue = make_float4(border, border, border, border);
        NVCVAffineTransform      a           = {1, 0, point.x, 0, 1, point.y};
        NVCVPerspectiveTransform p           = {1, 0, point.x, 0, 1, point.y, 0, 0, 1};
        ASSERT_EQ(cudaSuccess, cudaMemcpy(uv->sampleData(0), &point, sizeof(point), cudaMemcpyHostToDevice));
        for (int op = 0; op < 3; ++op)
        {
            SCOPED_TRACE(testing::Message() << "operator " << op);
            // Remap supports multi-channel F32 only via its per-plane path.
            if (op == 2 && channelsPerPlane != 1)
                continue;
            if (op == 0)
                affine(nullptr, input, output, a, interpolation | NVCV_WARP_INVERSE_MAP, NVCV_BORDER_CONSTANT_COVERAGE,
                       borderValue);
            else if (op == 1)
                perspective(nullptr, input, output, p, interpolation | NVCV_WARP_INVERSE_MAP,
                            NVCV_BORDER_CONSTANT_COVERAGE, borderValue);
            else
                remap(nullptr, input, output, map, interpolation, NVCV_INTERP_NEAREST, NVCV_REMAP_ABSOLUTE, false,
                      NVCV_BORDER_CONSTANT_COVERAGE, borderValue);
            ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());
            for (int plane = 0; plane < planes; ++plane)
            {
                std::array<float, 3> actual{};
                ASSERT_EQ(cudaSuccess, cudaMemcpy(actual.data(), out->sampleData(0) + plane * out->planeStride(),
                                                  channelsPerPlane * sizeof(float), cudaMemcpyDeviceToHost));
                ExpectPixel(actual, channelsPerPlane, CoverageGold(point.x, point.y, border, interpolation));
            }
        }
    }
}

INSTANTIATE_TEST_SUITE_P(FormatsAndInterpolation, OpConstantCoverage,
                         testing::Combine(testing::Values(nvcv::FMT_F32, nvcv::FMT_RGBf32, nvcv::FMT_RGBf32p),
                                          testing::Values(NVCV_INTERP_NEAREST, NVCV_INTERP_LINEAR, NVCV_INTERP_CUBIC)));

static nvcv::Tensor MakeImageTensor(int width, int height, int channels, bool planar, nvcv::DataType dtype)
{
    return planar ? nvcv::Tensor(nvcv::TensorShape({1, channels, height, width}, nvcv::TENSOR_NCHW), dtype)
                  : nvcv::Tensor(nvcv::TensorShape({1, height, width, channels}, nvcv::TENSOR_NHWC), dtype);
}

template<typename T>
void CheckWarpType(nvcv::DataType dtype)
{
    for (int channels : {1, 3, 4})
        for (bool planar : {false, true})
        {
            SCOPED_TRACE(testing::Message() << "channels=" << channels << ", planar=" << planar);
            nvcv::Tensor input             = MakeImageTensor(8, 8, channels, planar, dtype);
            nvcv::Tensor affineOutput      = MakeImageTensor(1, 1, channels, planar, dtype);
            nvcv::Tensor perspectiveOutput = MakeImageTensor(1, 1, channels, planar, dtype);
            auto         inputData         = input.exportData<nvcv::TensorDataStridedCuda>();
            auto         affineData        = affineOutput.exportData<nvcv::TensorDataStridedCuda>();
            auto         perspectiveData   = perspectiveOutput.exportData<nvcv::TensorDataStridedCuda>();
            ASSERT_TRUE(inputData && affineData && perspectiveData);

            std::vector<T> pixels(64 * channels, static_cast<T>(100));
            nvcv::util::SetImageTensorFromVector<T>(*inputData, pixels, 0);

            constexpr float          kBorder = 20.5f;
            NVCVAffineTransform      affineTransform{1, 0, -.5f, 0, 1, 3};
            NVCVPerspectiveTransform perspectiveTransform{1, 0, -.5f, 0, 1, 3, 0, 0, 1};
            cvcuda::WarpAffine       affine(1);
            cvcuda::WarpPerspective  perspective(1);
            affine(nullptr, input, affineOutput, affineTransform, NVCV_INTERP_LINEAR | NVCV_WARP_INVERSE_MAP,
                   NVCV_BORDER_CONSTANT_COVERAGE, make_float4(kBorder, kBorder, kBorder, kBorder));
            perspective(nullptr, input, perspectiveOutput, perspectiveTransform,
                        NVCV_INTERP_LINEAR | NVCV_WARP_INVERSE_MAP, NVCV_BORDER_CONSTANT_COVERAGE,
                        make_float4(kBorder, kBorder, kBorder, kBorder));
            ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

            std::vector<T> affineActual;
            std::vector<T> perspectiveActual;
            nvcv::util::GetImageVectorFromTensor<T>(*affineData, 0, affineActual);
            nvcv::util::GetImageVectorFromTensor<T>(*perspectiveData, 0, perspectiveActual);
            const T expected = cuda::SaturateCast<T>(CoverageGold(-.5f, 3, kBorder, NVCV_INTERP_LINEAR));
            EXPECT_EQ(affineActual, std::vector<T>(channels, expected));
            EXPECT_EQ(perspectiveActual, std::vector<T>(channels, expected));
        }
}

TEST(OpConstantCoverage, warp_declared_dtypes_channels_and_layouts)
{
    CheckWarpType<uint8_t>(nvcv::TYPE_U8);
    CheckWarpType<uint16_t>(nvcv::TYPE_U16);
    CheckWarpType<int16_t>(nvcv::TYPE_S16);
    CheckWarpType<__half>(nvcv::TYPE_F16);
    CheckWarpType<float>(nvcv::TYPE_F32);
}

template<typename T>
void CheckRemapType(nvcv::DataType dtype, int channels)
{
    for (bool planar : {false, true})
    {
        SCOPED_TRACE(testing::Message() << "channels=" << channels << ", planar=" << planar);
        nvcv::Tensor input  = MakeImageTensor(8, 8, channels, planar, dtype);
        nvcv::Tensor output = MakeImageTensor(1, 1, channels, planar, dtype);
        nvcv::Tensor map(1, {1, 1}, nvcv::FMT_2F32);
        auto         inputData  = input.exportData<nvcv::TensorDataStridedCuda>();
        auto         outputData = output.exportData<nvcv::TensorDataStridedCuda>();
        auto         mapData    = map.exportData<nvcv::TensorDataStridedCuda>();
        ASSERT_TRUE(inputData && outputData && mapData);
        std::vector<T> pixels(8 * 8 * channels, static_cast<T>(100));
        nvcv::util::SetImageTensorFromVector<T>(*inputData, pixels, 0);
        const float2 point = make_float2(-.5f, 3);
        ASSERT_EQ(cudaSuccess, cudaMemcpy(mapData->basePtr(), &point, sizeof(point), cudaMemcpyHostToDevice));

        constexpr float kBorder = 20.5f;
        cvcuda::Remap   remap;
        remap(nullptr, input, output, map, NVCV_INTERP_LINEAR, NVCV_INTERP_NEAREST, NVCV_REMAP_ABSOLUTE, false,
              NVCV_BORDER_CONSTANT_COVERAGE, make_float4(kBorder, kBorder, kBorder, kBorder));
        ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());
        std::vector<T> actual;
        nvcv::util::GetImageVectorFromTensor<T>(*outputData, 0, actual);
        const T expected = cuda::SaturateCast<T>(CoverageGold(point.x, point.y, kBorder, NVCV_INTERP_LINEAR));
        EXPECT_EQ(actual, std::vector<T>(channels, expected));
    }
}

TEST(OpConstantCoverage, remap_declared_dtypes_channels_and_layouts)
{
    CheckRemapType<uint8_t>(nvcv::TYPE_U8, 1);
    CheckRemapType<uint8_t>(nvcv::TYPE_U8, 3);
    CheckRemapType<uint8_t>(nvcv::TYPE_U8, 4);
    CheckRemapType<__half>(nvcv::TYPE_F16, 1);
    CheckRemapType<float>(nvcv::TYPE_F32, 1);
}

TEST(OpConstantCoverage, remap_varshape_rejects_coverage)
{
    nvcv::ImageBatchVarShape input(1);
    nvcv::ImageBatchVarShape output(1);
    input.pushBack(nvcv::Image({8, 8}, nvcv::FMT_U8));
    output.pushBack(nvcv::Image({8, 8}, nvcv::FMT_U8));
    nvcv::Tensor  map(1, {8, 8}, nvcv::FMT_2F32);
    cvcuda::Remap remap;
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcv::ProtectCall(
                                               [&]
                                               {
                                                   remap(nullptr, input, output, map, NVCV_INTERP_LINEAR,
                                                         NVCV_INTERP_NEAREST, NVCV_REMAP_ABSOLUTE, false,
                                                         NVCV_BORDER_CONSTANT_COVERAGE, float4{});
                                               }));
}

TEST(OpConstantCoverage, direct_linear_constructor_preserves_fraction)
{
    const unsigned char                     pixels[] = {100, 100, 100, 100};
    cuda::Tensor2DWrap<const unsigned char> tensor(pixels, 2);
    using Border = cuda::BorderWrap<decltype(tensor), NVCV_BORDER_CONSTANT_COVERAGE, true, true>;
    cuda::InterpolationWrap<Border, NVCV_INTERP_LINEAR> direct(tensor, 1.5f, 0, 0, 2, 2);
    EXPECT_EQ(direct[make_float2(-.4f, 0)], 37);
}

template<NVCVInterpolationType Interpolation>
void CheckCoverageEndpoints()
{
    std::array<float, 64> pixels;
    pixels.fill(100.f);
    cuda::Tensor2DWrap<const float> tensor(pixels.data(), static_cast<int64_t>(8 * sizeof(float)));
    using Border = cuda::BorderWrap<decltype(tensor), NVCV_BORDER_CONSTANT_COVERAGE, true, true>;
    cuda::InterpolationWrap<Border, Interpolation> sample(tensor, 1e20f, 0, 0, 8, 8);
    for (float2 point : {make_float2(0, 3), make_float2(7, 3), make_float2(3, 0), make_float2(3, 7)})
    {
        EXPECT_EQ(sample[point], 100.f);
    }
    EXPECT_EQ(sample[make_float2(-5, 3)], 1e20f);
}

TEST(OpConstantCoverage, zero_weight_outside_taps_preserve_coverage_endpoints)
{
    CheckCoverageEndpoints<NVCV_INTERP_LINEAR>();
    CheckCoverageEndpoints<NVCV_INTERP_CUBIC>();
}

template<NVCVInterpolationType Interpolation>
void CheckVectorConstructor()
{
    std::array<uchar3, 64> pixels;
    pixels.fill(make_uchar3(100, 110, 120));
    cuda::Tensor2DWrap<const uchar3> tensor(pixels.data(), static_cast<int64_t>(8 * sizeof(uchar3)));
    using Border = cuda::BorderWrap<decltype(tensor), NVCV_BORDER_CONSTANT_COVERAGE, true, true>;
    const auto                                     border = make_float3(1.5f, 2.5f, 3.5f);
    cuda::InterpolationWrap<Border, Interpolation> direct(tensor, border, 0, 0, 8, 8);
    cuda::InterpolationWrap<Border, Interpolation> wrapped(Border(tensor, border, 8, 8));
    nvcv::Tensor                                   image(1, {8, 8}, nvcv::FMT_RGB8);
    auto                                           data = image.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_TRUE(data);
    auto factory
        = cuda::CreateInterpolationWrapNHW<const uchar3, NVCV_BORDER_CONSTANT_COVERAGE, Interpolation>(*data, border);
    decltype(factory) fromData(*data, border);
    EXPECT_EQ(fromData.borderWrap().borderValue().x, factory.borderWrap().borderValue().x);
    EXPECT_EQ(fromData.borderWrap().borderValue().y, factory.borderWrap().borderValue().y);
    EXPECT_EQ(fromData.borderWrap().borderValue().z, factory.borderWrap().borderValue().z);
    for (float2 point : {make_float2(-5, 3), make_float2(-.4f, 3), make_float2(.25f, 3), make_float2(3, 3)})
    {
        const auto actual   = direct[point];
        const auto expected = wrapped[point];
        EXPECT_EQ(actual.x, expected.x);
        EXPECT_EQ(actual.y, expected.y);
        EXPECT_EQ(actual.z, expected.z);
    }
}

TEST(OpConstantCoverage, direct_vector_constructors)
{
    CheckVectorConstructor<NVCV_INTERP_NEAREST>();
    CheckVectorConstructor<NVCV_INTERP_LINEAR>();
    CheckVectorConstructor<NVCV_INTERP_CUBIC>();
}
