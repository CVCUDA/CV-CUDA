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

#include "ElementwiseOpHarness.hpp" // BaseKind, ExpectRejected
#include "PlanarParityUtils.hpp"

#include <common/TensorDataUtils.hpp>
#include <common/ValueTests.hpp>
#include <cuda.h>
#include <cvcuda/OpAdjustSaturation.hpp>
#include <nvcv/Exception.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>

#include <cmath>
#include <limits>
#include <vector>

namespace test = nvcv::test;
namespace ew   = nvcv::test::elementwise;

namespace {

// torchvision luminance coefficients, identical to the kernel (OpAdjustSaturation.cu).
constexpr float kR2Y = 0.2989f;
constexpr float kG2Y = 0.587f;
constexpr float kB2Y = 0.114f;

// Default device contraction can differ from the independent CPU evaluation by one output level.
constexpr double kU8MaxDiff  = 1;
constexpr double kF32MaxDiff = 1e-6;

// Clamp + store mirroring the kernel's StoreSaturation: integers truncate toward zero into the dtype
// range; floats clamp to [0, 1].
template<typename DT>
DT StoreGold(float v)
{
    if constexpr (std::is_floating_point_v<DT>)
    {
        return static_cast<DT>(fminf(fmaxf(v, 0.0f), 1.0f));
    }
    else
    {
        float t = truncf(v);
        t       = fminf(fmaxf(t, 0.0f), static_cast<float>(std::numeric_limits<DT>::max()));
        return static_cast<DT>(t);
    }
}

// Independent CPU gold on an interleaved HWC buffer: blend each 3-channel pixel toward its luminance;
// 1-channel is identity. Mirrors torchvision.transforms.v2.functional.adjust_saturation.
template<typename DT>
std::vector<DT> AdjustSaturationGold(const std::vector<DT> &in, int channels, double saturation)
{
    if (channels == 1)
    {
        return in; // identity
    }

    const auto      ratio    = static_cast<float>(saturation);
    const auto      oneMinus = static_cast<float>(1.0 - saturation);
    std::vector<DT> out(in.size());
    for (size_t p = 0; p + 3 <= in.size(); p += 3)
    {
        const auto r = static_cast<float>(in[p]);
        const auto g = static_cast<float>(in[p + 1]);
        const auto b = static_cast<float>(in[p + 2]);

        float gray = (kR2Y * r + kG2Y * g) + kB2Y * b;
        if constexpr (!std::is_floating_point_v<DT>)
        {
            gray = floorf(gray);
        }

        out[p]     = StoreGold<DT>(ratio * r + oneMinus * gray);
        out[p + 1] = StoreGold<DT>(ratio * g + oneMinus * gray);
        out[p + 2] = StoreGold<DT>(ratio * b + oneMinus * gray);
    }
    return out;
}

// Bind the operator with a fixed saturation for the parity / negative helpers.
auto invokeSat(double saturation)
{
    return [saturation](cudaStream_t s, const auto &in, auto &out)
    {
        cvcuda::AdjustSaturation op;
        op(s, in, out, saturation);
    };
}

} // namespace

// Tensor correctness over the declared dtype x channel matrix + edge factors --------------------
// clang-format off
NVCV_TEST_SUITE_P(OpAdjustSaturation, test::ValueList<int, int, int, nvcv::ImageFormat, double>
{
    //   width, height, batch,            format            saturation
    {       66,     55,     1,   nvcv::FMT_U8,      0.5}, // u8  / 1ch (identity)
    {      123,     67,     3,   nvcv::FMT_RGB8,    0.5}, // u8  / 3ch blend
    {       50,     40,     2,   nvcv::FMT_RGB8,    0.0}, // u8  / 3ch -> grayscale
    {       50,     40,     2,   nvcv::FMT_RGB8,    1.7}, // u8  / 3ch over-saturate (clamp to 255)
    {       17,     19,     1,   nvcv::FMT_F32,     0.5}, // f32 / 1ch (identity)
    {      101,     33,     2,   nvcv::FMT_RGBf32,  0.5}, // f32 / 3ch blend
    {       60,     48,     1,   nvcv::FMT_RGBf32,  1.5}, // f32 / 3ch over-saturate (clamp to 1)
    {       60,     48,     1,   nvcv::FMT_RGBf32,  0.0}, // f32 / 3ch -> grayscale
});

// clang-format on
TEST_P(OpAdjustSaturation, correct_output)
{
    const int               w   = GetParamValue<0>();
    const int               h   = GetParamValue<1>();
    const int               b   = GetParamValue<2>();
    const nvcv::ImageFormat fmt = GetParamValue<3>();
    const double            sat = GetParamValue<4>();

    if (ew::BaseKind(fmt) == 2)
    {
        ew::RunTensorCorrectBuffer<float>(
            w, h, b, fmt,
            [sat](const std::vector<float> &in, int channels) { return AdjustSaturationGold(in, channels, sat); },
            invokeSat(sat), fmt.numChannels() == 3 ? kF32MaxDiff : 0);
    }
    else
    {
        ew::RunTensorCorrectBuffer<uint8_t>(
            w, h, b, fmt,
            [sat](const std::vector<uint8_t> &in, int channels) { return AdjustSaturationGold(in, channels, sat); },
            invokeSat(sat), fmt.numChannels() == 3 ? kU8MaxDiff : 0);
    }
}

TEST(OpAdjustSaturation, varshape_correct_output)
{
    ew::RunVarShapeCorrectBufferTyped<uint8_t>(
        nvcv::FMT_RGB8,
        [](const std::vector<uint8_t> &in, int channels) { return AdjustSaturationGold(in, channels, 0.5); },
        invokeSat(0.5), kU8MaxDiff);
}

TEST(OpAdjustSaturation, varshape_grayscale_output)
{
    ew::RunVarShapeCorrectBufferTyped<uint8_t>(
        nvcv::FMT_RGB8,
        [](const std::vector<uint8_t> &in, int channels) { return AdjustSaturationGold(in, channels, 0.0); },
        invokeSat(0.0), kU8MaxDiff);
}

TEST(OpAdjustSaturation, varshape_rgbf32_float3_over_saturation_matches_gold)
{
    constexpr double saturation = 1.5;
    ew::RunVarShapeCorrectBufferTyped<float>(
        nvcv::FMT_RGBf32,
        [](const std::vector<float> &in, int channels) { return AdjustSaturationGold(in, channels, saturation); },
        invokeSat(saturation), kF32MaxDiff);
}

TEST(OpAdjustSaturation, varshape_float32_single_channel_zero_factor_preserves_input)
{
    constexpr double saturation = 0.0;
    ew::RunVarShapeCorrectBufferTyped<float>(
        nvcv::FMT_F32,
        [](const std::vector<float> &in, int channels) { return AdjustSaturationGold(in, channels, saturation); },
        invokeSat(saturation));
}

TEST(OpAdjustSaturation, tensor_hwc_rgbf32_float3_clamps_matches_gold)
{
    constexpr int    width      = 37;
    constexpr int    height     = 29;
    constexpr double saturation = 1.5;

    nvcv::Tensor src(
        nvcv::TensorShape{
            {height, width, 3},
            "HWC"
    },
        nvcv::TYPE_F32);
    nvcv::Tensor dst(
        nvcv::TensorShape{
            {height, width, 3},
            "HWC"
    },
        nvcv::TYPE_F32);

    std::vector<float> input(static_cast<size_t>(width) * height * 3);
    ew::FillDeterministicValues(input);
    input[0] = 0.0f;
    input[1] = 1.0f;
    input[2] = 0.25f;

    auto srcData = src.exportData<nvcv::TensorDataStridedCuda>();
    auto dstData = dst.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_NE(srcData, nullptr);
    ASSERT_NE(dstData, nullptr);
    nvcv::util::SetImageTensorFromVector<float>(*srcData, input, 0);

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));
    cvcuda::AdjustSaturation op;
    ASSERT_NO_THROW(op(stream, src, dst, saturation));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    std::vector<float> got;
    nvcv::util::GetImageVectorFromTensor<float>(*dstData, 0, got);
    ew::ExpectBuffer(AdjustSaturationGold(input, 3, saturation), got, kF32MaxDiff);

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

// F16 correctness: the kernel widens every half to float, evaluates the luma dot product and the
// per-channel blend in float (identical to the F32 path, no grayscale floor), clamps to [0, 1],
// and rounds once on the half store. kUlps = 4 per the HalfTestUtils.hpp policy: a 3-tap luma dot
// plus a blend whose host/device FMA contraction may differ at a few rounding steps.
constexpr float kSatF16Ulps = 4.f;

static auto satGoldF32(double saturation)
{
    return [saturation](const std::vector<float> &in, int channels)
    {
        return AdjustSaturationGold<float>(in, channels, saturation);
    };
}

// clang-format off
NVCV_TEST_SUITE_P(OpAdjustSaturationF16, test::ValueList<int, int, int, nvcv::ImageFormat, double>
{
    //   width, height, batch,           format             saturation
    {       33,     23,     2,   nvcv::FMT_F16,       0.5}, // f16 / 1ch (identity)
    {       57,     41,     2,   nvcv::FMT_RGBf16,    0.5}, // f16 / 3ch blend
    {       50,     40,     1,   nvcv::FMT_RGBf16,    1.5}, // f16 / 3ch over-saturate (clamp to 1)
    {       50,     40,     1,   nvcv::FMT_RGBf16,    0.0}, // f16 / 3ch -> grayscale
});

// clang-format on
TEST_P(OpAdjustSaturationF16, tensor_matches_fp32_gold)
{
    const double sat = GetParamValue<4>();
    ew::RunTensorCorrectBufferF16(GetParamValue<0>(), GetParamValue<1>(), GetParamValue<2>(), GetParamValue<3>(),
                                  satGoldF32(sat), invokeSat(sat), kSatF16Ulps);
}

TEST(OpAdjustSaturationF16, varshape_matches_fp32_gold)
{
    ew::RunVarShapeCorrectBufferF16(nvcv::FMT_RGBf16, satGoldF32(0.5), invokeSat(0.5), kSatF16Ulps);
}

TEST(OpAdjustSaturation, zero_extent_tensors_are_noops)
{
    ew::ExpectZeroExtentTensorNoop(
        nvcv::TensorShape{
            {0, 3, 5, 1},
            "NHWC"
    },
        nvcv::TYPE_U8, invokeSat(0.5));
    ew::ExpectZeroExtentTensorNoop(
        nvcv::TensorShape{
            {1, 0, 5, 1},
            "NHWC"
    },
        nvcv::TYPE_U8, invokeSat(0.5));
    ew::ExpectZeroExtentTensorNoop(
        nvcv::TensorShape{
            {1, 3, 0, 1},
            "NHWC"
    },
        nvcv::TYPE_U8, invokeSat(0.5));
}

TEST(OpAdjustSaturation, empty_matching_varshape_is_noop)
{
    ew::ExpectEmptyVarShapeNoop(invokeSat(0.5));
}

TEST(OpAdjustSaturation, tall_packed_u8_tensor_avoids_legacy_grid_limit)
{
    constexpr int    kWidth  = 16;
    constexpr int    kHeight = 65536 * 4;
    constexpr size_t kBytes  = size_t{kWidth} * kHeight * 3;
    cudaStream_t     stream{};
    ASSERT_EQ(cudaSuccess, cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    cvcuda::AdjustSaturation op;
    const auto               checkFormat = [&](nvcv::ImageFormat format)
    {
        nvcv::Tensor src(1, {kWidth, kHeight}, format, nvcv::MemAlignment{}.rowAddr(1));
        nvcv::Tensor dst(1, {kWidth, kHeight}, format, nvcv::MemAlignment{}.rowAddr(1));
        auto         srcData = src.exportData<nvcv::TensorDataStridedCuda>();
        auto         dstData = dst.exportData<nvcv::TensorDataStridedCuda>();
        ASSERT_TRUE(srcData && dstData);
        ASSERT_EQ(cudaSuccess, cudaMemsetAsync(srcData->basePtr(), 0, kBytes, stream));
        ASSERT_EQ(cudaSuccess, cudaMemsetAsync(dstData->basePtr(), 255, kBytes, stream));
        EXPECT_NO_THROW(op(stream, src, dst, 0.5));
        ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
        std::vector<uint8_t> output(kBytes);
        ASSERT_EQ(cudaSuccess, cudaMemcpy(output.data(), dstData->basePtr(), kBytes, cudaMemcpyDeviceToHost));
        EXPECT_EQ(std::vector<uint8_t>(kBytes, 0), output);
    };
    ASSERT_NO_FATAL_FAILURE(checkFormat(nvcv::FMT_RGB8p));
    ASSERT_NO_FATAL_FAILURE(checkFormat(nvcv::FMT_RGB8));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpAdjustSaturation, architecture_specific_u8_tensor_launch)
{
    int device = 0;
    ASSERT_EQ(cudaSuccess, cudaGetDevice(&device));
    cudaDeviceProp properties{};
    ASSERT_EQ(cudaSuccess, cudaGetDeviceProperties(&properties, device));
    const int sm = properties.major * 10 + properties.minor;
    if (sm != 86 && sm != 89)
    {
        GTEST_SKIP() << "Requires SM86 or SM89";
    }

    cudaStream_t stream{};
    ASSERT_EQ(cudaSuccess, cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    cvcuda::AdjustSaturation op;
    const auto               checkFormat = [&](nvcv::ImageFormat format)
    {
        nvcv::Tensor src     = nvcv::util::CreateTensor(1, 4096, 1, format);
        nvcv::Tensor dst     = nvcv::util::CreateTensor(1, 4096, 1, format);
        auto         srcData = src.exportData<nvcv::TensorDataStridedCuda>();
        ASSERT_TRUE(srcData);
        nvcv::util::SetTensorTo<uint8_t>(*srcData, 128);
        ASSERT_NO_THROW(op(stream, src, dst, 0.5));
        ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
        ASSERT_EQ(cudaSuccess, cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        EXPECT_NO_THROW(op(stream, src, dst, 0.5));
        cudaGraph_t graph{};
        ASSERT_EQ(cudaSuccess, cudaStreamEndCapture(stream, &graph));
        size_t nodeCount = 0;
        ASSERT_EQ(cudaSuccess, cudaGraphGetNodes(graph, nullptr, &nodeCount));
        EXPECT_EQ(1, nodeCount);
        std::vector<cudaGraphNode_t> nodes(nodeCount);
        ASSERT_EQ(cudaSuccess, cudaGraphGetNodes(graph, nodes.data(), &nodeCount));
        for (cudaGraphNode_t node : nodes)
        {
            CUDA_KERNEL_NODE_PARAMS parameters{};
            ASSERT_EQ(CUDA_SUCCESS, cuGraphKernelNodeGetParams(node, &parameters));
            EXPECT_EQ(sm == 86 ? 256u : 32u, parameters.blockDimX);
            EXPECT_EQ(sm == 86 ? 1u : 4u, parameters.blockDimY);
            EXPECT_EQ(sm == 86 ? 2u : 128u, parameters.gridDimX);
            EXPECT_EQ(1u, parameters.gridDimY);
            EXPECT_EQ(1u, parameters.gridDimZ);
        }
        EXPECT_EQ(cudaSuccess, cudaGraphDestroy(graph));
    };
    ASSERT_NO_FATAL_FAILURE(checkFormat(nvcv::FMT_RGB8p));
    ASSERT_NO_FATAL_FAILURE(checkFormat(nvcv::FMT_RGB8));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

// Planar == interleaved parity ------------------------------------------------------------------
// clang-format off
NVCV_TEST_SUITE_P(OpAdjustSaturationPlanar,
                  test::ValueList<int, int, int, nvcv::ImageFormat, nvcv::ImageFormat>{
    {  8,   1, 2,    nvcv::FMT_RGB8p,    nvcv::FMT_RGB8},
    { 23,  19, 2,    nvcv::FMT_RGB8p,    nvcv::FMT_RGB8},
    {176, 113, 2,    nvcv::FMT_RGB8p,    nvcv::FMT_RGB8},
    {100,  80, 2, nvcv::FMT_RGBf16p, nvcv::FMT_RGBf16},
    {100,  80, 2, nvcv::FMT_RGBf32p, nvcv::FMT_RGBf32},
});

// clang-format on
TEST_P(OpAdjustSaturationPlanar, tensor_matches_interleaved)
{
    test::planar::RunTensorParity(GetParamValue<3>(), GetParamValue<4>(), GetParamValue<0>(), GetParamValue<1>(),
                                  GetParamValue<0>(), GetParamValue<1>(), GetParamValue<2>(),
                                  [](cudaStream_t s, const nvcv::Tensor &src, const nvcv::Tensor &dst,
                                     nvcv::ImageFormat) { EXPECT_NO_THROW(invokeSat(0.5)(s, src, dst)); });
}

TEST_P(OpAdjustSaturationPlanar, varshape_matches_interleaved)
{
    test::planar::RunVarShapeParity(
        GetParamValue<3>(), GetParamValue<4>(), GetParamValue<0>(), GetParamValue<1>(), GetParamValue<0>(),
        GetParamValue<1>(), GetParamValue<2>(),
        [](cudaStream_t s, const nvcv::ImageBatchVarShape &src, const nvcv::ImageBatchVarShape &dst, nvcv::ImageFormat)
        { EXPECT_NO_THROW(invokeSat(0.5)(s, src, dst)); });
}

TEST(OpAdjustSaturationPlanar, single_channel_tensor_matches_interleaved)
{
    test::planar::RunTensorSingleChannelLayoutParity(
        37, 29, 37, 29, 2, nvcv::TYPE_U8,
        [](cudaStream_t stream, const nvcv::Tensor &src, const nvcv::Tensor &dst)
        { EXPECT_NO_THROW(invokeSat(0.5)(stream, src, dst)); });
}

// Negative tests: the complement of the support matrix must be rejected -------------------------
// clang-format off
NVCV_TEST_SUITE_P(OpAdjustSaturation_Negative, test::ValueList<nvcv::ImageFormat, nvcv::ImageFormat>{
    {nvcv::FMT_U16,      nvcv::FMT_U16   }, // unsupported dtype (16-bit unsigned)
    {nvcv::FMT_F64,      nvcv::FMT_F64   }, // unsupported dtype (64-bit float; F16 is now valid)
    {nvcv::FMT_RGBA8,    nvcv::FMT_RGBA8 }, // unsupported channel count (4)
    {nvcv::FMT_RGB8,     nvcv::FMT_RGB8p }, // layout mismatch (interleaved in, planar out)
    {nvcv::FMT_RGB8,     nvcv::FMT_RGBf32}, // input/output data type mismatch
});

// clang-format on
TEST_P(OpAdjustSaturation_Negative, rejects_unsupported)
{
    ew::ExpectRejected(GetParamValue<0>(), GetParamValue<1>(), invokeSat(0.5));
}

TEST(OpAdjustSaturation_Negative, rejects_two_channel)
{
    ew::ExpectRejected(nvcv::FMT_2F32, nvcv::FMT_2F32, invokeSat(0.5), 16, 16);
}

TEST(OpAdjustSaturation_Negative, rejects_non_rgb_varshape_formats)
{
    ew::ExpectVarShapeRejected(nvcv::FMT_YUV8, invokeSat(0.5));
    ew::ExpectVarShapeRejected(nvcv::FMT_YUV8p, invokeSat(0.5));
    ew::ExpectVarShapeRejected(nvcv::FMT_BGR8, invokeSat(0.5));
    ew::ExpectVarShapeRejected(nvcv::FMT_NV12, invokeSat(0.5));
}

TEST(OpAdjustSaturation_Negative, rejects_vector_tensor_dtype)
{
    nvcv::Tensor src(
        nvcv::TensorShape{
            {1, 2, 4, 1},
            "NHWC"
    },
        nvcv::TYPE_3U8);
    nvcv::Tensor dst(
        nvcv::TensorShape{
            {1, 2, 4, 1},
            "NHWC"
    },
        nvcv::TYPE_3U8);
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcv::ProtectCall([&] { cvcuda::AdjustSaturation{}(stream, src, dst, 0.5); }));
    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpAdjustSaturation_Negative, rejects_negative_saturation)
{
    ew::ExpectRejected(nvcv::FMT_RGB8, nvcv::FMT_RGB8, invokeSat(-1.0));
}

TEST(OpAdjustSaturation_Negative, rejects_tensor_batch_and_stride_overflow)
{
    constexpr int batch = 65536;
    nvcv::Tensor  src   = nvcv::util::CreateTensor(batch, 1, 1, nvcv::FMT_RGBf32);
    nvcv::Tensor  dst   = nvcv::util::CreateTensor(batch, 1, 1, nvcv::FMT_RGBf32);

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));
    cvcuda::AdjustSaturation op;
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcv::ProtectCall([&] { op(stream, src, dst, 0.5); }));
    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    constexpr int64_t kSampleStride = int64_t{1} << 31;
    NVCVByte         *srcPtr        = nullptr;
    NVCVByte         *dstPtr        = nullptr;
    ASSERT_EQ(cudaSuccess, cudaMalloc(&srcPtr, 1));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&dstPtr, 1));

    nvcv::TensorDataStridedCuda::Buffer srcBuffer{};
    srcBuffer.basePtr    = srcPtr;
    srcBuffer.strides[0] = kSampleStride;
    srcBuffer.strides[1] = 12;
    srcBuffer.strides[2] = 12;
    srcBuffer.strides[3] = 4;
    auto dstBuffer       = srcBuffer;
    dstBuffer.basePtr    = dstPtr;
    dstBuffer.strides[0] = 12;

    const nvcv::TensorShape shape{
        {1, 1, 1, 3},
        "NHWC"
    };
    src = nvcv::TensorWrapData(nvcv::TensorDataStridedCuda{shape, nvcv::TYPE_F32, srcBuffer});
    dst = nvcv::TensorWrapData(nvcv::TensorDataStridedCuda{shape, nvcv::TYPE_F32, dstBuffer});

    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));
    EXPECT_EQ(NVCV_ERROR_OVERFLOW, nvcv::ProtectCall([&] { op(stream, src, dst, 0.5); }));
    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
    ASSERT_EQ(cudaSuccess, cudaFree(srcPtr));
    ASSERT_EQ(cudaSuccess, cudaFree(dstPtr));
}

TEST(OpAdjustSaturation_Negative, rejects_mismatched_varshape_image_sizes)
{
    std::vector<nvcv::Image> srcImages{
        nvcv::Image{{17, 19}, nvcv::FMT_RGB8},
        nvcv::Image{{31, 37}, nvcv::FMT_RGB8}
    };
    std::vector<nvcv::Image> dstImages{
        nvcv::Image{{17, 19}, nvcv::FMT_RGB8},
        nvcv::Image{{23, 29}, nvcv::FMT_RGB8}
    };

    nvcv::ImageBatchVarShape src(2);
    nvcv::ImageBatchVarShape dst(2);
    src.pushBack(srcImages.begin(), srcImages.end());
    dst.pushBack(dstImages.begin(), dstImages.end());

    cvcuda::AdjustSaturation op;
    try
    {
        op(nullptr, src, dst, 0.5);
        FAIL() << "Expected mismatched image sizes to be rejected";
    }
    catch (const nvcv::Exception &e)
    {
        EXPECT_EQ(nvcv::Status::ERROR_INVALID_ARGUMENT, e.code());
        EXPECT_STREQ("Input and output image 1 sizes must match: input is 31x37, output is 23x29", e.msg());
    }
}

TEST(OpAdjustSaturation_Negative, create_null_handle)
{
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, cvcudaAdjustSaturationCreate(nullptr));
}
