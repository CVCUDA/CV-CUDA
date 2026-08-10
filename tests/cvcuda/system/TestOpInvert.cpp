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

#include "ElementwiseOpHarness.hpp"
#include "PlanarParityUtils.hpp"

#include <common/ValueTests.hpp>
#include <cvcuda/OpInvert.hpp>

#include <cstring>

namespace test = nvcv::test;
namespace ew   = nvcv::test::elementwise;

namespace {

// Independent CPU gold reference: out = bound - in per element, where bound is the dtype maximum
// (255 / 65535) for unsigned integers and 1.0 for float. This mirrors the documented oracle
// (torchvision.transforms.v2.functional.invert / OpenCV cv::bitwise_not) and is intentionally
// computed independently of the kernel so the bit-exact EXPECT_EQ is a real regression check.
//
// goldFor(DT{}) yields the per-element reference for a given element type; one expression covers
// every declared dtype.
const auto goldFor = []<typename DT>(DT)
{
    return [](DT v)
    {
        return static_cast<DT>(ew::Bound<DT>() - v);
    };
};

// Run the operator. A single generic invoker serves Tensor, ImageBatchVarShape, and the parity
// harness (Invert takes no extra parameters).
const auto invoke = [](cudaStream_t s, const auto &in, auto &out)
{
    cvcuda::Invert op;
    op(s, in, out);
};

} // namespace

// Tensor correctness: bit-exact vs the CPU gold, over the declared dtype × channel matrix --------
// clang-format off
NVCV_TEST_SUITE_P(OpInvert, test::ValueList<int, int, int, nvcv::ImageFormat>
{
    //   width, height, batch,            format          (dtype / channels)
    {       66,     55,     1,   nvcv::FMT_U8     }, // u8  / 1ch
    {      123,     67,     3,   nvcv::FMT_RGB8   }, // u8  / 3ch
    {       42,     53,     4,   nvcv::FMT_RGBA8  }, // u8  / 4ch
    {       80,     40,     2,   nvcv::FMT_U16    }, // u16 / 1ch
    {       17,     19,     1,   nvcv::FMT_F32    }, // f32 / 1ch
    {      101,     33,     2,   nvcv::FMT_RGBf32 }, // f32 / 3ch
    {       64,     48,     3,   nvcv::FMT_RGBAf32}, // f32 / 4ch
});

// clang-format on
TEST_P(OpInvert, tensor_correct_output)
{
    // Invert has no dtype-dependent parameter, so the invoke factory returns the same invoker for
    // every element type.
    ew::RunTensorCorrectDispatch(GetParamValue<0>(), GetParamValue<1>(), GetParamValue<2>(),
                                 nvcv::ImageFormat{GetParamValue<3>()}, goldFor, [](auto) { return invoke; });
}

// VarShape correctness: bit-exact vs the CPU gold ------------------------------------------------
TEST(OpInvert, varshape_correct_output)
{
    ew::RunVarShapeCorrect([](uint8_t v) { return static_cast<uint8_t>(ew::Bound<uint8_t>() - v); }, invoke);
}

NVCV_TEST_SUITE_P(OpInvertVarShapeTyped, test::ValueList<nvcv::ImageFormat>{nvcv::FMT_U16, nvcv::FMT_F32});

TEST_P(OpInvertVarShapeTyped, correct_output)
{
    nvcv::ImageFormat fmt = GetParam();
    if (ew::BaseKind(fmt) == 2)
    {
        ew::RunVarShapeCorrectTyped<float>(fmt, goldFor(float{}), invoke);
    }
    else
    {
        ew::RunVarShapeCorrectTyped<uint16_t>(fmt, goldFor(uint16_t{}), invoke);
    }
}

// Planar ≡ interleaved parity (fake-planar): native planar output must be byte-for-byte identical
// to the interleaved result on the same data. Uses the shared PlanarParityUtils scaffolding. -----
// clang-format off
NVCV_TEST_SUITE_P(OpInvertPlanar,
                  test::ValueList<int, int, int, nvcv::ImageFormat, nvcv::ImageFormat>{
    {177, 113, 2,    nvcv::FMT_RGB8p,    nvcv::FMT_RGB8},
    { 65,  48, 1,   nvcv::FMT_RGBA8p,   nvcv::FMT_RGBA8},
    {101,  80, 2, nvcv::FMT_RGBAf32p, nvcv::FMT_RGBAf32},
});

NVCV_TEST_SUITE_P(OpInvertPlanarVarShape,
                  test::ValueList<int, int, int, nvcv::ImageFormat, nvcv::ImageFormat>{
    {177, 113, 2,    nvcv::FMT_RGB8p,    nvcv::FMT_RGB8},
    {101,  80, 2, nvcv::FMT_RGBAf32p, nvcv::FMT_RGBAf32},
});

// clang-format on
TEST_P(OpInvertPlanar, tensor_matches_interleaved)
{
    test::planar::RunTensorParity(GetParamValue<3>(), GetParamValue<4>(), GetParamValue<0>(), GetParamValue<1>(),
                                  GetParamValue<0>(), GetParamValue<1>(), GetParamValue<2>(),
                                  [](cudaStream_t s, const nvcv::Tensor &src, const nvcv::Tensor &dst,
                                     nvcv::ImageFormat) { EXPECT_NO_THROW(invoke(s, src, dst)); });
}

TEST_P(OpInvertPlanarVarShape, varshape_matches_interleaved)
{
    test::planar::RunVarShapeParity(
        GetParamValue<3>(), GetParamValue<4>(), GetParamValue<0>(), GetParamValue<1>(), GetParamValue<0>(),
        GetParamValue<1>(), GetParamValue<2>(),
        [](cudaStream_t s, const nvcv::ImageBatchVarShape &src, const nvcv::ImageBatchVarShape &dst, nvcv::ImageFormat)
        { EXPECT_NO_THROW(invoke(s, src, dst)); });
}

// Padded-stride tensors: the dense flat fast path must reject padded rows and fall back to the
// layout kernels, which must stay bit-exact vs the CPU gold and must not touch the padding. ------
namespace {

// Check every payload lane equals bound - src (the inverse) and every padding byte is still the
// 0xA5 sentinel, i.e. the fallback kernel wrote the payload correctly and never touched the padding.
template<typename BT>
void VerifyPaddedTensor(const std::vector<uint8_t> &srcBytes, const std::vector<uint8_t> &dstBytes, int batch,
                        int rowsPerSample, int64_t sampleStride, int64_t rowStride, int64_t rowBytes)
{
    for (int n = 0; n < batch; ++n)
    {
        for (int r = 0; r < rowsPerSample; ++r)
        {
            const size_t rowOff = n * sampleStride + r * rowStride;
            for (size_t e = 0; e < rowBytes / sizeof(BT); ++e)
            {
                // memcpy the lanes out of the byte buffers: rowOff need not satisfy alignof(BT)
                // and the payload is raw bytes, so a typed reinterpret_cast read would be UB.
                BT in{};
                BT out{};
                std::memcpy(&in, srcBytes.data() + rowOff + e * sizeof(BT), sizeof(BT));
                std::memcpy(&out, dstBytes.data() + rowOff + e * sizeof(BT), sizeof(BT));
                ASSERT_EQ(static_cast<BT>(ew::Bound<BT>() - in), out)
                    << "payload mismatch at sample " << n << " row " << r << " elem " << e;
            }
            for (int64_t b = rowBytes; b < rowStride; ++b)
            {
                ASSERT_EQ(0xA5, dstBytes[rowOff + b]) << "padding overwritten at sample " << n << " row " << r;
            }
        }
    }
}

template<typename BT>
void RunPaddedTensor(nvcv::ImageFormat fmt, int width, int height, int batch, int padBytes)
{
    const bool    planar    = fmt.numPlanes() > 1;
    const int     channels  = fmt.numChannels();
    const int64_t elemBytes = sizeof(BT);
    // Packed row (per plane for planar formats) plus deliberate padding.
    const int64_t rowBytes     = width * elemBytes * (planar ? 1 : channels);
    const int64_t rowStride    = rowBytes + padBytes;
    const int64_t planeStride  = rowStride * height;
    const int64_t sampleStride = planeStride * (planar ? channels : 1);
    const size_t  totalBytes   = static_cast<size_t>(sampleStride) * batch;

    NVCVByte *srcPtr = nullptr;
    NVCVByte *dstPtr = nullptr;
    ASSERT_EQ(cudaSuccess, cudaMalloc(&srcPtr, totalBytes));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&dstPtr, totalBytes));

    auto wrapTensor = [&](NVCVByte *ptr)
    {
        nvcv::TensorDataStridedCuda::Buffer buffer{};
        buffer.basePtr = ptr;
        if (planar)
        {
            buffer.strides[0] = sampleStride;
            buffer.strides[1] = planeStride;
            buffer.strides[2] = rowStride;
            buffer.strides[3] = elemBytes;
            return nvcv::TensorWrapData(nvcv::TensorDataStridedCuda{
                nvcv::TensorShape{{batch, channels, height, width}, "NCHW"},
                nvcv::DataType{            fmt.planeDataType(0)       },
                buffer
            });
        }
        buffer.strides[0] = sampleStride;
        buffer.strides[1] = rowStride;
        buffer.strides[2] = channels * elemBytes;
        buffer.strides[3] = elemBytes;
        return nvcv::TensorWrapData(nvcv::TensorDataStridedCuda{
            nvcv::TensorShape{   {batch, height, width, channels}, "NHWC"},
            nvcv::DataType{fmt.planeDataType(0).channelType(0)       },
            buffer
        });
    };

    nvcv::Tensor src = wrapTensor(srcPtr);
    nvcv::Tensor dst = wrapTensor(dstPtr);

    // Fill the whole src allocation (payload + padding) with a deterministic byte pattern and the
    // dst allocation with a sentinel, so untouched padding can be verified after the run.
    std::vector<uint8_t> srcBytes(totalBytes);
    for (size_t i = 0; i < totalBytes; ++i)
    {
        srcBytes[i] = static_cast<uint8_t>((i * 31 + 7) % 251);
    }
    if constexpr (std::is_floating_point_v<BT>)
    {
        // Overwrite payload lanes with valid floats (deterministic in [0, 1]). memcpy each value in
        // rather than a typed store so the write stays well-defined on the uint8_t backing storage.
        for (size_t i = 0; i < totalBytes / sizeof(float); ++i)
        {
            const float v = static_cast<float>((i * 37 + 11) % 1000) / 1000.0f;
            std::memcpy(srcBytes.data() + i * sizeof(float), &v, sizeof(float));
        }
    }
    const std::vector<uint8_t> dstSentinel(totalBytes, 0xA5);
    ASSERT_EQ(cudaSuccess, cudaMemcpy(srcPtr, srcBytes.data(), totalBytes, cudaMemcpyHostToDevice));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(dstPtr, dstSentinel.data(), totalBytes, cudaMemcpyHostToDevice));

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));
    EXPECT_NO_THROW(invoke(stream, src, dst));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    std::vector<uint8_t> dstBytes(totalBytes);
    ASSERT_EQ(cudaSuccess, cudaMemcpy(dstBytes.data(), dstPtr, totalBytes, cudaMemcpyDeviceToHost));
    ASSERT_EQ(cudaSuccess, cudaFree(srcPtr));
    ASSERT_EQ(cudaSuccess, cudaFree(dstPtr));

    const int rowsPerSample = height * (planar ? channels : 1);
    VerifyPaddedTensor<BT>(srcBytes, dstBytes, batch, rowsPerSample, sampleStride, rowStride, rowBytes);
}

} // namespace

TEST(OpInvert, tensor_padded_strides_correct_output)
{
    // Interleaved fallbacks (scalar uchar3/uchar4/float3 kernels).
    RunPaddedTensor<uint8_t>(nvcv::FMT_RGB8, 41, 13, 2, 4);
    RunPaddedTensor<uint8_t>(nvcv::FMT_RGBA8, 22, 9, 2, 8);
    RunPaddedTensor<float>(nvcv::FMT_RGBf32, 21, 7, 1, 12);
    // Single-channel interleaved: 8-byte-aligned pad keeps the ushort4 vector path (with padding),
    // odd pad forces the scalar tensor kernel.
    RunPaddedTensor<uint16_t>(nvcv::FMT_U16, 30, 11, 2, 4);
    RunPaddedTensor<uint16_t>(nvcv::FMT_U16, 30, 11, 2, 6);
    // Planar u8 (41-byte packed rows): pad 3 gives 4-byte-aligned strides and keeps the vectorized
    // planar kernel, pad 2 gives odd strides and forces the scalar planar kernel.
    RunPaddedTensor<uint8_t>(nvcv::FMT_RGB8p, 41, 13, 2, 3);
    RunPaddedTensor<uint8_t>(nvcv::FMT_RGB8p, 41, 13, 2, 2);
    // Planar f32 (84-byte packed rows): pad 12 gives 16-byte-aligned strides and keeps the
    // vectorized planar kernel, pad 8 forces its scalar fallback.
    RunPaddedTensor<float>(nvcv::FMT_RGBf32p, 21, 7, 2, 12);
    RunPaddedTensor<float>(nvcv::FMT_RGBf32p, 21, 7, 2, 8);
}

// Negative tests: the complement of the support matrix must be rejected -------------------------
// clang-format off
NVCV_TEST_SUITE_P(OpInvert_Negative, test::ValueList<nvcv::ImageFormat, nvcv::ImageFormat>{
    {nvcv::FMT_F16,   nvcv::FMT_F16  }, // unsupported dtype (16-bit float)
    {nvcv::FMT_S16,   nvcv::FMT_S16  }, // unsupported dtype (signed 16-bit)
    {nvcv::FMT_RGB8,  nvcv::FMT_RGB8p}, // layout mismatch (interleaved in, planar out)
    {nvcv::FMT_RGB8,  nvcv::FMT_RGBf32}, // input/output data type mismatch
});

// clang-format on
TEST_P(OpInvert_Negative, rejects_unsupported)
{
    ew::ExpectRejected(GetParamValue<0>(), GetParamValue<1>(), invoke);
}

// 2-channel input (outside the 1/3/4 matrix) is rejected.
TEST(OpInvert_Negative, rejects_two_channel)
{
    ew::ExpectRejected(nvcv::FMT_2F32, nvcv::FMT_2F32, invoke, 16, 16);
}
