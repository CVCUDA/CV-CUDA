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

#ifndef NVCV_TEST_SYSTEM_ELEMENTWISE_OP_HARNESS_HPP
#define NVCV_TEST_SYSTEM_ELEMENTWISE_OP_HARNESS_HPP

// Shared system-test scaffold for unary element-wise operators (Invert, Solarize, Posterize, ...).
// These operators share an identical test shape — CPU gold over a dtype x channel matrix,
// a var-shape correctness case, and complement negative cases — and differ only in (a) the
// per-element gold function and (b) how the operator is invoked (extra scalar params). make-op
// generates each operator's test from one template, so the structural scaffold is centralized here
// and each TestOp<Name>.cpp supplies only the op-specific gold + invoke callables. This keeps the
// real per-operator content (the gold reference) explicit while removing copy-paste of the harness.

#include "Definitions.hpp"

#include <common/TensorDataUtils.hpp>
#include <nvcv/Exception.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorDataAccess.hpp>

#include <array>
#include <cstdint>
#include <limits>
#include <optional>
#include <type_traits>
#include <vector>

namespace nvcv::test::elementwise {

// 0 = u8, 1 = u16, 2 = f32 -- the base type of a test image format.
inline int BaseKind(nvcv::ImageFormat fmt)
{
    if (fmt == nvcv::FMT_F32 || fmt == nvcv::FMT_RGBf32 || fmt == nvcv::FMT_RGBAf32)
    {
        return 2;
    }
    if (fmt == nvcv::FMT_U16)
    {
        return 1;
    }
    return 0; // FMT_U8 / FMT_RGB8 / FMT_RGBA8
}

// The dtype maximum used as the "bound" by the photometric operators (255 / 65535 for unsigned
// integers, 1.0 for float). Exposed so each operator's gold lambda can reuse the same convention.
template<typename DT>
constexpr DT Bound()
{
    return std::is_floating_point_v<DT> ? static_cast<DT>(1) : std::numeric_limits<DT>::max();
}

// Stable, type-appropriate input data for exact-result tests. This deliberately uses an explicit
// arithmetic sequence rather than a pseudo-random API: reproducibility matters here, not entropy.
template<typename DT>
void FillDeterministicValues(std::vector<DT> &values, size_t sequence = 0)
{
    const uint64_t range = static_cast<uint64_t>(Bound<DT>()) + 1;
    size_t         i     = 0;
    for (DT &value : values)
    {
        const auto sample = static_cast<uint32_t>((i + sequence * 131U) * 1664525U + 1013904223U);
        if constexpr (std::is_floating_point_v<DT>)
        {
            value = static_cast<DT>(sample & 0xffffU) / static_cast<DT>(0xffffU);
        }
        else
        {
            value = static_cast<DT>(static_cast<uint64_t>(sample) % range);
        }
        ++i;
    }
}

template<typename DT>
void ExpectBuffer(const std::vector<DT> &gold, const std::vector<DT> &got, double maxDiff)
{
    if (maxDiff == 0)
    {
        EXPECT_EQ(gold, got);
        return;
    }

    ASSERT_EQ(gold.size(), got.size());
    for (size_t i = 0; i < gold.size(); ++i)
    {
        ASSERT_NEAR(static_cast<double>(gold[i]), static_cast<double>(got[i]), maxDiff) << "at flat index " << i;
    }
}

// Common tensor scaffold for gold functions that operate on the whole interleaved image buffer.
// `goldBuffer` receives the input vector and channel count, and returns the expected output vector.
template<typename DT, typename GoldBufferFn, typename InvokeFn>
void RunTensorCorrectBuffer(int width, int height, int batch, nvcv::ImageFormat fmt, GoldBufferFn goldBuffer,
                            InvokeFn invoke, double maxDiff = 0)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    const int    channels = fmt.numChannels();
    const size_t count    = static_cast<size_t>(width) * height * channels;

    nvcv::Tensor inTensor  = nvcv::util::CreateTensor(batch, width, height, fmt);
    nvcv::Tensor outTensor = nvcv::util::CreateTensor(batch, width, height, fmt);
    auto         inData    = inTensor.exportData<nvcv::TensorDataStridedCuda>();
    auto         outData   = outTensor.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_NE(inData, nullptr);
    ASSERT_NE(outData, nullptr);

    for (int s = 0; s < batch; ++s)
    {
        std::vector<DT> in(count);
        FillDeterministicValues(in, static_cast<size_t>(s));
        if (!in.empty())
        {
            in[0] = static_cast<DT>(0);
        }
        if (in.size() > 1)
        {
            in[1] = Bound<DT>();
        }
        nvcv::util::SetImageTensorFromVector<DT>(*inData, in, s);
    }

    ASSERT_NO_THROW(invoke(stream, inTensor, outTensor));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    for (int s = 0; s < batch; ++s)
    {
        SCOPED_TRACE(s);
        std::vector<DT> in;
        std::vector<DT> got;
        nvcv::util::GetImageVectorFromTensor<DT>(*inData, s, in);
        nvcv::util::GetImageVectorFromTensor<DT>(*outData, s, got);
        ExpectBuffer(goldBuffer(in, channels), got, maxDiff);
    }

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

// Typed tensor correctness: fill a tensor with finite, type-appropriate values, run the operator
// via `invoke`, and compare bit-exact against the per-element `gold`. Using the matching element
// type (not raw bytes) keeps SetImage/GetImage element-size-correct and avoids feeding NaN/Inf bit
// patterns into float (whose host/device arithmetic can differ).
//
//   gold:   DT(DT)                                       -- per-element reference (captures op params)
//   invoke: void(cudaStream_t, const Tensor&, Tensor&)   -- runs the operator (captures op params)
template<typename DT, typename GoldFn, typename InvokeFn>
void RunTensorCorrect(int width, int height, int batch, nvcv::ImageFormat fmt, GoldFn gold, InvokeFn invoke)
{
    RunTensorCorrectBuffer<DT>(
        width, height, batch, fmt,
        [gold](const std::vector<DT> &in, int)
        {
            std::vector<DT> out(in.size());
            for (size_t i = 0; i < in.size(); ++i)
            {
                out[i] = gold(in[i]);
            }
            return out;
        },
        invoke);
}

// Dispatch RunTensorCorrect over the u8 / u16 / f32 base type implied by `fmt`. Both `goldFor` and
// `invokeFor` are generic callables returning, for a given element type DT, the per-element gold
// lambda and the operator-invoke lambda respectively. Making invoke a per-DT factory lets operators
// with a dtype-dependent parameter (e.g. Solarize's mid-range threshold, which differs for u8 / u16
// / f32) keep the gold and the invocation in lockstep; parameter-free operators (Invert) simply
// return the same invoke for every DT.
template<typename GoldFactory, typename InvokeFactory>
void RunTensorCorrectDispatch(int width, int height, int batch, nvcv::ImageFormat fmt, GoldFactory goldFor,
                              InvokeFactory invokeFor)
{
    switch (BaseKind(fmt))
    {
    case 2:
        RunTensorCorrect<float>(width, height, batch, fmt, goldFor(float{}), invokeFor(float{}));
        break;
    case 1:
        RunTensorCorrect<uint16_t>(width, height, batch, fmt, goldFor(uint16_t{}), invokeFor(uint16_t{}));
        break;
    default:
        RunTensorCorrect<uint8_t>(width, height, batch, fmt, goldFor(uint8_t{}), invokeFor(uint8_t{}));
        break;
    }
}

// Typed VarShape correctness on a small batch. `gold` is the per-element reference and
// `invoke` runs the operator on an ImageBatchVarShape pair. Each image starts with explicit zero and
// maximum-value sentinels so bound arithmetic is deterministic rather than left to random coverage.
//   gold:   DT(DT)
//   invoke: void(cudaStream_t, const ImageBatchVarShape&, ImageBatchVarShape&)
template<typename DT, typename GoldBufferFn, typename InvokeFn>
void RunVarShapeCorrectBufferTyped(nvcv::ImageFormat fmt, GoldBufferFn goldBuffer, InvokeFn invoke, double maxDiff = 0)
{
    ASSERT_EQ(fmt.numPlanes(), 1) << "RunVarShapeCorrectBufferTyped requires an interleaved format";

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    constexpr std::array widths  = {23, 57, 89};
    constexpr std::array heights = {31, 71, 43};
    const auto           n       = static_cast<int>(widths.size());
    const int            ch      = fmt.numChannels();

    std::vector<nvcv::Image> srcImgs;
    std::vector<nvcv::Image> dstImgs;
    std::vector<int>         ws(n);
    std::vector<int>         hs(n);
    for (int i = 0; i < n; ++i)
    {
        ws[i] = widths[i];
        hs[i] = heights[i];
        srcImgs.emplace_back(nvcv::Size2D{ws[i], hs[i]}, fmt);
        dstImgs.emplace_back(nvcv::Size2D{ws[i], hs[i]}, fmt);
    }

    std::vector<std::vector<DT>> golds(n);
    for (int i = 0; i < n; ++i)
    {
        const size_t    rowElements = static_cast<size_t>(ws[i]) * ch;
        const size_t    rowBytes    = rowElements * sizeof(DT);
        std::vector<DT> hwc(rowElements * hs[i]);
        FillDeterministicValues(hwc, static_cast<size_t>(i));
        hwc[0]   = static_cast<DT>(0);
        hwc[1]   = Bound<DT>();
        golds[i] = goldBuffer(hwc, ch);

        auto idata = srcImgs[i].exportData<nvcv::ImageDataStridedCuda>();
        ASSERT_NE(idata, nullptr);
        ASSERT_EQ(cudaSuccess, cudaMemcpy2D(idata->plane(0).basePtr, idata->plane(0).rowStride, hwc.data(), rowBytes,
                                            rowBytes, hs[i], cudaMemcpyHostToDevice));
    }

    nvcv::ImageBatchVarShape src(n);
    nvcv::ImageBatchVarShape dst(n);
    src.pushBack(srcImgs.begin(), srcImgs.end());
    dst.pushBack(dstImgs.begin(), dstImgs.end());

    ASSERT_NO_THROW(invoke(stream, src, dst));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    for (int i = 0; i < n; ++i)
    {
        SCOPED_TRACE(i);
        const size_t    rowElements = static_cast<size_t>(ws[i]) * ch;
        const size_t    rowBytes    = rowElements * sizeof(DT);
        std::vector<DT> got(rowElements * hs[i]);
        auto            odata = dstImgs[i].exportData<nvcv::ImageDataStridedCuda>();
        ASSERT_NE(odata, nullptr);
        ASSERT_EQ(cudaSuccess, cudaMemcpy2D(got.data(), rowBytes, odata->plane(0).basePtr, odata->plane(0).rowStride,
                                            rowBytes, hs[i], cudaMemcpyDeviceToHost));
        ExpectBuffer(golds[i], got, maxDiff);
    }

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

template<typename DT, typename GoldFn, typename InvokeFn>
void RunVarShapeCorrectTyped(nvcv::ImageFormat fmt, GoldFn gold, InvokeFn invoke)
{
    RunVarShapeCorrectBufferTyped<DT>(
        fmt,
        [gold](const std::vector<DT> &in, int)
        {
            std::vector<DT> out(in.size());
            for (size_t i = 0; i < in.size(); ++i)
            {
                out[i] = gold(in[i]);
            }
            return out;
        },
        invoke);
}

// Backward-compatible RGB8 entry point used by the existing unary element-wise tests.
template<typename GoldFn, typename InvokeFn>
void RunVarShapeCorrect(GoldFn gold, InvokeFn invoke)
{
    RunVarShapeCorrectTyped<uint8_t>(nvcv::FMT_RGB8, gold, invoke);
}

// A single in/out tensor pair must be rejected with NVCV_ERROR_INVALID_ARGUMENT. `invoke` runs the
// operator on the tensor pair; the harness owns the stream + ProtectCall plumbing.
//   invoke: void(cudaStream_t, const Tensor&, const Tensor&)
template<typename InvokeFn>
void ExpectRejected(nvcv::ImageFormat inFmt, nvcv::ImageFormat outFmt, InvokeFn invoke, int width = 32, int height = 24)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    nvcv::Tensor inTensor  = nvcv::util::CreateTensor(1, width, height, inFmt);
    nvcv::Tensor outTensor = nvcv::util::CreateTensor(1, width, height, outFmt);

    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcv::ProtectCall([&] { invoke(stream, inTensor, outTensor); }));

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

template<typename InvokeFn>
void ExpectVarShapeRejected(nvcv::ImageFormat fmt, InvokeFn invoke, int width = 32, int height = 24)
{
    nvcv::Image srcImage({width, height}, fmt);
    nvcv::Image dstImage({width, height}, fmt);

    nvcv::ImageBatchVarShape src(1);
    nvcv::ImageBatchVarShape dst(1);
    src.pushBack(srcImage);
    dst.pushBack(dstImage);

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcv::ProtectCall([&] { invoke(stream, src, dst); }));
    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

template<typename InvokeFn>
void ExpectZeroExtentTensorNoop(const nvcv::TensorShape &shape, nvcv::DataType dtype, InvokeFn invoke)
{
    std::optional<nvcv::Tensor> src;
    std::optional<nvcv::Tensor> dst;
    try
    {
        src.emplace(shape, dtype);
        dst.emplace(shape, dtype);
    }
    catch (const nvcv::Exception &e)
    {
        GTEST_SKIP() << "zero-extent tensors are not constructible: " << e.what();
    }

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));
    EXPECT_EQ(NVCV_SUCCESS, nvcv::ProtectCall([&] { invoke(stream, *src, *dst); }));
    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

template<typename InvokeFn>
void ExpectEmptyVarShapeNoop(InvokeFn invoke)
{
    nvcv::ImageBatchVarShape src(1);
    nvcv::ImageBatchVarShape dst(1);

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));
    EXPECT_EQ(NVCV_SUCCESS, nvcv::ProtectCall([&] { invoke(stream, src, dst); }));
    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

} // namespace nvcv::test::elementwise

#endif // NVCV_TEST_SYSTEM_ELEMENTWISE_OP_HARNESS_HPP
