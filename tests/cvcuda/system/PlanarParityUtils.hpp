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

#ifndef NVCV_TEST_PLANAR_PARITY_UTILS_HPP
#define NVCV_TEST_PLANAR_PARITY_UTILS_HPP

// Shared scaffolding for "planar matches interleaved" parity tests.
//
// An operator that treats channels independently must produce byte-for-byte the same pixels in a
// planar (NCHW/CHW) layout as in the interleaved ((N)HWC) layout. These helpers feed identical data
// through an operator in both layouts and assert the (re-interleaved) planar output equals the
// interleaved output exactly. The op call is supplied as a lambda so every operator reuses the same
// upload/run/download/compare flow; see TestOpResize.cpp and TestOpFlip.cpp for usage.

#include <common/TensorDataUtils.hpp>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <nvcv/BorderType.h>
#include <nvcv/Exception.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorDataAccess.hpp>

#include <array>
#include <cstdint>
#include <cstring>
#include <tuple>
#include <type_traits>
#include <vector>

namespace nvcv::test::planar {

// Deinterleave HWC bytes into C contiguous single-channel planes (no row padding).
inline std::vector<uint8_t> DeinterleaveToPlanes(const std::vector<uint8_t> &hwc, int w, int h, int channels,
                                                 int elemSize)
{
    std::vector<uint8_t> planes(hwc.size());
    const int            planeBytes = w * h * elemSize;
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            for (int c = 0; c < channels; ++c)
                std::memcpy(&planes[c * planeBytes + (y * w + x) * elemSize],
                            &hwc[((y * w + x) * channels + c) * elemSize], elemSize);
    return planes;
}

// Inverse of DeinterleaveToPlanes.
inline std::vector<uint8_t> InterleaveFromPlanes(const std::vector<uint8_t> &planes, int w, int h, int channels,
                                                 int elemSize)
{
    std::vector<uint8_t> hwc(planes.size());
    const int            planeBytes = w * h * elemSize;
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            for (int c = 0; c < channels; ++c)
                std::memcpy(&hwc[((y * w + x) * channels + c) * elemSize],
                            &planes[c * planeBytes + (y * w + x) * elemSize], elemSize);
    return hwc;
}

template<typename T>
inline void FillDeterministicValues(std::vector<uint8_t> &buf, size_t seed)
{
    ASSERT_EQ(size_t{0}, buf.size() % sizeof(T));
    for (size_t k = 0; k < buf.size() / sizeof(T); ++k)
    {
        const auto pattern = static_cast<int>((k * 31 + seed) % 251);
        T          value;
        if constexpr (std::is_floating_point_v<T>)
        {
            value = static_cast<T>((pattern + 1) / 252.0);
        }
        else if constexpr (std::is_signed_v<T>)
        {
            value = static_cast<T>(pattern % 127 - 63);
        }
        else
        {
            value = static_cast<T>(pattern);
        }
        std::memcpy(buf.data() + k * sizeof(T), &value, sizeof(T));
    }
}

// Fill the logical input with bounded values of the actual scalar type. In particular, do not
// reinterpret arbitrary bytes as floats: filter, resize, and normalization operators consume these
// values arithmetically, and NaNs or near-FLT_MAX inputs turn parity into an overflow comparison.
inline void FillDeterministicValues(std::vector<uint8_t> &buf, size_t seed, nvcv::DataType dtype)
{
    switch (static_cast<NVCVDataType>(dtype))
    {
    case NVCV_DATA_TYPE_U8:
        FillDeterministicValues<uint8_t>(buf, seed);
        break;
    case NVCV_DATA_TYPE_S8:
        FillDeterministicValues<int8_t>(buf, seed);
        break;
    case NVCV_DATA_TYPE_U16:
        FillDeterministicValues<uint16_t>(buf, seed);
        break;
    case NVCV_DATA_TYPE_S16:
        FillDeterministicValues<int16_t>(buf, seed);
        break;
    case NVCV_DATA_TYPE_U32:
        FillDeterministicValues<uint32_t>(buf, seed);
        break;
    case NVCV_DATA_TYPE_S32:
        FillDeterministicValues<int32_t>(buf, seed);
        break;
    case NVCV_DATA_TYPE_F32:
        FillDeterministicValues<float>(buf, seed);
        break;
    case NVCV_DATA_TYPE_F64:
        FillDeterministicValues<double>(buf, seed);
        break;
    default:
        FAIL() << "Unsupported planar parity scalar dtype " << static_cast<NVCVDataType>(dtype);
    }
}

// Upload an interleaved HWC host buffer into one sample of a tensor (interleaved or single channel).
inline void UploadInterleavedSample(const nvcv::TensorDataAccessStridedImagePlanar &access, int sample,
                                    const std::vector<uint8_t> &hwc, int /*w*/, int h, int rowStrideBytes)
{
    ASSERT_EQ(cudaSuccess, cudaMemcpy2D(access.sampleData(sample), access.rowStride(), hwc.data(), rowStrideBytes,
                                        rowStrideBytes, h, cudaMemcpyHostToDevice));
}

// Upload C contiguous host planes into one planar (NCHW/CHW) sample of a tensor.
inline void UploadPlanarSample(const nvcv::TensorDataAccessStridedImagePlanar &access, int sample,
                               const std::vector<uint8_t> &planes, int w, int h, int channels, int elemSize)
{
    const int planeBytes = w * h * elemSize;
    for (int c = 0; c < channels; ++c)
    {
        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy2D(access.sampleData(sample) + c * access.chStride(), access.rowStride(),
                               planes.data() + c * planeBytes, w * elemSize, w * elemSize, h, cudaMemcpyHostToDevice));
    }
}

// Download one interleaved sample of a tensor into an HWC host buffer.
inline std::vector<uint8_t> DownloadInterleavedSample(const nvcv::TensorDataAccessStridedImagePlanar &access,
                                                      int sample, int /*w*/, int h, int rowStrideBytes)
{
    std::vector<uint8_t> hwc(h * rowStrideBytes);
    EXPECT_EQ(cudaSuccess, cudaMemcpy2D(hwc.data(), rowStrideBytes, access.sampleData(sample), access.rowStride(),
                                        rowStrideBytes, h, cudaMemcpyDeviceToHost));
    return hwc;
}

// Download one planar sample of a tensor into C contiguous host planes.
inline std::vector<uint8_t> DownloadPlanarSample(const nvcv::TensorDataAccessStridedImagePlanar &access, int sample,
                                                 int w, int h, int channels, int elemSize)
{
    std::vector<uint8_t> planes(w * h * channels * elemSize);
    const int            planeBytes = w * h * elemSize;
    for (int c = 0; c < channels; ++c)
    {
        EXPECT_EQ(cudaSuccess, cudaMemcpy2D(planes.data() + c * planeBytes, w * elemSize,
                                            access.sampleData(sample) + c * access.chStride(), access.rowStride(),
                                            w * elemSize, h, cudaMemcpyDeviceToHost));
    }
    return planes;
}

// Upload one value per image into a tensor shaped ({{numImages}, "N"}, dtype). Used by var-shape
// filter ops whose per-sample parameters (e.g. kernel size / anchor int2 tensors) are passed as
// N-length tensors. Shared so per-op planar tests do not re-implement the upload boilerplate.
template<typename T>
inline void UploadTensorValues(nvcv::Tensor &tensor, const std::vector<T> &values)
{
    auto dev = tensor.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_NE(dev, nullptr);
    ASSERT_EQ(cudaSuccess,
              cudaMemcpy(dev->basePtr(), values.data(), values.size() * sizeof(T), cudaMemcpyHostToDevice));
}

// Create an N-length parameter tensor with the same `value` for every image and upload it.
template<typename T>
inline nvcv::Tensor MakePerImageTensor(int numImages, nvcv::DataType dtype, const T &value)
{
    nvcv::Tensor tensor({{numImages}, "N"}, dtype);
    UploadTensorValues(tensor, std::vector<T>(numImages, value));
    return tensor;
}

// Run the same data through `invoke` in interleaved and planar tensor layouts; require bit-exact
// outputs. `invoke(stream, src, dst, fmt)` performs the operator call (the caller binds op-specific
// parameters); `fmt` is that pair's image format, for ops that need it (e.g. workspace sizing).
// `srcW/srcH` is the input size, `dstW/dstH` the output size (equal for size-preserving ops). The
// data is uploaded at the input size and the result compared at the output size.
template<class OpInvoke>
inline void RunTensorParity(nvcv::ImageFormat planarFmt, nvcv::ImageFormat interleavedFmt, int srcW, int srcH, int dstW,
                            int dstH, int numImages, OpInvoke &&invoke)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    const int channels  = planarFmt.numChannels();
    const int elemSize  = planarFmt.planePixelStrideBytes(0);
    const int srcRowStr = srcW * channels * elemSize;
    const int dstRowStr = dstW * channels * elemSize;

    nvcv::Tensor srcI = nvcv::util::CreateTensor(numImages, srcW, srcH, interleavedFmt);
    nvcv::Tensor dstI = nvcv::util::CreateTensor(numImages, dstW, dstH, interleavedFmt);
    nvcv::Tensor srcP = nvcv::util::CreateTensor(numImages, srcW, srcH, planarFmt);
    nvcv::Tensor dstP = nvcv::util::CreateTensor(numImages, dstW, dstH, planarFmt);

    auto srcIData = srcI.exportData<nvcv::TensorDataStridedCuda>();
    auto dstIData = dstI.exportData<nvcv::TensorDataStridedCuda>();
    auto srcPData = srcP.exportData<nvcv::TensorDataStridedCuda>();
    auto dstPData = dstP.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_TRUE(srcIData && dstIData && srcPData && dstPData);
    ASSERT_TRUE(srcPData->layout() == nvcv::TENSOR_NCHW || srcPData->layout() == nvcv::TENSOR_CHW);

    auto srcIAcc = nvcv::TensorDataAccessStridedImagePlanar::Create(*srcIData);
    auto dstIAcc = nvcv::TensorDataAccessStridedImagePlanar::Create(*dstIData);
    auto srcPAcc = nvcv::TensorDataAccessStridedImagePlanar::Create(*srcPData);
    auto dstPAcc = nvcv::TensorDataAccessStridedImagePlanar::Create(*dstPData);
    ASSERT_TRUE(srcIAcc && dstIAcc && srcPAcc && dstPAcc);

    for (int i = 0; i < numImages; ++i)
    {
        std::vector<uint8_t> hwc(srcH * srcRowStr);
        FillDeterministicValues(hwc, static_cast<size_t>(i) * 101 + 1, planarFmt.planeDataType(0));

        UploadInterleavedSample(*srcIAcc, i, hwc, srcW, srcH, srcRowStr);
        UploadPlanarSample(*srcPAcc, i, DeinterleaveToPlanes(hwc, srcW, srcH, channels, elemSize), srcW, srcH, channels,
                           elemSize);

        // Different sentinels ensure a missing write cannot pass merely because both output allocations
        // happen to contain the same bytes.
        UploadInterleavedSample(*dstIAcc, i, std::vector<uint8_t>(dstH * dstRowStr, 0xA5), dstW, dstH, dstRowStr);
        UploadPlanarSample(*dstPAcc, i, std::vector<uint8_t>(dstH * dstRowStr, 0x5A), dstW, dstH, channels, elemSize);
    }

    invoke(stream, srcI, dstI, interleavedFmt);
    invoke(stream, srcP, dstP, planarFmt);
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    for (int i = 0; i < numImages; ++i)
    {
        SCOPED_TRACE(i);
        auto gpuInter    = DownloadInterleavedSample(*dstIAcc, i, dstW, dstH, dstRowStr);
        auto planesOut   = DownloadPlanarSample(*dstPAcc, i, dstW, dstH, channels, elemSize);
        auto planarInter = InterleaveFromPlanes(planesOut, dstW, dstH, channels, elemSize);

        EXPECT_EQ(gpuInter, planarInter);
    }

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

// Var-shape counterpart of RunTensorParity. `invoke(stream, batchSrc, batchDst)` performs the
// operator call on the two image batches.
template<class OpInvoke>
inline void RunVarShapeParity(nvcv::ImageFormat planarFmt, nvcv::ImageFormat interleavedFmt, int srcW, int srcH,
                              int dstW, int dstH, int numImages, OpInvoke &&invoke)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    const int channels  = planarFmt.numChannels();
    const int elemSize  = planarFmt.planePixelStrideBytes(0);
    const int srcRowStr = srcW * channels * elemSize;

    std::vector<nvcv::Image> srcI;
    std::vector<nvcv::Image> dstI;
    std::vector<nvcv::Image> srcP;
    std::vector<nvcv::Image> dstP;
    for (int i = 0; i < numImages; ++i)
    {
        srcI.emplace_back(nvcv::Size2D{srcW, srcH}, interleavedFmt);
        dstI.emplace_back(nvcv::Size2D{dstW, dstH}, interleavedFmt);
        srcP.emplace_back(nvcv::Size2D{srcW, srcH}, planarFmt);
        dstP.emplace_back(nvcv::Size2D{dstW, dstH}, planarFmt);
    }

    nvcv::ImageBatchVarShape batchSrcI(numImages);
    nvcv::ImageBatchVarShape batchDstI(numImages);
    nvcv::ImageBatchVarShape batchSrcP(numImages);
    nvcv::ImageBatchVarShape batchDstP(numImages);
    batchSrcI.pushBack(srcI.begin(), srcI.end());
    batchDstI.pushBack(dstI.begin(), dstI.end());
    batchSrcP.pushBack(srcP.begin(), srcP.end());
    batchDstP.pushBack(dstP.begin(), dstP.end());

    for (int i = 0; i < numImages; ++i)
    {
        std::vector<uint8_t> hwc(srcH * srcRowStr);
        FillDeterministicValues(hwc, static_cast<size_t>(i) * 101 + 7, planarFmt.planeDataType(0));

        auto idata = srcI[i].exportData<nvcv::ImageDataStridedCuda>();
        ASSERT_EQ(cudaSuccess, cudaMemcpy2D(idata->plane(0).basePtr, idata->plane(0).rowStride, hwc.data(), srcRowStr,
                                            srcRowStr, srcH, cudaMemcpyHostToDevice));

        auto      planes     = DeinterleaveToPlanes(hwc, srcW, srcH, channels, elemSize);
        auto      pdata      = srcP[i].exportData<nvcv::ImageDataStridedCuda>();
        const int planeBytes = srcW * srcH * elemSize;
        ASSERT_EQ(pdata->numPlanes(), channels);
        for (int c = 0; c < channels; ++c)
        {
            ASSERT_EQ(cudaSuccess,
                      cudaMemcpy2D(pdata->plane(c).basePtr, pdata->plane(c).rowStride, planes.data() + c * planeBytes,
                                   srcW * elemSize, srcW * elemSize, srcH, cudaMemcpyHostToDevice));
        }

        const int            dstRowStr   = dstW * channels * elemSize;
        const int            dstPlaneByt = dstW * dstH * elemSize;
        std::vector<uint8_t> interleavedCanary(dstH * dstRowStr, 0xA5);
        auto                 dstIData = dstI[i].exportData<nvcv::ImageDataStridedCuda>();
        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy2D(dstIData->plane(0).basePtr, dstIData->plane(0).rowStride, interleavedCanary.data(),
                               dstRowStr, dstRowStr, dstH, cudaMemcpyHostToDevice));

        std::vector<uint8_t> planarCanary(dstPlaneByt, 0x5A);
        auto                 dstPData = dstP[i].exportData<nvcv::ImageDataStridedCuda>();
        ASSERT_EQ(dstPData->numPlanes(), channels);
        for (int c = 0; c < channels; ++c)
        {
            ASSERT_EQ(cudaSuccess,
                      cudaMemcpy2D(dstPData->plane(c).basePtr, dstPData->plane(c).rowStride, planarCanary.data(),
                                   dstW * elemSize, dstW * elemSize, dstH, cudaMemcpyHostToDevice));
        }
    }

    invoke(stream, batchSrcI, batchDstI, interleavedFmt);
    invoke(stream, batchSrcP, batchDstP, planarFmt);
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    const int dstRowStr   = dstW * channels * elemSize;
    const int dstPlaneByt = dstW * dstH * elemSize;
    for (int i = 0; i < numImages; ++i)
    {
        SCOPED_TRACE(i);
        std::vector<uint8_t> gpuInter(dstH * dstRowStr);
        auto                 idata = dstI[i].exportData<nvcv::ImageDataStridedCuda>();
        EXPECT_EQ(cudaSuccess, cudaMemcpy2D(gpuInter.data(), dstRowStr, idata->plane(0).basePtr,
                                            idata->plane(0).rowStride, dstRowStr, dstH, cudaMemcpyDeviceToHost));

        std::vector<uint8_t> planesOut(dstW * dstH * channels * elemSize);
        auto                 pdata = dstP[i].exportData<nvcv::ImageDataStridedCuda>();
        for (int c = 0; c < channels; ++c)
        {
            EXPECT_EQ(cudaSuccess,
                      cudaMemcpy2D(planesOut.data() + c * dstPlaneByt, dstW * elemSize, pdata->plane(c).basePtr,
                                   pdata->plane(c).rowStride, dstW * elemSize, dstH, cudaMemcpyDeviceToHost));
        }
        auto planarInter = InterleaveFromPlanes(planesOut, dstW, dstH, channels, elemSize);

        EXPECT_EQ(gpuInter, planarInter);
    }

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

// Build two NCHW U8 tensors with the given {N, C, H, W} extents, invoke `invoke(stream, in, out)`,
// and require the call to fail with NVCV_ERROR_INVALID_ARGUMENT. Shared by the crop operators'
// negative planar tests, which reject 2-channel planar input and sample/channel-count mismatches
// before the N*C single-channel flattening. `invoke` binds the operator and its crop geometry.
template<class Invoke>
inline void ExpectPlanarTensorRejected(const std::array<int, 4> &inNCHW, const std::array<int, 4> &outNCHW,
                                       Invoke &&invoke)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    nvcv::Tensor imgIn(
        {
            {inNCHW[0], inNCHW[1], inNCHW[2], inNCHW[3]},
            "NCHW"
    },
        nvcv::TYPE_U8);
    nvcv::Tensor imgOut(
        {
            {outNCHW[0], outNCHW[1], outNCHW[2], outNCHW[3]},
            "NCHW"
    },
        nvcv::TYPE_U8);

    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcv::ProtectCall([&] { invoke(stream, imgIn, imgOut); }));

    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

// Var-shape counterpart of ExpectPlanarTensorRejected: build src/dst image-batch var-shapes with the
// given per-image formats (sizes are arbitrary — negative cases must be rejected regardless of size)
// and require `invoke(stream, batchSrc, batchDst)` to fail with NVCV_ERROR_INVALID_ARGUMENT. The
// operator and its per-sample parameter tensors are bound by the caller's `invoke`. Shared so the
// filter ops' var-shape negative tests do not each re-implement the batch-build/run/expect flow.
template<class Invoke>
inline void ExpectVarShapeRejected(const std::vector<nvcv::ImageFormat> &srcFmts,
                                   const std::vector<nvcv::ImageFormat> &dstFmts, Invoke &&invoke)
{
    ASSERT_EQ(srcFmts.size(), dstFmts.size());
    const auto numImages = static_cast<int>(srcFmts.size());

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    std::vector<nvcv::Image> imgSrc;
    std::vector<nvcv::Image> imgDst;
    for (int i = 0; i < numImages; ++i)
    {
        imgSrc.emplace_back(nvcv::Size2D{32, 32}, srcFmts[i]);
        imgDst.emplace_back(nvcv::Size2D{32, 32}, dstFmts[i]);
    }

    nvcv::ImageBatchVarShape batchSrc(numImages);
    nvcv::ImageBatchVarShape batchDst(numImages);
    batchSrc.pushBack(imgSrc.begin(), imgSrc.end());
    batchDst.pushBack(imgDst.begin(), imgDst.end());

    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcv::ProtectCall([&] { invoke(stream, batchSrc, batchDst); }));

    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

// Var-shape negative case for a filter op: a uniform batch of `batches` images all in
// (srcFmt -> dstFmt) is expected to be rejected. `invoke(stream, src, dst, maxBatches, borderMode)`
// builds the op (with the given maxBatches) and its per-image parameter tensors and runs it with the
// given border. `borderMode` is threaded so the sanitizer out-of-range-border negative row is still
// exercised. Shared so planar-capable filter ops do not each re-implement the batch-build/run/reject
// flow.
template<class Invoke>
inline void ExpectVarShapeUniformFormatRejected(nvcv::ImageFormat srcFmt, nvcv::ImageFormat dstFmt, int batches,
                                                int maxBatches, NVCVBorderType borderMode, Invoke &&invoke)
{
    ExpectVarShapeRejected(std::vector<nvcv::ImageFormat>(batches, srcFmt),
                           std::vector<nvcv::ImageFormat>(batches, dstFmt),
                           [&invoke, maxBatches, borderMode](cudaStream_t s, const nvcv::ImageBatchVarShape &src,
                                                             const nvcv::ImageBatchVarShape &dst)
                           { invoke(s, src, dst, maxBatches, borderMode); });
}

// Var-shape negative case for a filter op: a batch of 3 images where one image has a mismatched
// format (covering both U8/RGB8 mismatch directions) is expected to be rejected. `invoke` is as above.
template<class Invoke>
inline void ExpectVarShapeMixedFormatRejected(Invoke &&invoke)
{
    const nvcv::ImageFormat fmt     = nvcv::FMT_RGB8;
    const int               batches = 3;
    for (auto [inputFmtExtra, outputFmtExtra] : std::vector<std::tuple<nvcv::ImageFormat, nvcv::ImageFormat>>{
             {nvcv::FMT_U8,          fmt},
             {         fmt, nvcv::FMT_U8}
    })
    {
        std::vector<nvcv::ImageFormat> srcFmts(batches - 1, fmt);
        std::vector<nvcv::ImageFormat> dstFmts(batches - 1, fmt);
        srcFmts.push_back(inputFmtExtra);
        dstFmts.push_back(outputFmtExtra);
        ExpectVarShapeRejected(
            srcFmts, dstFmts,
            [&invoke](cudaStream_t s, const nvcv::ImageBatchVarShape &src, const nvcv::ImageBatchVarShape &dst)
            { invoke(s, src, dst, batches, NVCV_BORDER_CONSTANT); });
    }
}

} // namespace nvcv::test::planar

#endif // NVCV_TEST_PLANAR_PARITY_UTILS_HPP
