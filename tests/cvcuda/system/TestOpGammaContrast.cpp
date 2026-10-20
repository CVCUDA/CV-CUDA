/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "ConvUtils.hpp"
#include "Definitions.hpp"
#include "HalfTestUtils.hpp"
#include "PlanarParityUtils.hpp"

#include <common/ValueTests.hpp>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cvcuda/OpGammaContrast.hpp>
#include <cvcuda/cuda_tools/MathWrappers.hpp>
#include <cvcuda/cuda_tools/TypeTraits.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorDataAccess.hpp>

#include <random>

namespace test = nvcv::test;
namespace cuda = nvcv::cuda;

constexpr bool DBG_GAMMA_CONTRAST = false;

TEST(OpGammaContrast, tall_packed_u8_tensor_avoids_legacy_grid_limit)
{
    constexpr int    kWidth  = 16;
    constexpr int    kHeight = 65536 * 8;
    constexpr size_t kBytes  = size_t{kWidth} * kHeight * 3;
    nvcv::Tensor     src(1, {kWidth, kHeight}, nvcv::FMT_RGB8p, nvcv::MemAlignment{}.rowAddr(1));
    nvcv::Tensor     dst(1, {kWidth, kHeight}, nvcv::FMT_RGB8p, nvcv::MemAlignment{}.rowAddr(1));
    nvcv::Tensor     gamma({{1}, "N"}, nvcv::TYPE_F32);
    auto             srcData   = src.exportData<nvcv::TensorDataStridedCuda>();
    auto             dstData   = dst.exportData<nvcv::TensorDataStridedCuda>();
    auto             gammaData = gamma.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_TRUE(srcData && dstData && gammaData);
    nvcv::util::SetTensorTo<float>(*gammaData, 0.75f);
    cudaStream_t stream{};
    ASSERT_EQ(cudaSuccess, cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    ASSERT_EQ(cudaSuccess, cudaMemsetAsync(srcData->basePtr(), 0, kBytes, stream));
    ASSERT_EQ(cudaSuccess, cudaMemsetAsync(dstData->basePtr(), 255, kBytes, stream));
    cvcuda::GammaContrast op(1, 3);
    EXPECT_NO_THROW(op(stream, src, dst, gamma));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    std::vector<uint8_t> output(kBytes);
    ASSERT_EQ(cudaSuccess, cudaMemcpy(output.data(), dstData->basePtr(), kBytes, cudaMemcpyDeviceToHost));
    EXPECT_EQ(std::vector<uint8_t>(kBytes, 0), output);
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpGammaContrast, wide_stride_u8_tensor_capture_avoids_int32_legacy)
{
    nvcv::Tensor srcStorage(3, {16, 1}, nvcv::FMT_RGB8p, nvcv::MemAlignment{}.rowAddr(1));
    nvcv::Tensor dstStorage(3, {16, 1}, nvcv::FMT_RGB8p, nvcv::MemAlignment{}.rowAddr(1));
    nvcv::Tensor gamma({{3}, "N"}, nvcv::TYPE_F32);
    auto         srcData   = srcStorage.exportData<nvcv::TensorDataStridedCuda>();
    auto         dstData   = dstStorage.exportData<nvcv::TensorDataStridedCuda>();
    auto         gammaData = gamma.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_TRUE(srcData && dstData && gammaData);
    nvcv::util::SetTensorTo<uint8_t>(*srcData, 0);
    nvcv::util::SetTensorTo<float>(*gammaData, 0.75f);
    cudaStream_t stream{};
    ASSERT_EQ(cudaSuccess, cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    cvcuda::GammaContrast op(3, 3);
    ASSERT_NO_THROW(op(stream, srcStorage, dstStorage, gamma));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    const auto checkStride = [&](int64_t sampleStride, bool wideSource)
    {
        SCOPED_TRACE(::testing::Message() << sampleStride << ":" << wideSource);
        nvcv::TensorDataStridedCuda::Buffer srcBuffer{};
        srcBuffer.basePtr    = reinterpret_cast<NVCVByte *>(srcData->basePtr());
        srcBuffer.strides[0] = wideSource ? sampleStride : 48;
        srcBuffer.strides[1] = 16;
        srcBuffer.strides[2] = 16;
        srcBuffer.strides[3] = 1;
        auto dstBuffer       = srcBuffer;
        dstBuffer.basePtr    = reinterpret_cast<NVCVByte *>(dstData->basePtr());
        dstBuffer.strides[0] = wideSource ? 48 : sampleStride;
        const nvcv::TensorShape shape{
            {3, 3, 1, 16},
            "NCHW"
        };
        nvcv::Tensor src = nvcv::TensorWrapData(nvcv::TensorDataStridedCuda{shape, nvcv::TYPE_U8, srcBuffer});
        nvcv::Tensor dst = nvcv::TensorWrapData(nvcv::TensorDataStridedCuda{shape, nvcv::TYPE_U8, dstBuffer});
        ASSERT_EQ(cudaSuccess, cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        EXPECT_NO_THROW(op(stream, src, dst, gamma));
        cudaGraph_t graph{};
        ASSERT_EQ(cudaSuccess, cudaStreamEndCapture(stream, &graph));
        size_t nodeCount = 0;
        ASSERT_EQ(cudaSuccess, cudaGraphGetNodes(graph, nullptr, &nodeCount));
        std::vector<cudaGraphNode_t> nodes(nodeCount);
        ASSERT_EQ(cudaSuccess, cudaGraphGetNodes(graph, nodes.data(), &nodeCount));
        int imageKernels = 0;
        for (cudaGraphNode_t node : nodes)
        {
            cudaGraphNodeType type{};
            ASSERT_EQ(cudaSuccess, cudaGraphNodeGetType(node, &type));
            if (type != cudaGraphNodeTypeKernel)
                continue;
            CUDA_KERNEL_NODE_PARAMS parameters{};
            ASSERT_EQ(CUDA_SUCCESS, cuGraphKernelNodeGetParams(node, &parameters));
            EXPECT_EQ(1u, parameters.blockDimY);
            if (parameters.gridDimY == 3 && parameters.gridDimZ == 3)
                ++imageKernels;
        }
        EXPECT_EQ(1, imageKernels);
        EXPECT_EQ(cudaSuccess, cudaGraphDestroy(graph));
    };
    ASSERT_NO_FATAL_FAILURE(checkStride(int64_t{1} << 30, true));
    ASSERT_NO_FATAL_FAILURE(checkStride(int64_t{1} << 30, false));
    ASSERT_NO_FATAL_FAILURE(checkStride(int64_t{1} << 31, true));
    ASSERT_NO_FATAL_FAILURE(checkStride(int64_t{1} << 31, false));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpGammaContrast, ada_planar_u8_tensor_launch)
{
    int device = 0;
    ASSERT_EQ(cudaSuccess, cudaGetDevice(&device));
    cudaDeviceProp properties{};
    ASSERT_EQ(cudaSuccess, cudaGetDeviceProperties(&properties, device));
    if (properties.major != 8 || properties.minor != 9)
    {
        GTEST_SKIP() << "Requires SM89";
    }

    nvcv::Tensor src = nvcv::util::CreateTensor(2, 4096, 8, nvcv::FMT_RGB8p);
    nvcv::Tensor dst = nvcv::util::CreateTensor(2, 4096, 8, nvcv::FMT_RGB8p);
    nvcv::Tensor gamma({{6}, "N"}, nvcv::TYPE_F32);
    auto         srcData   = src.exportData<nvcv::TensorDataStridedCuda>();
    auto         gammaData = gamma.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_TRUE(srcData && gammaData);
    nvcv::util::SetTensorTo<uint8_t>(*srcData, 128);
    nvcv::util::SetTensorTo<float>(*gammaData, 0.75f);
    cudaStream_t stream{};
    ASSERT_EQ(cudaSuccess, cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    cvcuda::GammaContrast op(2, 3);
    ASSERT_NO_THROW(op(stream, src, dst, gamma));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    EXPECT_NO_THROW(op(stream, src, dst, gamma));
    cudaGraph_t graph{};
    ASSERT_EQ(cudaSuccess, cudaStreamEndCapture(stream, &graph));
    size_t nodeCount = 0;
    ASSERT_EQ(cudaSuccess, cudaGraphGetNodes(graph, nullptr, &nodeCount));
    std::vector<cudaGraphNode_t> nodes(nodeCount);
    ASSERT_EQ(cudaSuccess, cudaGraphGetNodes(graph, nodes.data(), &nodeCount));
    int imageKernels = 0;
    for (cudaGraphNode_t node : nodes)
    {
        cudaGraphNodeType type{};
        ASSERT_EQ(cudaSuccess, cudaGraphNodeGetType(node, &type));
        if (type != cudaGraphNodeTypeKernel)
        {
            continue;
        }
        CUDA_KERNEL_NODE_PARAMS parameters{};
        ASSERT_EQ(CUDA_SUCCESS, cuGraphKernelNodeGetParams(node, &parameters));
        if (parameters.blockDimY == 8)
        {
            ++imageKernels;
            EXPECT_EQ(32u, parameters.blockDimX);
            EXPECT_EQ(16u, parameters.gridDimX);
            EXPECT_EQ(1u, parameters.gridDimY);
            EXPECT_EQ(2u, parameters.gridDimZ);
        }
    }
    EXPECT_EQ(1, imageKernels);
    EXPECT_EQ(cudaSuccess, cudaGraphDestroy(graph));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

static int ScaledSize(int size, double scale)
{
    return static_cast<int>(size * scale);
}

static void printChannelRow(const std::vector<uint8_t> &vec, int row, int rowPitch, int bytesPerPixel, int channel)
{
    for (int col = 0; col < rowPitch / bytesPerPixel; col++)
    {
        printf("%4d, ", static_cast<int>(vec[row * rowPitch + col * bytesPerPixel + channel]));
    }
    std::cout << std::endl;
}

static void printChannel(const std::vector<uint8_t> &vec, int height, int rowPitch, int bytesPerPixel, int channel,
                         const std::string &name)
{
    std::cout << "\nPrint " << name << " for channel: " << channel << std::endl;

    for (int row = 0; row < height; row++)
    {
        printChannelRow(vec, row, rowPitch, bytesPerPixel, channel);
    }
}

static void printVec(const std::vector<uint8_t> &vec, int height, int rowPitch, int bytesPerPixel,
                     const std::string &name)
{
    if constexpr (DBG_GAMMA_CONTRAST)
    {
        for (int channel = 0; channel < bytesPerPixel; channel++)
        {
            printChannel(vec, height, rowPitch, bytesPerPixel, channel, name);
        }
        std::cout << std::endl;
    }
}

// The tolerance accounts for the host std::pow/std::rint reference versus device __powf/SaturateCast rounding.
#define VEC_EXPECT_NEAR(vec1, vec2, delta, dtype)                                                                    \
    ASSERT_EQ(vec1.size(), vec2.size());                                                                             \
    for (std::size_t idx = 0; idx < vec1.size() / sizeof(dtype); ++idx)                                              \
    {                                                                                                                \
        EXPECT_NEAR(reinterpret_cast<dtype *>(vec1.data())[idx], reinterpret_cast<dtype *>(vec2.data())[idx], delta) \
            << "At index " << idx;                                                                                   \
    }

namespace {

// uint8 cpu op
template<typename T>
void GammaContrastVarShapeCpuOp(std::vector<T> &hDst, int dstRowStride, nvcv::Size2D dstSize,
                                const std::vector<T> &hSrc, int, nvcv::Size2D, nvcv::ImageFormat fmt,
                                const std::vector<float> &gamma, const int imageIndex, bool perChannel)
{
    assert(fmt.numPlanes() == 1);

    int elementsPerPixel = fmt.numChannels();

    T       *dstPtr = hDst.data();
    const T *srcPtr = hSrc.data();

    for (int dst_y = 0; dst_y < dstSize.h; dst_y++)
    {
        for (int dst_x = 0; dst_x < dstSize.w; dst_x++)
        {
            for (int k = 0; k < elementsPerPixel; k++)
            {
                int   index     = dst_y * dstRowStride + dst_x * elementsPerPixel + k;
                float gamma_tmp = perChannel ? gamma[imageIndex * elementsPerPixel + k] : gamma[imageIndex];
                float tmp       = (srcPtr[index] + 0.0f) / 255.0f;
                auto  out       = static_cast<T>(std::rint(pow(tmp, gamma_tmp) * 255.0f));
                dstPtr[index]   = out;
            }
        }
    }
}

// float cpu op
template<>
void GammaContrastVarShapeCpuOp(std::vector<float> &hDst, int dstRowStride, nvcv::Size2D dstSize,
                                const std::vector<float> &hSrc, int, nvcv::Size2D, nvcv::ImageFormat fmt,
                                const std::vector<float> &gamma, const int imageIndex, bool perChannel)
{
    assert(fmt.numPlanes() == 1);

    int elementsPerPixel = fmt.numChannels();

    for (int dst_y = 0; dst_y < dstSize.h; dst_y++)
    {
        for (int dst_x = 0; dst_x < dstSize.w; dst_x++)
        {
            for (int k = 0; k < elementsPerPixel; k++)
            {
                int   index     = dst_y * dstRowStride + dst_x * elementsPerPixel + k;
                float gamma_tmp = perChannel ? gamma[imageIndex * elementsPerPixel + k] : gamma[imageIndex];
                float out       = nvcv::cuda::clamp(nvcv::cuda::pow(hSrc[index], gamma_tmp), 0.f, 1.f);
                hDst[index]     = out;
            }
        }
    }
}

void GammaContrastVarShapeCpuOpWrapper(std::vector<uint8_t> &hDst, int dstRowStride, nvcv::Size2D dstSize,
                                       const std::vector<uint8_t> &hSrc, int srcRowStride, nvcv::Size2D srcSize,
                                       nvcv::ImageFormat fmt, const std::vector<float> &gamma, const int imageIndex,
                                       bool perChannel, NVCVDataType nvcvDataType)
{
    if (nvcvDataType == NVCV_DATA_TYPE_F32 || nvcvDataType == NVCV_DATA_TYPE_2F32 || nvcvDataType == NVCV_DATA_TYPE_3F32
        || nvcvDataType == NVCV_DATA_TYPE_4F32)
    {
        std::vector<float> src_tmp(hSrc.size() / sizeof(float));
        std::vector<float> dst_tmp(hDst.size() / sizeof(float));
        size_t             copySize = hSrc.size();
        memcpy(static_cast<void *>(src_tmp.data()), static_cast<const void *>(hSrc.data()), copySize);
        memcpy(static_cast<void *>(dst_tmp.data()), static_cast<void *>(hDst.data()), copySize);
        GammaContrastVarShapeCpuOp(dst_tmp, dstRowStride / sizeof(float), dstSize, src_tmp,
                                   srcRowStride / sizeof(float), srcSize, fmt, gamma, imageIndex, perChannel);
        memcpy(static_cast<void *>(hDst.data()), static_cast<void *>(dst_tmp.data()), copySize);
    }
    else
    {
        if (nvcvDataType == NVCV_DATA_TYPE_U16 || nvcvDataType == NVCV_DATA_TYPE_2U16
            || nvcvDataType == NVCV_DATA_TYPE_3U16 || nvcvDataType == NVCV_DATA_TYPE_4U16)
        {
            std::vector<uint16_t> src_tmp(hSrc.size() / sizeof(uint16_t));
            std::vector<uint16_t> dst_tmp(hDst.size() / sizeof(uint16_t));
            size_t                copySize = hSrc.size();
            memcpy(static_cast<void *>(src_tmp.data()), static_cast<const void *>(hSrc.data()), copySize);
            memcpy(static_cast<void *>(dst_tmp.data()), static_cast<void *>(hDst.data()), copySize);
            GammaContrastVarShapeCpuOp(dst_tmp, dstRowStride / sizeof(uint16_t), dstSize, src_tmp,
                                       srcRowStride / sizeof(uint16_t), srcSize, fmt, gamma, imageIndex, perChannel);
            memcpy(static_cast<void *>(hDst.data()), static_cast<void *>(dst_tmp.data()), copySize);
        }
        else
        {
            GammaContrastVarShapeCpuOp(hDst, dstRowStride, dstSize, hSrc, srcRowStride, srcSize, fmt, gamma, imageIndex,
                                       perChannel);
        }
    }
}

} // namespace

// clang-format off

NVCV_TEST_SUITE_P(OpGammaContrast, test::ValueList<int, int, int, NVCVImageFormat, float, bool>
{
    // width, height, batches,                    format,  Gamma,  per channel
    {   5,      5,       1,      NVCV_IMAGE_FORMAT_U8,       0.5,        true},
    {   9,     11,       2,      NVCV_IMAGE_FORMAT_U8,      0.75,        true},
    {  37,      5,       2,      NVCV_IMAGE_FORMAT_U8,      0.65,        true},
    {   12,     7,       3,    NVCV_IMAGE_FORMAT_RGB8,       1.0,        true},
    {   11,    11,       4,   NVCV_IMAGE_FORMAT_RGBA8,       0.4,        true},
    {   7,      8,       3,    NVCV_IMAGE_FORMAT_RGB8,       0.9,        true},
    {   7,      6,       4,   NVCV_IMAGE_FORMAT_RGBA8,       0.8,        true},

    {   5,      5,       1,      NVCV_IMAGE_FORMAT_U8,        0.5,      false},
    {   9,     11,       2,      NVCV_IMAGE_FORMAT_U8,       0.75,      false},
    {   12,     7,       3,    NVCV_IMAGE_FORMAT_RGB8,        1.0,      false},
    {   11,    11,       4,   NVCV_IMAGE_FORMAT_RGBA8,        0.4,      false},
    {   7,      8,       3,    NVCV_IMAGE_FORMAT_RGB8,        0.9,      false},
    {   7,      6,       4,   NVCV_IMAGE_FORMAT_RGBA8,        0.8,      false},
    {  17,      9,       2,     NVCV_IMAGE_FORMAT_U16,        1.2,      false},

    {   5,      5,       1,     NVCV_IMAGE_FORMAT_F32,       0.5,        true},
    {   9,     11,       2,     NVCV_IMAGE_FORMAT_F32,      0.75,        true},
    {   12,     7,       3,  NVCV_IMAGE_FORMAT_RGBf32,       1.0,        true},
    {   11,    11,       4, NVCV_IMAGE_FORMAT_RGBAf32,       0.4,        true},
    {   7,      8,       3,  NVCV_IMAGE_FORMAT_RGBf32,       0.9,        true},
    {   7,      6,       4, NVCV_IMAGE_FORMAT_RGBAf32,       0.8,        true},

    {   5,      5,       1,     NVCV_IMAGE_FORMAT_F32,        0.5,      false},
    {   9,     11,       2,     NVCV_IMAGE_FORMAT_F32,       0.75,      false},
    {   12,     7,       3,  NVCV_IMAGE_FORMAT_RGBf32,        1.0,      false},
    {   11,    11,       4, NVCV_IMAGE_FORMAT_RGBAf32,        0.4,      false},
    {   7,      8,       3,  NVCV_IMAGE_FORMAT_RGBf32,        0.9,      false},
    {   7,      6,       4, NVCV_IMAGE_FORMAT_RGBAf32,        0.8,      false},

    // F16 rows follow the float path (pow, no /255, clamp to [0, 1]) with an FP32-computed gold.
    {   5,      5,       1,     NVCV_IMAGE_FORMAT_F16,        0.5,       true},
    {   12,     7,       3,  NVCV_IMAGE_FORMAT_RGBf16,        0.9,       true},
    {   11,    11,       4, NVCV_IMAGE_FORMAT_RGBAf16,        0.4,       true},
    {   9,     11,       2,     NVCV_IMAGE_FORMAT_F16,       0.75,      false},
    {   7,      6,       4, NVCV_IMAGE_FORMAT_RGBAf16,        0.8,      false},
});

// clang-format on

TEST_P(OpGammaContrast, varshape_correct_output)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    int width   = GetParamValue<0>();
    int height  = GetParamValue<1>();
    int batches = GetParamValue<2>();

    nvcv::ImageFormat format{GetParamValue<3>()};

    NVCVDataType nvcvDataType;
    ASSERT_EQ(NVCV_SUCCESS, nvcvImageFormatGetPlaneDataType(static_cast<NVCVImageFormat>(format), 0, &nvcvDataType));
    float gamma       = GetParamValue<4>();
    bool  isFloatTest = false;
    bool  isHalfTest  = false;

    bool perChannel = GetParamValue<5>();

    // Create input varshape
    std::default_random_engine            rng;
    std::uniform_int_distribution         udistWidth(ScaledSize(width, 0.8), ScaledSize(width, 1.1));
    std::uniform_int_distribution         udistHeight(ScaledSize(height, 0.8), ScaledSize(height, 1.1));
    std::uniform_real_distribution<float> udistGamma(gamma * 0.8f, 1.f);

    std::vector<nvcv::Image> imgSrc;

    std::vector<std::vector<uint8_t>> srcVec(batches);
    std::vector<int>                  srcVecRowStride(batches);

    for (int i = 0; i < batches; ++i)
    {
        imgSrc.emplace_back(nvcv::Size2D{udistWidth(rng), udistHeight(rng)}, format);

        int srcRowStride   = imgSrc[i].size().w * format.planePixelStrideBytes(0);
        srcVecRowStride[i] = srcRowStride;

        std::uniform_int_distribution<uint8_t> udist(0, 255);
        std::uniform_real_distribution<float>  udistf(0.f, 1.f);

        srcVec[i].resize(imgSrc[i].size().h * srcRowStride);
        switch (nvcvDataType)
        {
        case NVCV_DATA_TYPE_F32:
        case NVCV_DATA_TYPE_2F32:
        case NVCV_DATA_TYPE_3F32:
        case NVCV_DATA_TYPE_4F32:
            isFloatTest = true;
            for (size_t idx = 0; idx < (srcVec[i].size() / sizeof(float)); ++idx)
            {
                reinterpret_cast<float *>(srcVec[i].data())[idx] = udistf(rng);
            }
            break;
        case NVCV_DATA_TYPE_F16:
        case NVCV_DATA_TYPE_2F16:
        case NVCV_DATA_TYPE_3F16:
        case NVCV_DATA_TYPE_4F16:
            // Same [0, 1] range as the float rows, quantized to half so the FP32 gold consumes
            // exactly the values the kernel reads.
            isHalfTest = true;
            for (size_t idx = 0; idx < (srcVec[i].size() / sizeof(__half)); ++idx)
            {
                reinterpret_cast<__half *>(srcVec[i].data())[idx] = __float2half(udistf(rng));
            }
            break;
        default:
            std::ranges::generate(srcVec[i], [&udist, &rng]() { return udist(rng); });
            break;
        }

        auto imgData = imgSrc[i].exportData<nvcv::ImageDataStridedCuda>();
        ASSERT_NE(imgData, nvcv::NullOpt);

        printVec(srcVec[i], imgSrc[i].size().h, srcVecRowStride[i], format.numChannels(), "input");

        // Copy input data to the GPU
        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy2DAsync(imgData->plane(0).basePtr, imgData->plane(0).rowStride, srcVec[i].data(),
                                    srcRowStride, srcRowStride, imgSrc[i].size().h, cudaMemcpyHostToDevice, stream));
    }

    nvcv::ImageBatchVarShape batchSrc(batches);
    batchSrc.pushBack(imgSrc.begin(), imgSrc.end());

    // Create output varshape
    std::vector<nvcv::Image> imgDst;
    for (int i = 0; i < batches; ++i)
    {
        imgDst.emplace_back(imgSrc[i].size(), imgSrc[i].format());
    }
    nvcv::ImageBatchVarShape batchDst(batches);
    batchDst.pushBack(imgDst.begin(), imgDst.end());

    // Create gamma tensor
    std::vector<float> gammaVec;
    if (perChannel)
    {
        gammaVec.resize(batches * format.numChannels());
    }
    else
    {
        gammaVec.resize(batches);
    }
    std::ranges::generate(gammaVec, [&udistGamma, &rng]() { return udistGamma(rng); });

    auto         nElements = static_cast<int>(gammaVec.size());
    nvcv::Tensor gammaTensor({{nElements}, "N"}, nvcv::TYPE_F32);
    {
        auto dev = gammaTensor.exportData<nvcv::TensorDataStridedCuda>();
        ASSERT_NE(dev, nullptr);

        ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(dev->basePtr(), gammaVec.data(), gammaVec.size() * sizeof(float),
                                               cudaMemcpyHostToDevice, stream));
    }

    // Run operator
    cvcuda::GammaContrast gammacontrastOp(batches, format.numChannels());

    EXPECT_NO_THROW(gammacontrastOp(stream, batchSrc, batchDst, gammaTensor));

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    // Check test data against gold
    for (int i = 0; i < batches; ++i)
    {
        SCOPED_TRACE(i);

        const auto srcData = imgSrc[i].exportData<nvcv::ImageDataStridedCuda>();
        ASSERT_EQ(srcData->numPlanes(), 1);
        int srcWidth  = srcData->plane(0).width;
        int srcHeight = srcData->plane(0).height;

        const auto dstData = imgDst[i].exportData<nvcv::ImageDataStridedCuda>();
        ASSERT_EQ(dstData->numPlanes(), 1);

        int dstWidth  = dstData->plane(0).width;
        int dstHeight = dstData->plane(0).height;

        int dstRowStride = dstWidth * format.planePixelStrideBytes(0);
        int srcRowStride = dstWidth * format.planePixelStrideBytes(0);

        std::vector<uint8_t> testVec(dstHeight * dstRowStride);

        // Copy output data to Host
        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy2D(testVec.data(), dstRowStride, dstData->plane(0).basePtr, dstData->plane(0).rowStride,
                               dstRowStride, dstHeight, cudaMemcpyDeviceToHost));

        // Copy output data to Host
        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy2D(testVec.data(), dstRowStride, dstData->plane(0).basePtr, dstData->plane(0).rowStride,
                               dstRowStride, // vec has no padding
                               dstHeight, cudaMemcpyDeviceToHost));

        if (isHalfTest)
        {
            // F16 gold: run the FP32 reference (pow, clamp to [0, 1]) on the widened half input.
            // Bit-exactness against the FP32 gold is not applicable to the half output; pow is a
            // transcendental chain, so per the HalfTestUtils.hpp policy the bound is kUlps = 8
            // half-ULPs at the reference magnitude.
            std::vector<float> srcF = test::HalfBytesToFloat(srcVec[i]);
            std::vector<float> goldF(srcF.size());
            GammaContrastVarShapeCpuOp(goldF, dstRowStride / static_cast<int>(sizeof(__half)), {dstWidth, dstHeight},
                                       srcF, srcRowStride / static_cast<int>(sizeof(__half)), {srcWidth, srcHeight},
                                       format, gammaVec, i, perChannel);
            test::ExpectNearHalfUlps(goldF, test::HalfBytesToFloat(testVec), 8.f);
            continue;
        }

        std::vector<uint8_t> goldVec(dstHeight * dstRowStride);
        std::ranges::generate(goldVec, []() { return 0; });

        // Generate gold result
        GammaContrastVarShapeCpuOpWrapper(goldVec, dstRowStride, {dstWidth, dstHeight}, srcVec[i], srcRowStride,
                                          {srcWidth, srcHeight}, format, gammaVec, i, perChannel, nvcvDataType);

        printVec(goldVec, srcHeight, dstRowStride, format.numChannels(), "golden output");

        printVec(testVec, srcHeight, dstRowStride, format.numChannels(), "operator output");

        if (!isFloatTest)
        {
            EXPECT_EQ(testVec, goldVec);
        }
        else
        {
            VEC_EXPECT_NEAR(testVec, goldVec, 1E-6F, float);
        }
    }
}

// clang-format off

NVCV_TEST_SUITE_P(OpGammaContrast_Negative, test::ValueList<int, nvcv::ImageFormat, nvcv::ImageFormat>
{
    // batches, inFmt, outFmt
    {6, nvcv::FMT_U8, nvcv::FMT_U8}, // larger than max batches
    {2, nvcv::FMT_RGBA8, nvcv::FMT_RGBA8}, // larger than max channels
    {2, nvcv::FMT_RGB8p, nvcv::FMT_RGB8}, // input/output format mismatch (planar in, interleaved out)
    {2, nvcv::FMT_F64, nvcv::FMT_F64}, // unsupported data type (64-bit float; F16 is now valid)
    {2, nvcv::FMT_U8, nvcv::FMT_S8},
});

// clang-format on

TEST_P(OpGammaContrast_Negative, op)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    int               batches = GetParamValue<0>();
    nvcv::ImageFormat inFmt   = GetParamValue<1>();
    nvcv::ImageFormat outFmt  = GetParamValue<2>();

    int width       = 24;
    int height      = 24;
    int maxBatches  = 5;
    int maxChannels = 3;

    // Create input varshape
    std::default_random_engine    rng;
    std::uniform_int_distribution udistWidth(ScaledSize(width, 0.8), ScaledSize(width, 1.1));
    std::uniform_int_distribution udistHeight(ScaledSize(height, 0.8), ScaledSize(height, 1.1));
    std::vector<nvcv::Image>      imgSrc;

    for (int i = 0; i < batches; ++i)
    {
        imgSrc.emplace_back(nvcv::Size2D{udistWidth(rng), udistHeight(rng)}, inFmt);
    }
    nvcv::ImageBatchVarShape batchSrc(batches);
    batchSrc.pushBack(imgSrc.begin(), imgSrc.end());

    // Create output varshape
    std::vector<nvcv::Image> imgDst;
    for (int i = 0; i < batches; ++i)
    {
        imgDst.emplace_back(imgSrc[i].size(), outFmt);
    }
    nvcv::ImageBatchVarShape batchDst(batches);
    batchDst.pushBack(imgDst.begin(), imgDst.end());

    nvcv::Tensor gammaTensor({{batches}, "N"}, nvcv::TYPE_F32); // not per channel

    // Run operator
    cvcuda::GammaContrast gammacontrastOp(maxBatches, maxChannels);

    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcv::ProtectCall([&gammacontrastOp, &stream, &batchSrc, &batchDst, &gammaTensor]
                                { gammacontrastOp(stream, batchSrc, batchDst, gammaTensor); }));

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpGammaContrast_Negative, varshape_hasDifferentFormat)
{
    cudaStream_t stream;
    EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    const int batches     = 5;
    int       maxChannels = 4;

    int srcWidthBase  = 24;
    int srcHeightBase = 24;

    nvcv::ImageFormat fmt = nvcv::FMT_RGB8;

    std::vector<std::tuple<nvcv::ImageFormat, nvcv::ImageFormat>> testSet{
        {nvcv::FMT_RGBA8,  nvcv::FMT_RGB8},
        { nvcv::FMT_RGB8, nvcv::FMT_RGBA8}
    };

    for (const auto &[inputFmtExtra, outputFmtExtra] : testSet)
    {
        // Create input and output
        std::default_random_engine    randEng;
        std::uniform_int_distribution rndSrcWidth(ScaledSize(srcWidthBase, 0.8), ScaledSize(srcWidthBase, 1.1));
        std::uniform_int_distribution rndSrcHeight(ScaledSize(srcHeightBase, 0.8), ScaledSize(srcHeightBase, 1.1));

        nvcv::Tensor gammaTensor({{batches}, "N"}, nvcv::TYPE_F32); // not per channel

        std::vector<nvcv::Image> imgSrc;

        std::vector<nvcv::Image> imgDst;

        for (int i = 0; i < batches - 1; ++i)
        {
            int tmpWidth  = i == 0 ? srcWidthBase : rndSrcWidth(randEng);
            int tmpHeight = i == 0 ? srcHeightBase : rndSrcHeight(randEng);

            imgSrc.emplace_back(nvcv::Size2D{tmpWidth, tmpHeight}, fmt);
            imgDst.emplace_back(nvcv::Size2D{tmpWidth, tmpHeight}, fmt);
        }
        imgSrc.emplace_back(imgSrc[0].size(), inputFmtExtra);
        imgDst.emplace_back(imgSrc.back().size(), outputFmtExtra);

        nvcv::ImageBatchVarShape batchSrc(batches);
        batchSrc.pushBack(imgSrc.begin(), imgSrc.end());

        nvcv::ImageBatchVarShape batchDst(batches);
        batchDst.pushBack(imgDst.begin(), imgDst.end());

        // Run operator
        cvcuda::GammaContrast gammacontrastOp(batches, maxChannels);
        EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
                  nvcv::ProtectCall([&gammacontrastOp, &stream, &batchSrc, &batchDst, &gammaTensor]
                                    { gammacontrastOp(stream, batchSrc, batchDst, gammaTensor); }));
    }

    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpGammaContrast_Negative, gamma_length_must_match_batch_or_batch_channels)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    const int batches     = 3;
    const int channels    = 3;
    const int width       = 8;
    const int height      = 8;
    const int gammaLength = batches + 1;

    std::vector<nvcv::Image> imgSrc;
    std::vector<nvcv::Image> imgDst;

    for (int i = 0; i < batches; ++i)
    {
        imgSrc.emplace_back(nvcv::Size2D{width, height}, nvcv::FMT_RGB8);
        imgDst.emplace_back(nvcv::Size2D{width, height}, nvcv::FMT_RGB8);
    }

    nvcv::ImageBatchVarShape batchSrc(batches);
    batchSrc.pushBack(imgSrc.begin(), imgSrc.end());

    nvcv::ImageBatchVarShape batchDst(batches);
    batchDst.pushBack(imgDst.begin(), imgDst.end());

    nvcv::Tensor          gammaTensor({{gammaLength}, "N"}, nvcv::TYPE_F32);
    cvcuda::GammaContrast gammacontrastOp(batches, channels);

    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcv::ProtectCall([&] { gammacontrastOp(stream, batchSrc, batchDst, gammaTensor); }));

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpGammaContrast_Negative, gamma_tensor_must_be_float32)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    constexpr int samples = 2;
    constexpr int width   = 8;
    constexpr int height  = 8;

    nvcv::Tensor input  = nvcv::util::CreateTensor(samples, width, height, nvcv::FMT_RGB8);
    nvcv::Tensor output = nvcv::util::CreateTensor(samples, width, height, nvcv::FMT_RGB8);
    nvcv::Tensor gammas({{samples}, "N"}, nvcv::TYPE_U8);

    cvcuda::GammaContrast gammacontrastOp(samples, 3);

    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcv::ProtectCall([&] { gammacontrastOp(stream, input, output, gammas); }));

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpGammaContrast_Negative, varshape_gamma_tensor_must_be_float32)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    constexpr int samples = 2;
    constexpr int width   = 8;
    constexpr int height  = 8;

    std::vector<nvcv::Image> inputImages;
    std::vector<nvcv::Image> outputImages;
    for (int i = 0; i < samples; ++i)
    {
        inputImages.emplace_back(nvcv::Size2D{width, height}, nvcv::FMT_RGB8);
        outputImages.emplace_back(nvcv::Size2D{width, height}, nvcv::FMT_RGB8);
    }

    nvcv::ImageBatchVarShape input(samples);
    input.pushBack(inputImages.begin(), inputImages.end());
    nvcv::ImageBatchVarShape output(samples);
    output.pushBack(outputImages.begin(), outputImages.end());
    nvcv::Tensor gammas({{samples}, "N"}, nvcv::TYPE_U8);

    cvcuda::GammaContrast gammacontrastOp(samples, 3);

    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcv::ProtectCall([&] { gammacontrastOp(stream, input, output, gammas); }));

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpGammaContrast_Negative, input_output_data_types_and_channels_must_match)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    const int batches = 2;
    const int width   = 8;
    const int height  = 8;

    const std::array<std::tuple<nvcv::ImageFormat, nvcv::ImageFormat, int, const char *>, 2> testCases{
        {{nvcv::FMT_U16, nvcv::FMT_F32, 1, "dtype"}, {nvcv::FMT_RGB8, nvcv::FMT_RGBA8, 4, "channels"}}
    };

    for (auto [inputFormat, outputFormat, maxChannels, name] : testCases)
    {
        SCOPED_TRACE(name);

        std::vector<nvcv::Image> imgSrc;
        std::vector<nvcv::Image> imgDst;

        for (int i = 0; i < batches; ++i)
        {
            imgSrc.emplace_back(nvcv::Size2D{width, height}, inputFormat);
            imgDst.emplace_back(nvcv::Size2D{width, height}, outputFormat);
        }

        nvcv::ImageBatchVarShape batchSrc(batches);
        batchSrc.pushBack(imgSrc.begin(), imgSrc.end());

        nvcv::ImageBatchVarShape batchDst(batches);
        batchDst.pushBack(imgDst.begin(), imgDst.end());

        nvcv::Tensor          gammaTensor({{batches}, "N"}, nvcv::TYPE_F32);
        cvcuda::GammaContrast gammacontrastOp(batches, maxChannels);

        EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
                  nvcv::ProtectCall([&] { gammacontrastOp(stream, batchSrc, batchDst, gammaTensor); }));
    }

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

namespace {

template<typename Submit>
void TestTensorInputOutputShapeMismatches(Submit &&submit)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    constexpr int samples = 2;
    constexpr int width   = 8;
    constexpr int height  = 8;

    nvcv::Tensor input = nvcv::util::CreateTensor(samples, width, height, nvcv::FMT_RGB8);

    struct OutputShape
    {
        int         samples;
        int         width;
        int         height;
        const char *name;
    };

    const std::array<OutputShape, 3> outputShapes{
        {{samples - 1, width, height, "samples"},
         {samples, width - 1, height, "width"},
         {samples, width, height - 1, "height"}}
    };

    for (const auto &outputShape : outputShapes)
    {
        SCOPED_TRACE(outputShape.name);
        nvcv::Tensor output
            = nvcv::util::CreateTensor(outputShape.samples, outputShape.width, outputShape.height, nvcv::FMT_RGB8);

        EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcv::ProtectCall([&] { submit(stream, input, output); }));
    }

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

} // namespace

TEST(OpGammaContrast_Negative, tensor_input_output_shapes_must_match)
{
    constexpr int         samples = 2;
    nvcv::Tensor          gammaTensor({{samples}, "N"}, nvcv::TYPE_F32);
    cvcuda::GammaContrast gammacontrastOp(samples, 3);

    TestTensorInputOutputShapeMismatches(
        [&gammacontrastOp, &gammaTensor](cudaStream_t stream, const nvcv::Tensor &input, const nvcv::Tensor &output)
        { gammacontrastOp(stream, input, output, gammaTensor); });
}

// The scalar (host-float) overload must reject sample/width/height mismatches too -- grid sizing is
// derived from the source access, so a smaller output would otherwise drive out-of-bounds writes.
TEST(OpGammaContrast_Negative, scalar_tensor_input_output_shapes_must_match)
{
    constexpr int         samples = 2;
    constexpr float       gamma   = 1.0f;
    constexpr float       gain    = 1.0f;
    cvcuda::GammaContrast gammacontrastOp(samples, 3);

    TestTensorInputOutputShapeMismatches(
        [&gammacontrastOp](cudaStream_t stream, const nvcv::Tensor &input, const nvcv::Tensor &output)
        { gammacontrastOp(stream, input, output, gamma, gain); });
}

TEST(OpGammaContrast_Negative, varshape_input_output_shapes_must_match)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    constexpr int samples = 2;
    constexpr int width   = 8;
    constexpr int height  = 8;

    nvcv::ImageBatchVarShape input(samples);
    for (int i = 0; i < samples; ++i)
    {
        input.pushBack(nvcv::Image({width, height}, nvcv::FMT_RGB8));
    }

    nvcv::Tensor          gammaTensor({{samples}, "N"}, nvcv::TYPE_F32);
    cvcuda::GammaContrast gammacontrastOp(samples, 3);

    const std::array<std::pair<std::vector<nvcv::Size2D>, const char *>, 3> outputShapes{
        {{{{width, height}}, "samples"},
         {{{width, height}, {width - 1, height}}, "width"},
         {{{width, height}, {width, height - 1}}, "height"}}
    };

    for (const auto &[sizes, name] : outputShapes)
    {
        SCOPED_TRACE(name);
        nvcv::ImageBatchVarShape output(static_cast<int32_t>(sizes.size()));
        for (nvcv::Size2D size : sizes)
        {
            output.pushBack(nvcv::Image(size, nvcv::FMT_RGB8));
        }

        EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
                  nvcv::ProtectCall([&] { gammacontrastOp(stream, input, output, gammaTensor); }));
    }

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpGammaContrast_Negative, create_with_null_handle)
{
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, cvcudaGammaContrastCreate(nullptr, 4, 4));
}

TEST(OpGammaContrast_Negative, create_rejects_non_positive_max_batch_or_channels)
{
    const std::vector<std::pair<int32_t, int32_t>> invalidLimits = {
        { 0,  4},
        {-1,  4},
        { 4,  0},
        { 4, -1}
    };

    for (auto [maxBatchSize, maxChannelCount] : invalidLimits)
    {
        NVCVOperatorHandle handle = nullptr;

        EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, cvcudaGammaContrastCreate(&handle, maxBatchSize, maxChannelCount))
            << "maxBatchSize=" << maxBatchSize << ", maxChannelCount=" << maxChannelCount;
        EXPECT_EQ(nullptr, handle);

        if (handle != nullptr)
        {
            nvcvOperatorDestroy(handle);
        }
    }
}

#undef VEC_EXPECT_NEAR

// ---------------------------------------------------------------------------
// Planar (NCHW/CHW) layout support
//
// Gamma contrast applies x**gamma (or 255*(x/255)**gamma) per pixel per channel, independently of the
// other channels, so a planar input is processed plane-by-plane and the per-channel gamma maps to the
// same value the interleaved kernel reads. These tests run identical data + the same gamma tensor in
// both layouts through cvcuda::GammaContrast and require the (re-interleaved) planar output to match
// the interleaved output bit-for-bit. Scaffolding lives in PlanarParityUtils.hpp.
namespace {

const nvcv::ImageFormat FMT_RGBU16{nvcv::ColorModel::RGB,    nvcv::CSPEC_UNDEFINED, nvcv::MemLayout::PITCH_LINEAR,
                                   nvcv::DataKind::UNSIGNED, nvcv::Swizzle::S_XYZ1, nvcv::Packing::X16_Y16_Z16};
const nvcv::ImageFormat FMT_RGBU16p{nvcv::ColorModel::RGB,    nvcv::CSPEC_UNDEFINED, nvcv::MemLayout::PITCH_LINEAR,
                                    nvcv::DataKind::UNSIGNED, nvcv::Swizzle::S_XYZ0, nvcv::Packing::X16,
                                    nvcv::Packing::X16,       nvcv::Packing::X16};

float DeterministicGammaValue(size_t idx, float minValue, float maxValue, int salt)
{
    const int bucket = (static_cast<int>(idx) * 37 + salt) % 101;
    return minValue + (maxValue - minValue) * static_cast<float>(bucket) / 100.f;
}

uint8_t DeterministicByteValue(size_t idx, int sample)
{
    return static_cast<uint8_t>((static_cast<int>(idx) * 13 + sample * 29 + 17) % 256);
}

float DeterministicUnitValue(size_t idx, int sample)
{
    const int bucket = (static_cast<int>(idx) * 17 + sample * 31 + 11) % 100;
    return static_cast<float>(bucket) / 99.f;
}

struct TensorTestData
{
    NVCVDataType                      dataType;
    bool                              isFloat;
    bool                              isHalf;
    int                               rowStride;
    size_t                            sampleBytes;
    std::vector<std::vector<uint8_t>> hostSamples;
};

void UploadDeterministicTensorInput(nvcv::Tensor &tensor, int samples, int width, int height, nvcv::ImageFormat format,
                                    TensorTestData &testData)
{
    auto tensorData = tensor.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_NE(tensorData, nullptr);
    auto tensorAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*tensorData);
    ASSERT_TRUE(tensorAccess);

    ASSERT_EQ(NVCV_SUCCESS,
              nvcvImageFormatGetPlaneDataType(static_cast<NVCVImageFormat>(format), 0, &testData.dataType));
    testData.isFloat = testData.dataType == NVCV_DATA_TYPE_F32 || testData.dataType == NVCV_DATA_TYPE_2F32
                    || testData.dataType == NVCV_DATA_TYPE_3F32 || testData.dataType == NVCV_DATA_TYPE_4F32;
    testData.isHalf = testData.dataType == NVCV_DATA_TYPE_F16 || testData.dataType == NVCV_DATA_TYPE_2F16
                   || testData.dataType == NVCV_DATA_TYPE_3F16 || testData.dataType == NVCV_DATA_TYPE_4F16;
    const int bytesPerElement = format.planePixelStrideBytes(0) / format.numChannels();
    testData.rowStride        = width * format.numChannels() * bytesPerElement;
    testData.sampleBytes      = static_cast<size_t>(testData.rowStride) * height;
    testData.hostSamples.assign(samples, std::vector<uint8_t>(testData.sampleBytes));

    for (int sample = 0; sample < samples; ++sample)
    {
        auto &hostSample = testData.hostSamples[sample];
        if (testData.isFloat)
        {
            auto *values = reinterpret_cast<float *>(hostSample.data());
            for (size_t i = 0; i < testData.sampleBytes / sizeof(float); ++i)
            {
                values[i] = DeterministicUnitValue(i, sample);
            }
        }
        else if (testData.isHalf)
        {
            // Same unit-range values as float, quantized to half so the FP32 gold consumes
            // exactly the values the kernel reads.
            auto *values = reinterpret_cast<__half *>(hostSample.data());
            for (size_t i = 0; i < testData.sampleBytes / sizeof(__half); ++i)
            {
                values[i] = __float2half(DeterministicUnitValue(i, sample));
            }
        }
        else
        {
            for (size_t i = 0; i < hostSample.size(); ++i)
            {
                hostSample[i] = DeterministicByteValue(i, sample);
            }
        }

        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy2D(tensorAccess->sampleData(sample), tensorAccess->rowStride(), hostSample.data(),
                               testData.rowStride, testData.rowStride, height, cudaMemcpyHostToDevice));
    }
}

void DownloadTensorSample(const nvcv::Tensor &tensor, int sample, int rowStride, int height,
                          std::vector<uint8_t> &hostSample)
{
    auto tensorData = tensor.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_NE(tensorData, nullptr);
    auto tensorAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*tensorData);
    ASSERT_TRUE(tensorAccess);

    hostSample.resize(static_cast<size_t>(rowStride) * height);
    ASSERT_EQ(cudaSuccess, cudaMemcpy2D(hostSample.data(), rowStride, tensorAccess->sampleData(sample),
                                        tensorAccess->rowStride(), rowStride, height, cudaMemcpyDeviceToHost));
}

void UploadGammaTensor(nvcv::Tensor &tensor, const std::vector<float> &values)
{
    auto tensorData = tensor.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_NE(tensorData, nullptr);
    ASSERT_EQ(cudaSuccess,
              cudaMemcpy(tensorData->basePtr(), values.data(), values.size() * sizeof(float), cudaMemcpyHostToDevice));
}

void ExpectTensorNear(const std::vector<uint8_t> &got, const std::vector<uint8_t> &gold, bool isFloat)
{
    ASSERT_EQ(got.size(), gold.size());
    if (isFloat)
    {
        const auto *actual   = reinterpret_cast<const float *>(got.data());
        const auto *expected = reinterpret_cast<const float *>(gold.data());
        // Host std::pow and device __powf have different precision; this bounds the expected approximation error.
        for (size_t i = 0; i < got.size() / sizeof(float); ++i)
        {
            EXPECT_NEAR(actual[i], expected[i], 1e-4f) << "at " << i;
        }
    }
    else
    {
        // Integer results can differ by one when host and device power approximations reach the cast boundary.
        for (size_t i = 0; i < got.size(); ++i)
        {
            EXPECT_NEAR(static_cast<int>(got[i]), static_cast<int>(gold[i]), 1) << "at " << i;
        }
    }
}

void RunGammaContrastPlanarParityCase(nvcv::ImageFormat planarFmt, nvcv::ImageFormat interleavedFmt, int w, int h,
                                      int numImages, bool perChannel)
{
    const int channels  = interleavedFmt.numChannels();
    const int nElements = perChannel ? numImages * channels : numImages;

    // Build the gamma tensor once (same instance feeds both layouts).
    std::vector<float> gammaVec(nElements);
    for (int i = 0; i < nElements; ++i)
    {
        gammaVec[i] = DeterministicGammaValue(i, 0.5f, 2.0f, 23);
    }

    nvcv::Tensor gammaTensor({{nElements}, "N"}, nvcv::TYPE_F32);
    ASSERT_NO_FATAL_FAILURE(UploadGammaTensor(gammaTensor, gammaVec));

    test::planar::RunVarShapeParity(
        planarFmt, interleavedFmt, w, h, w, h, numImages,
        [numImages, channels, &gammaTensor](cudaStream_t s, const nvcv::ImageBatchVarShape &src,
                                            const nvcv::ImageBatchVarShape &dst, nvcv::ImageFormat)
        {
            cvcuda::GammaContrast op(numImages, channels);
            EXPECT_NO_THROW(op(s, src, dst, gammaTensor));
        });
}

void RunGammaContrastPaddedPerChannelGammaParity()
{
    constexpr int     numImages     = 2;
    constexpr int     channels      = 3;
    constexpr int64_t sampleStride  = sizeof(float);
    constexpr int64_t channelStride = 3 * sizeof(float);
    constexpr size_t  bufferSize    = channels * channelStride;

    const std::array<std::array<float, channels>, numImages> gammaValues{
        std::array<float, channels>{0.6f, 0.9f, 1.2f},
         std::array<float, channels>{1.5f, 1.8f, 2.1f}
    };
    std::vector<uint8_t> paddedGamma(bufferSize, 0);
    for (int channel = 0; channel < channels; ++channel)
    {
        for (int image = 0; image < numImages; ++image)
        {
            std::memcpy(paddedGamma.data() + channel * channelStride + image * sampleStride,
                        &gammaValues[image][channel], sizeof(float));
        }
    }

    NVCVByte *allocation{};
    ASSERT_EQ(cudaSuccess, cudaMalloc(reinterpret_cast<void **>(&allocation), bufferSize));
    if (allocation == nullptr)
    {
        return;
    }
    ASSERT_EQ(cudaSuccess, cudaMemcpy(allocation, paddedGamma.data(), bufferSize, cudaMemcpyHostToDevice));

    nvcv::TensorDataStridedCuda::Buffer buffer{};
    buffer.basePtr           = allocation;
    buffer.strides[0]        = channelStride;
    buffer.strides[1]        = sampleStride;
    nvcv::Tensor gammaTensor = nvcv::TensorWrapData(
        nvcv::TensorDataStridedCuda{
            nvcv::TensorShape{{channels, numImages}, "CN"},
            nvcv::TYPE_F32, buffer
    },
        nvcv::TensorDataCleanupCallback{[allocation](const nvcv::TensorData &)
                                        {
                                            cudaFree(allocation);
                                        }});

    test::planar::RunVarShapeParity(nvcv::FMT_RGB8p, nvcv::FMT_RGB8, 37, 23, 37, 23, numImages,
                                    [&gammaTensor](cudaStream_t stream, const nvcv::ImageBatchVarShape &src,
                                                   const nvcv::ImageBatchVarShape &dst, nvcv::ImageFormat)
                                    {
                                        cvcuda::GammaContrast op(numImages, channels);
                                        EXPECT_NO_THROW(op(stream, src, dst, gammaTensor));
                                    });
}

} // namespace

// Parameters: width, height, numImages, perChannelGamma, planarFmt, interleavedFmt
// clang-format off
NVCV_TEST_SUITE_P(OpGammaContrastPlanar,
                  test::ValueList<int, int, int, bool, nvcv::ImageFormat, nvcv::ImageFormat>{
    {176, 113, 2, false,    nvcv::FMT_RGB8p,    nvcv::FMT_RGB8}, // RGB8, per-image gamma
    {123,  66, 2,  true,    nvcv::FMT_RGB8p,    nvcv::FMT_RGB8}, // RGB8, per-channel gamma
    { 64,  48, 1,  true,   nvcv::FMT_RGBA8p,   nvcv::FMT_RGBA8}, // RGBA8, per-channel gamma
    { 41,  33, 2, false,          FMT_RGBU16p,          FMT_RGBU16}, // RGB16, per-image gamma
    { 50,  40, 2, false,  nvcv::FMT_RGBf32p,  nvcv::FMT_RGBf32}, // RGBf32, per-image gamma
    { 72,  54, 2,  true, nvcv::FMT_RGBAf32p, nvcv::FMT_RGBAf32}, // RGBAf32, per-channel gamma
    // F16 parity is bit-exact: both layouts run the same float chain and round to half once.
    { 50,  40, 2, false,  nvcv::FMT_RGBf16p,  nvcv::FMT_RGBf16}, // RGBf16, per-image gamma
    { 72,  54, 2,  true, nvcv::FMT_RGBAf16p, nvcv::FMT_RGBAf16}, // RGBAf16, per-channel gamma
});

// clang-format on

TEST_P(OpGammaContrastPlanar, varshape_matches_interleaved)
{
    RunGammaContrastPlanarParityCase(GetParamValue<4>(), GetParamValue<5>(), GetParamValue<0>(), GetParamValue<1>(),
                                     GetParamValue<2>(), GetParamValue<3>());
}

TEST(OpGammaContrastPlanar, padded_per_channel_gamma_matches_interleaved)
{
    RunGammaContrastPaddedPerChannelGammaParity();
}

TEST(OpGammaContrastPlanar, strided_per_image_gamma_matches_interleaved)
{
    constexpr int     numImages    = 2;
    constexpr int     channels     = 3;
    constexpr int64_t sampleStride = 2 * sizeof(float);
    constexpr size_t  bufferSize   = numImages * sampleStride;

    const std::array<float, numImages> gammaValues{0.6f, 1.4f};
    std::vector<uint8_t>               paddedGamma(bufferSize, 0);
    for (int image = 0; image < numImages; ++image)
    {
        std::memcpy(paddedGamma.data() + image * sampleStride, &gammaValues[image], sizeof(float));
    }

    NVCVByte *allocation{};
    ASSERT_EQ(cudaSuccess, cudaMalloc(reinterpret_cast<void **>(&allocation), bufferSize));
    if (allocation == nullptr)
    {
        return;
    }
    ASSERT_EQ(cudaSuccess, cudaMemcpy(allocation, paddedGamma.data(), bufferSize, cudaMemcpyHostToDevice));

    nvcv::TensorDataStridedCuda::Buffer buffer{};
    buffer.basePtr           = allocation;
    buffer.strides[0]        = sampleStride;
    buffer.strides[1]        = sizeof(float);
    buffer.strides[2]        = sizeof(float);
    nvcv::Tensor gammaTensor = nvcv::TensorWrapData(
        nvcv::TensorDataStridedCuda{
            nvcv::TensorShape{{numImages, 1, 1}, "NWC"},
            nvcv::TYPE_F32, buffer
    },
        nvcv::TensorDataCleanupCallback{[allocation](const nvcv::TensorData &)
                                        {
                                            cudaFree(allocation);
                                        }});

    test::planar::RunVarShapeParity(nvcv::FMT_RGB8p, nvcv::FMT_RGB8, 37, 23, 37, 23, numImages,
                                    [&gammaTensor](cudaStream_t stream, const nvcv::ImageBatchVarShape &src,
                                                   const nvcv::ImageBatchVarShape &dst, nvcv::ImageFormat)
                                    {
                                        cvcuda::GammaContrast op(numImages, channels);
                                        EXPECT_NO_THROW(op(stream, src, dst, gammaTensor));
                                    });
}

TEST(OpGammaContrastPlanar, rank2_unlabeled_strided_per_channel_gamma_matches_interleaved)
{
    constexpr int     numImages     = 2;
    constexpr int     channels      = 3;
    constexpr int64_t sampleStride  = 4 * sizeof(float);
    constexpr int64_t channelStride = sizeof(float);
    constexpr size_t  bufferSize    = (numImages - 1) * sampleStride + channels * channelStride;

    const std::array<std::array<float, channels>, numImages> gammaValues{
        std::array<float, channels>{0.6f, 0.9f, 1.2f},
         std::array<float, channels>{1.5f, 1.8f, 2.1f}
    };
    std::vector<uint8_t> paddedGamma(bufferSize, 0);
    for (int image = 0; image < numImages; ++image)
    {
        for (int channel = 0; channel < channels; ++channel)
        {
            std::memcpy(paddedGamma.data() + image * sampleStride + channel * channelStride,
                        &gammaValues[image][channel], sizeof(float));
        }
    }

    NVCVByte *allocation{};
    ASSERT_EQ(cudaSuccess, cudaMalloc(reinterpret_cast<void **>(&allocation), bufferSize));
    if (allocation == nullptr)
    {
        return;
    }
    ASSERT_EQ(cudaSuccess, cudaMemcpy(allocation, paddedGamma.data(), bufferSize, cudaMemcpyHostToDevice));

    nvcv::TensorDataStridedCuda::Buffer buffer{};
    buffer.basePtr           = allocation;
    buffer.strides[0]        = sampleStride;
    buffer.strides[1]        = channelStride;
    nvcv::Tensor gammaTensor = nvcv::TensorWrapData(
        nvcv::TensorDataStridedCuda{
            nvcv::TensorShape{{numImages, channels}, "HW"},
            nvcv::TYPE_F32, buffer
    },
        nvcv::TensorDataCleanupCallback{[allocation](const nvcv::TensorData &)
                                        {
                                            cudaFree(allocation);
                                        }});

    test::planar::RunTensorParity(
        nvcv::FMT_RGB8p, nvcv::FMT_RGB8, 37, 23, 37, 23, numImages,
        [&gammaTensor](cudaStream_t stream, const nvcv::Tensor &src, const nvcv::Tensor &dst, nvcv::ImageFormat)
        {
            cvcuda::GammaContrast op(numImages, channels);
            EXPECT_NO_THROW(op(stream, src, dst, gammaTensor));
        });
}

TEST(OpGammaContrastPlanar, u8_vector_tail_matches_interleaved)
{
    RunGammaContrastPlanarParityCase(nvcv::FMT_RGB8p, nvcv::FMT_RGB8, 65, 9, 1, true);
}

TEST(OpGammaContrastPlanar, f32_ragged_widths_matches_interleaved)
{
    constexpr int numImages = 2;
    constexpr int channels  = 3;

    nvcv::Tensor gammaTensor({{numImages * channels}, "N"}, nvcv::TYPE_F32);
    ASSERT_NO_FATAL_FAILURE(UploadGammaTensor(gammaTensor, std::vector<float>{0.6f, 0.9f, 1.2f, 1.5f, 1.8f, 2.1f}));

    const std::vector<nvcv::Size2D> sizes{
        {33, 9},
        {17, 5}
    };
    test::planar::RunVarShapeParity(nvcv::FMT_RGBf32p, nvcv::FMT_RGBf32, sizes, sizes,
                                    [&gammaTensor](cudaStream_t stream, const nvcv::ImageBatchVarShape &src,
                                                   const nvcv::ImageBatchVarShape &dst, nvcv::ImageFormat)
                                    {
                                        cvcuda::GammaContrast op(numImages, channels);
                                        EXPECT_NO_THROW(op(stream, src, dst, gammaTensor));
                                    });
}

// ---------------------------------------------------------------------------
// Tensor variant (interleaved + planar)
namespace {

// Tensor planar (NCHW/CHW) parity: planar tensor output must match interleaved tensor output bit-for-
// bit for the same data + gamma. Validates the planar tensor kernel against the interleaved one.
void RunGammaContrastTensorPlanarParityCase(nvcv::ImageFormat planarFmt, nvcv::ImageFormat interleavedFmt, int w, int h,
                                            int numSamples, bool perChannel)
{
    const int channels  = interleavedFmt.numChannels();
    const int nElements = perChannel ? numSamples * channels : numSamples;

    std::vector<float> gammaVec(nElements);
    for (int i = 0; i < nElements; ++i)
    {
        gammaVec[i] = DeterministicGammaValue(i, 0.5f, 2.0f, 47);
    }

    nvcv::Tensor gammaTensor({{nElements}, "N"}, nvcv::TYPE_F32);
    ASSERT_NO_FATAL_FAILURE(UploadGammaTensor(gammaTensor, gammaVec));

    test::planar::RunTensorParity(planarFmt, interleavedFmt, w, h, w, h, numSamples,
                                  [numSamples, channels, &gammaTensor](cudaStream_t s, const nvcv::Tensor &src,
                                                                       const nvcv::Tensor &dst, nvcv::ImageFormat)
                                  {
                                      cvcuda::GammaContrast op(numSamples, channels);
                                      EXPECT_NO_THROW(op(s, src, dst, gammaTensor));
                                  });
}

} // namespace

// width, height, numSamples, perChannelGamma, planarFmt, interleavedFmt
// clang-format off
NVCV_TEST_SUITE_P(OpGammaContrastTensorPlanar,
                  test::ValueList<int, int, int, bool, nvcv::ImageFormat, nvcv::ImageFormat>{
    {176, 113, 2, false,    nvcv::FMT_RGB8p,    nvcv::FMT_RGB8},
    {123,  66, 2,  true,    nvcv::FMT_RGB8p,    nvcv::FMT_RGB8},
    // These widths leave partial 16-byte carriers, exercising lookup tails in both layouts.
    {1921,  5, 2,  true,    nvcv::FMT_RGB8p,    nvcv::FMT_RGB8},
    {1913,  3, 1,  true,   nvcv::FMT_RGBA8p,   nvcv::FMT_RGBA8},
    {  17, 13, 2,  true,    nvcv::FMT_RGB8p,    nvcv::FMT_RGB8},
    { 64,  48, 1,  true,   nvcv::FMT_RGBA8p,   nvcv::FMT_RGBA8},
    { 41,  33, 2, false,          FMT_RGBU16p,          FMT_RGBU16},
    { 50,  40, 3, false,  nvcv::FMT_RGBf32p,  nvcv::FMT_RGBf32},
    { 72,  54, 2,  true, nvcv::FMT_RGBAf32p, nvcv::FMT_RGBAf32},
    // F16 parity is bit-exact: both layouts run the same float chain and round to half once.
    { 50,  40, 3, false,  nvcv::FMT_RGBf16p,  nvcv::FMT_RGBf16},
    { 72,  54, 2,  true, nvcv::FMT_RGBAf16p, nvcv::FMT_RGBAf16},
});

// clang-format on

TEST_P(OpGammaContrastTensorPlanar, tensor_matches_interleaved)
{
    RunGammaContrastTensorPlanarParityCase(GetParamValue<4>(), GetParamValue<5>(), GetParamValue<0>(),
                                           GetParamValue<1>(), GetParamValue<2>(), GetParamValue<3>());
}

TEST(OpGammaContrastTensorPlanar, signed16_single_channel_matches_interleaved)
{
    constexpr int samples = 2;
    constexpr int width   = 37;
    constexpr int height  = 19;

    nvcv::Tensor gammaTensor({{samples}, "N"}, nvcv::TYPE_F32);
    ASSERT_NO_FATAL_FAILURE(UploadGammaTensor(gammaTensor, std::vector<float>{1.0f, 2.0f}));

    test::planar::RunTensorSingleChannelLayoutParity(
        width, height, width, height, samples, nvcv::TYPE_S16,
        [&gammaTensor](cudaStream_t stream, const nvcv::Tensor &src, const nvcv::Tensor &dst)
        {
            cvcuda::GammaContrast op(samples, 1);
            EXPECT_NO_THROW(op(stream, src, dst, gammaTensor));
        });
}

// Tensor interleaved (NHWC) correctness against an independent CPU gold (the same gold the var-shape
// path is validated with), exercising the tensor kernel directly.
// width, height, numSamples, format, perChannelGamma
// clang-format off
NVCV_TEST_SUITE_P(OpGammaContrastTensor,
                  test::ValueList<int, int, int, NVCVImageFormat, bool>{
    {  9, 11, 2,    NVCV_IMAGE_FORMAT_U8,  true},
    { 12,  7, 3,  NVCV_IMAGE_FORMAT_RGB8,  true},
    // These widths exercise lookup tails and row boundaries.
    {1921,  5, 2,  NVCV_IMAGE_FORMAT_RGB8,  true},
    {4097,  3, 1,    NVCV_IMAGE_FORMAT_U8,  true},
    {  31,  7, 2, NVCV_IMAGE_FORMAT_RGBA8,  true},
    { 11, 11, 2, NVCV_IMAGE_FORMAT_RGBA8, false},
    {  7,  9, 2, NVCV_IMAGE_FORMAT_RGBA8,  true},
    {  8,  6, 2, NVCV_IMAGE_FORMAT_RGBf32, true},
    {  9,  7, 2, NVCV_IMAGE_FORMAT_RGBAf32, true},
    // F16 rows are validated against an FP32 gold within 8 half-ULPs (see the F16 gold branch).
    {  9, 11, 2,     NVCV_IMAGE_FORMAT_F16, false},
    {  8,  6, 2,  NVCV_IMAGE_FORMAT_RGBf16, true},
    {  9,  7, 2, NVCV_IMAGE_FORMAT_RGBAf16, true},
});

// clang-format on

TEST_P(OpGammaContrastTensor, correct_output)
{
    const int               width   = GetParamValue<0>();
    const int               height  = GetParamValue<1>();
    const int               samples = GetParamValue<2>();
    const nvcv::ImageFormat fmt{GetParamValue<3>()};
    const bool              perCh    = GetParamValue<4>();
    const int               channels = fmt.numChannels();

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    nvcv::Tensor src = nvcv::util::CreateTensor(samples, width, height, fmt);
    nvcv::Tensor dst = nvcv::util::CreateTensor(samples, width, height, fmt);

    TensorTestData testData;
    ASSERT_NO_FATAL_FAILURE(UploadDeterministicTensorInput(src, samples, width, height, fmt, testData));

    // Gamma tensor.
    const int          nElements = perCh ? samples * channels : samples;
    std::vector<float> gammaVec(nElements);
    for (int i = 0; i < nElements; ++i)
    {
        gammaVec[i] = DeterministicGammaValue(i, 0.4f, 2.0f, 61);
    }
    nvcv::Tensor gammaTensor({{nElements}, "N"}, nvcv::TYPE_F32);
    ASSERT_NO_FATAL_FAILURE(UploadGammaTensor(gammaTensor, gammaVec));

    cvcuda::GammaContrast op(samples, channels);
    EXPECT_NO_THROW(op(stream, src, dst, gammaTensor));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    for (int n = 0; n < samples; ++n)
    {
        std::vector<uint8_t> got;
        std::vector<uint8_t> gold = testData.hostSamples[n];
        ASSERT_NO_FATAL_FAILURE(DownloadTensorSample(dst, n, testData.rowStride, height, got));
        if (testData.isHalf)
        {
            // F16 gold: FP32 reference (pow, clamp to [0, 1]) on the widened half input, bounded
            // by 8 half-ULPs -- pow is a transcendental chain per the HalfTestUtils.hpp policy.
            std::vector<float> srcF = test::HalfBytesToFloat(testData.hostSamples[n]);
            std::vector<float> goldF(srcF.size());
            GammaContrastVarShapeCpuOp(goldF, testData.rowStride / static_cast<int>(sizeof(__half)), {width, height},
                                       srcF, testData.rowStride / static_cast<int>(sizeof(__half)), {width, height},
                                       fmt, gammaVec, n, perCh);
            test::ExpectNearHalfUlps(goldF, test::HalfBytesToFloat(got), 8.f);
            continue;
        }
        GammaContrastVarShapeCpuOpWrapper(gold, testData.rowStride, {width, height}, testData.hostSamples[n],
                                          testData.rowStride, {width, height}, fmt, gammaVec, n, perCh,
                                          testData.dataType);
        ExpectTensorNear(got, gold, testData.isFloat);
    }

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

// ---------------------------------------------------------------------------
// Scalar (host-float) gamma/gain tensor path
//
// The scalar overload applies out = gain * in**gamma with a single gamma/gain baked into the kernel
// launch (no gamma tensor, no host->device copy). Three properties are checked below:
//   * planar (NCHW/CHW) output matches interleaved output bit-for-bit (layout parity);
//   * with gain != 1 it matches an independent CPU gold that includes the gain term;
//   * with gain == 1 (an exact IEEE no-op) it is bit-exact with the device-tensor gamma path fed a
//     gamma tensor filled with the same value -- i.e. the already-gold-validated kernel, across every
//     supported dtype.
namespace {

// Independent CPU gold for the scalar path (single gamma/gain for all samples/channels).
void GammaContrastScalarCpuGold(std::vector<uint8_t> &hDst, const std::vector<uint8_t> &hSrc, float gamma, float gain,
                                bool isFloat, NVCVRoundMode roundMode)
{
    if (isFloat)
    {
        const auto  *src = reinterpret_cast<const float *>(hSrc.data());
        auto        *dst = reinterpret_cast<float *>(hDst.data());
        const size_t n   = hSrc.size() / sizeof(float);
        for (size_t i = 0; i < n; ++i)
        {
            dst[i] = nvcv::cuda::clamp(gain * std::pow(src[i], gamma), 0.f, 1.f);
        }
    }
    else
    {
        for (size_t i = 0; i < hSrc.size(); ++i)
        {
            const float tmp = (hSrc[i] + 0.0f) / 255.0f;
            float       out = gain * std::pow(tmp, gamma) * 255.0f;
            out             = nvcv::cuda::clamp(out, 0.0f, 255.0f);
            hDst[i]         = static_cast<uint8_t>(roundMode == NVCV_ROUND_TRUNCATE ? std::trunc(out) : std::rint(out));
        }
    }
}

void RunGammaContrastTensorScalarPlanarParityCase(nvcv::ImageFormat planarFmt, nvcv::ImageFormat interleavedFmt, int w,
                                                  int h, int numSamples, float gamma, float gain,
                                                  NVCVRoundMode roundMode)
{
    const int channels = interleavedFmt.numChannels();
    test::planar::RunTensorParity(
        planarFmt, interleavedFmt, w, h, w, h, numSamples,
        [numSamples, channels, gamma, gain, roundMode](cudaStream_t s, const nvcv::Tensor &src, const nvcv::Tensor &dst,
                                                       nvcv::ImageFormat)
        {
            cvcuda::GammaContrast op(numSamples, channels);
            EXPECT_NO_THROW(op(s, src, dst, gamma, gain, roundMode));
        });
}

} // namespace

// width, height, numSamples, gamma, gain, planarFmt, interleavedFmt, roundMode
// clang-format off
NVCV_TEST_SUITE_P(OpGammaContrastTensorScalarPlanar,
                  test::ValueList<int, int, int, float, float, nvcv::ImageFormat, nvcv::ImageFormat, NVCVRoundMode>{
    {176, 113, 2, 0.75f, 1.0f,    nvcv::FMT_RGB8p,    nvcv::FMT_RGB8,  NVCV_ROUND_NEAREST},
    {123,  66, 2, 1.50f, 0.9f,    nvcv::FMT_RGB8p,    nvcv::FMT_RGB8, NVCV_ROUND_TRUNCATE},
    { 64,  48, 1, 0.50f, 0.8f,   nvcv::FMT_RGBA8p,   nvcv::FMT_RGBA8,  NVCV_ROUND_NEAREST},
    { 41,  33, 2, 1.20f, 1.0f,        FMT_RGBU16p,        FMT_RGBU16,  NVCV_ROUND_NEAREST},
    { 50,  40, 3, 2.00f, 1.0f,  nvcv::FMT_RGBf32p,  nvcv::FMT_RGBf32, NVCV_ROUND_TRUNCATE},
    { 72,  54, 2, 0.70f, 1.2f, nvcv::FMT_RGBAf32p, nvcv::FMT_RGBAf32,  NVCV_ROUND_NEAREST},
    // F16 parity is bit-exact: both layouts run the same float chain and round to half once.
    { 50,  40, 2, 0.75f, 1.1f,  nvcv::FMT_RGBf16p,  nvcv::FMT_RGBf16,  NVCV_ROUND_NEAREST},
});

// clang-format on

TEST_P(OpGammaContrastTensorScalarPlanar, planar_matches_interleaved)
{
    RunGammaContrastTensorScalarPlanarParityCase(GetParamValue<5>(), GetParamValue<6>(), GetParamValue<0>(),
                                                 GetParamValue<1>(), GetParamValue<2>(), GetParamValue<3>(),
                                                 GetParamValue<4>(), GetParamValue<7>());
}

// Scalar path against an independent CPU gold that includes the gain term (interleaved).
// width, height, numSamples, format, gamma, gain, roundMode
// clang-format off
NVCV_TEST_SUITE_P(OpGammaContrastTensorScalar,
                  test::ValueList<int, int, int, NVCVImageFormat, float, float, NVCVRoundMode>{
    {  9, 11, 2,      NVCV_IMAGE_FORMAT_U8, 0.50f, 0.8f,  NVCV_ROUND_NEAREST},
    { 12,  7, 3,    NVCV_IMAGE_FORMAT_RGB8, 1.50f, 0.9f, NVCV_ROUND_TRUNCATE},
    { 11, 11, 2,   NVCV_IMAGE_FORMAT_RGBA8, 0.80f, 1.0f,  NVCV_ROUND_NEAREST},
    {  8,  6, 2,  NVCV_IMAGE_FORMAT_RGBf32, 2.00f, 1.0f, NVCV_ROUND_TRUNCATE},
    { 10,  5, 2, NVCV_IMAGE_FORMAT_RGBAf32, 0.70f, 1.2f,  NVCV_ROUND_NEAREST},
    // F16 rows are validated against an FP32 gold within 8 half-ULPs; round mode does not apply
    // to the float path.
    {  8,  6, 2,  NVCV_IMAGE_FORMAT_RGBf16, 0.75f, 1.1f,  NVCV_ROUND_NEAREST},
});

// clang-format on

TEST_P(OpGammaContrastTensorScalar, matches_cpu_gold)
{
    const int               width   = GetParamValue<0>();
    const int               height  = GetParamValue<1>();
    const int               samples = GetParamValue<2>();
    const nvcv::ImageFormat fmt{GetParamValue<3>()};
    const float             gamma     = GetParamValue<4>();
    const float             gain      = GetParamValue<5>();
    const NVCVRoundMode     roundMode = GetParamValue<6>();
    const int               channels  = fmt.numChannels();

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    nvcv::Tensor src = nvcv::util::CreateTensor(samples, width, height, fmt);
    nvcv::Tensor dst = nvcv::util::CreateTensor(samples, width, height, fmt);

    TensorTestData testData;
    ASSERT_NO_FATAL_FAILURE(UploadDeterministicTensorInput(src, samples, width, height, fmt, testData));

    cvcuda::GammaContrast op(samples, channels);
    EXPECT_NO_THROW(op(stream, src, dst, gamma, gain, roundMode));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    for (int n = 0; n < samples; ++n)
    {
        std::vector<uint8_t> got;
        std::vector<uint8_t> gold(testData.sampleBytes);
        ASSERT_NO_FATAL_FAILURE(DownloadTensorSample(dst, n, testData.rowStride, height, got));
        if (testData.isHalf)
        {
            // F16 gold: FP32 reference (gain * in**gamma, clamp to [0, 1]) on the widened half
            // input, bounded by 8 half-ULPs -- pow is a transcendental chain per the
            // HalfTestUtils.hpp policy. Round mode does not apply to the float path.
            std::vector<float> srcF = test::HalfBytesToFloat(testData.hostSamples[n]);
            std::vector<float> goldF(srcF.size());
            for (size_t i = 0; i < srcF.size(); ++i)
            {
                goldF[i] = nvcv::cuda::clamp(gain * std::pow(srcF[i], gamma), 0.f, 1.f);
            }
            test::ExpectNearHalfUlps(goldF, test::HalfBytesToFloat(got), 8.f);
            continue;
        }
        GammaContrastScalarCpuGold(gold, testData.hostSamples[n], gamma, gain, testData.isFloat, roundMode);
        ExpectTensorNear(got, gold, testData.isFloat);
    }

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpGammaContrastTensorScalar, round_mode_controls_integer_conversion)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    nvcv::Tensor src      = nvcv::util::CreateTensor(1, 1, 1, nvcv::FMT_U8);
    nvcv::Tensor nearest  = nvcv::util::CreateTensor(1, 1, 1, nvcv::FMT_U8);
    nvcv::Tensor truncate = nvcv::util::CreateTensor(1, 1, 1, nvcv::FMT_U8);

    auto srcData      = src.exportData<nvcv::TensorDataStridedCuda>();
    auto nearestData  = nearest.exportData<nvcv::TensorDataStridedCuda>();
    auto truncateData = truncate.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_TRUE(srcData && nearestData && truncateData);

    constexpr uint8_t input = 255;
    ASSERT_EQ(cudaSuccess, cudaMemcpy(srcData->basePtr(), &input, sizeof(input), cudaMemcpyHostToDevice));

    cvcuda::GammaContrast op(1, 1);
    constexpr float       gain = 1.5f / 255.0f;
    EXPECT_NO_THROW(op(stream, src, nearest, 1.0f, gain, NVCV_ROUND_NEAREST));
    EXPECT_NO_THROW(op(stream, src, truncate, 1.0f, gain, NVCV_ROUND_TRUNCATE));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    uint8_t nearestValue  = 0;
    uint8_t truncateValue = 0;
    ASSERT_EQ(cudaSuccess,
              cudaMemcpy(&nearestValue, nearestData->basePtr(), sizeof(nearestValue), cudaMemcpyDeviceToHost));
    ASSERT_EQ(cudaSuccess,
              cudaMemcpy(&truncateValue, truncateData->basePtr(), sizeof(truncateValue), cudaMemcpyDeviceToHost));
    EXPECT_EQ(nearestValue, 2);
    EXPECT_EQ(truncateValue, 1);

    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcv::ProtectCall([&] { op(stream, src, nearest, 1.0f, gain, static_cast<NVCVRoundMode>(-1)); }));

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

// Scalar path with gain == 1 must be bit-exact with the device-tensor gamma path fed a gamma tensor
// filled with the same value (that path is itself validated against the CPU gold above). Exercises the
// full supported dtype set through the scalar kernels.
// width, height, numSamples, format, gamma
// clang-format off
NVCV_TEST_SUITE_P(OpGammaContrastScalarVsTensor,
                  test::ValueList<int, int, int, NVCVImageFormat, float>{
    {  9, 11, 2,      NVCV_IMAGE_FORMAT_U8, 0.60f},
    { 12,  7, 3,    NVCV_IMAGE_FORMAT_RGB8, 1.40f},
    // These widths exercise lookup tails through the scalar-gamma entry point.
    {1921,  5, 2,    NVCV_IMAGE_FORMAT_RGB8, 0.70f},
    {1913,  3, 1,   NVCV_IMAGE_FORMAT_RGBA8, 1.30f},
    {4097,  2, 2,      NVCV_IMAGE_FORMAT_U8, 0.85f},
    { 11, 11, 2,   NVCV_IMAGE_FORMAT_RGBA8, 0.80f},
    { 10,  9, 2,      NVCV_IMAGE_FORMAT_U16, 1.20f},
    {  7,  8, 2,      NVCV_IMAGE_FORMAT_S16, 0.90f},
    {  8,  6, 2,      NVCV_IMAGE_FORMAT_S32, 1.10f},
    {  8,  6, 2,  NVCV_IMAGE_FORMAT_RGBf32, 0.70f},
    { 10,  5, 2, NVCV_IMAGE_FORMAT_RGBAf32, 1.30f},
    {  8,  6, 2,  NVCV_IMAGE_FORMAT_RGBf16, 0.70f},
});

// clang-format on

TEST_P(OpGammaContrastScalarVsTensor, bit_exact_gain1)
{
    const int               width   = GetParamValue<0>();
    const int               height  = GetParamValue<1>();
    const int               samples = GetParamValue<2>();
    const nvcv::ImageFormat fmt{GetParamValue<3>()};
    const float             gamma    = GetParamValue<4>();
    const int               channels = fmt.numChannels();

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    nvcv::Tensor src       = nvcv::util::CreateTensor(samples, width, height, fmt);
    nvcv::Tensor dstScalar = nvcv::util::CreateTensor(samples, width, height, fmt);
    nvcv::Tensor dstTensor = nvcv::util::CreateTensor(samples, width, height, fmt);

    TensorTestData testData;
    ASSERT_NO_FATAL_FAILURE(UploadDeterministicTensorInput(src, samples, width, height, fmt, testData));

    // Per-image gamma tensor filled with the same value -> every sample/channel uses `gamma`, matching
    // the scalar broadcast; gain == 1 is an exact IEEE no-op, so the two outputs must be identical.
    std::vector<float> gammaVec(samples, gamma);
    nvcv::Tensor       gammaTensor({{samples}, "N"}, nvcv::TYPE_F32);
    ASSERT_NO_FATAL_FAILURE(UploadGammaTensor(gammaTensor, gammaVec));

    cvcuda::GammaContrast op(samples, channels);
    EXPECT_NO_THROW(op(stream, src, dstScalar, gamma, 1.0f));
    EXPECT_NO_THROW(op(stream, src, dstTensor, gammaTensor));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    auto compareSample = [&dstScalar, &dstTensor, &testData, height](int n)
    {
        std::vector<uint8_t> scalarOut;
        std::vector<uint8_t> tensorOut;
        ASSERT_NO_FATAL_FAILURE(DownloadTensorSample(dstScalar, n, testData.rowStride, height, scalarOut));
        ASSERT_NO_FATAL_FAILURE(DownloadTensorSample(dstTensor, n, testData.rowStride, height, tensorOut));
        EXPECT_EQ(scalarOut, tensorOut) << "scalar (gain=1) != device-tensor gamma path at sample " << n;
    };

    for (int n = 0; n < samples; ++n)
    {
        ASSERT_NO_FATAL_FAILURE(compareSample(n));
    }

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

// Mismatched stream partitions once compacted output; CPU gold prevents a shared layout fault from masking it.
TEST(OpGammaContrastTensorScalarPlanar, packed_source_with_padded_destination_matches_gold)
{
    constexpr int   kSamples     = 2;
    constexpr int   kChannels    = 3;
    constexpr int   kHeight      = 16;
    constexpr int   kWidth       = 23;
    constexpr int   kSrcRowPitch = kWidth;
    constexpr int   kDstRowPitch = 32; // Matches allocator padding for a 23-byte row.
    constexpr float kGamma       = 0.5f;
    constexpr float kGain        = 1.0f;

    const int64_t srcBytes = int64_t{kSamples} * kChannels * kHeight * kSrcRowPitch;
    const int64_t dstBytes = int64_t{kSamples} * kChannels * kHeight * kDstRowPitch;

    std::vector<uint8_t> hostSrc(static_cast<size_t>(srcBytes));
    for (size_t i = 0; i < hostSrc.size(); ++i) hostSrc[i] = static_cast<uint8_t>((i * 37 + 11) % 256);

    NVCVByte *srcAlloc{};
    NVCVByte *dstAlloc{};
    ASSERT_EQ(cudaSuccess, cudaMalloc(reinterpret_cast<void **>(&srcAlloc), srcBytes));
    ASSERT_EQ(cudaSuccess, cudaMalloc(reinterpret_cast<void **>(&dstAlloc), dstBytes));
    ASSERT_EQ(cudaSuccess, cudaMemset(dstAlloc, 0, dstBytes));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(srcAlloc, hostSrc.data(), srcBytes, cudaMemcpyHostToDevice));

    nvcv::TensorDataStridedCuda::Buffer srcBuffer{};
    srcBuffer.basePtr    = srcAlloc;
    srcBuffer.strides[3] = 1;
    srcBuffer.strides[2] = kSrcRowPitch;
    srcBuffer.strides[1] = int64_t{kHeight} * kSrcRowPitch;
    srcBuffer.strides[0] = int64_t{kChannels} * kHeight * kSrcRowPitch;

    nvcv::TensorDataStridedCuda::Buffer dstBuffer{};
    dstBuffer.basePtr    = dstAlloc;
    dstBuffer.strides[3] = 1;
    dstBuffer.strides[2] = kDstRowPitch;
    dstBuffer.strides[1] = int64_t{kHeight} * kDstRowPitch;
    dstBuffer.strides[0] = int64_t{kChannels} * kHeight * kDstRowPitch;

    const nvcv::TensorShape shape{
        {kSamples, kChannels, kHeight, kWidth},
        "NCHW"
    };
    nvcv::Tensor src = nvcv::TensorWrapData(nvcv::TensorDataStridedCuda{shape, nvcv::TYPE_U8, srcBuffer});
    nvcv::Tensor dst = nvcv::TensorWrapData(nvcv::TensorDataStridedCuda{shape, nvcv::TYPE_U8, dstBuffer});

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));
    cvcuda::GammaContrast op(kSamples, kChannels);
    EXPECT_NO_THROW(op(stream, src, dst, kGamma, kGain, NVCV_ROUND_NEAREST));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    std::vector<uint8_t> hostDst(static_cast<size_t>(dstBytes));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(hostDst.data(), dstAlloc, dstBytes, cudaMemcpyDeviceToHost));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
    ASSERT_EQ(cudaSuccess, cudaFree(srcAlloc));
    ASSERT_EQ(cudaSuccess, cudaFree(dstAlloc));

    int mismatches = 0;
    for (int row = 0; row < kSamples * kChannels * kHeight; ++row)
    {
        for (int x = 0; x < kWidth; ++x)
        {
            const uint8_t in   = hostSrc[row * kSrcRowPitch + x];
            const float   v    = std::pow(in / 255.0f, kGamma) * kGain * 255.0f;
            const int     gold = std::min(255, std::max(0, static_cast<int>(std::lround(v))));
            const uint8_t got  = hostDst[row * kDstRowPitch + x];
            if (static_cast<int>(got) == gold)
            {
                continue;
            }
            ++mismatches;
            if (mismatches <= 4)
            {
                const int n = row / (kChannels * kHeight);
                const int c = (row / kHeight) % kChannels;
                const int y = row % kHeight;
                ADD_FAILURE() << "n=" << n << " c=" << c << " y=" << y << " x=" << x << " got=" << static_cast<int>(got)
                              << " gold=" << gold;
            }
        }
    }
    EXPECT_EQ(0, mismatches);
}

TEST(OpGammaContrastTensor, strided_gamma_shape_fallbacks)
{
    static constexpr int samples       = 2;
    static constexpr int channels      = 3;
    constexpr int        width         = 13;
    constexpr int        height        = 7;
    constexpr int64_t    elementStride = 2 * sizeof(float);

    auto makeStridedGamma
        = [](const nvcv::TensorShape &shape, const std::vector<int64_t> &strides, const std::vector<float> &storage)
    {
        NVCVByte *allocation{};
        EXPECT_EQ(cudaSuccess, cudaMalloc(reinterpret_cast<void **>(&allocation), storage.size() * sizeof(float)));
        if (allocation == nullptr)
        {
            return nvcv::Tensor{};
        }

        EXPECT_EQ(cudaSuccess,
                  cudaMemcpy(allocation, storage.data(), storage.size() * sizeof(float), cudaMemcpyHostToDevice));

        nvcv::TensorDataStridedCuda::Buffer buffer{};
        buffer.basePtr = allocation;
        for (int dim = 0; dim < shape.rank(); ++dim)
        {
            buffer.strides[dim] = strides[dim];
        }

        return nvcv::TensorWrapData(nvcv::TensorDataStridedCuda{shape, nvcv::TYPE_F32, buffer},
                                    nvcv::TensorDataCleanupCallback{[allocation](const nvcv::TensorData &)
                                                                    {
                                                                        EXPECT_EQ(cudaSuccess, cudaFree(allocation));
                                                                    }});
    };

    auto runParity = [&](nvcv::Tensor &gammaTensor, const char *caseName)
    {
        SCOPED_TRACE(caseName);
        test::planar::RunTensorParity(
            nvcv::FMT_RGB8p, nvcv::FMT_RGB8, width, height, width, height, samples,
            [&gammaTensor](cudaStream_t stream, const nvcv::Tensor &src, const nvcv::Tensor &dst, nvcv::ImageFormat)
            {
                cvcuda::GammaContrast op(samples, channels);
                EXPECT_NO_THROW(op(stream, src, dst, gammaTensor));
            });
    };

    const std::vector<float> perImageValues{0.6f, 1.4f};
    nvcv::Tensor             perImageRank1({{samples}, "N"}, nvcv::TYPE_F32);
    ASSERT_NO_FATAL_FAILURE(UploadGammaTensor(perImageRank1, perImageValues));
    runParity(perImageRank1, "rank1 per-image gamma");

    const std::vector<float> perImageStorage{0.6f, 0.f, 1.4f, 0.f};
    nvcv::Tensor             perImageRank2 = makeStridedGamma(
                    nvcv::TensorShape{
            {samples, 1},
            "XY"
    },
                    {elementStride, sizeof(float)}, perImageStorage);
    runParity(perImageRank2, "rank2 per-image gamma");

    const std::array<float, samples * channels> perChannelValues{0.6f, 0.9f, 1.2f, 1.5f, 1.8f, 2.1f};
    std::vector<float>                          perChannelStorage(perChannelValues.size() * 2, 0.f);
    for (size_t i = 0; i < perChannelValues.size(); ++i)
    {
        perChannelStorage[2 * i] = perChannelValues[i];
    }
    const std::vector<float> perChannelDense(perChannelValues.begin(), perChannelValues.end());
    nvcv::Tensor             perChannelRank1({{samples * channels}, "N"}, nvcv::TYPE_F32);
    ASSERT_NO_FATAL_FAILURE(UploadGammaTensor(perChannelRank1, perChannelDense));
    runParity(perChannelRank1, "rank1 per-channel gamma");

    nvcv::Tensor invalidPerChannel = makeStridedGamma(
        nvcv::TensorShape{
            {samples * channels, 1},
            "XY"
    },
        {elementStride, sizeof(float)}, perChannelStorage);

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));
    nvcv::Tensor          src = nvcv::util::CreateTensor(samples, width, height, nvcv::FMT_RGB8);
    nvcv::Tensor          dst = nvcv::util::CreateTensor(samples, width, height, nvcv::FMT_RGB8);
    cvcuda::GammaContrast op(samples, channels);
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcv::ProtectCall([&] { op(stream, src, dst, invalidPerChannel); }));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}
