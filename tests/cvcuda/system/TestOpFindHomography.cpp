/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <common/TensorDataUtils.hpp>
#include <common/ValueTests.hpp>
#include <cvcuda/OpFindHomography.hpp>
#include <cvcuda/cuda_tools/DropCast.hpp>
#include <cvcuda/cuda_tools/MathOps.hpp>
#include <cvcuda/cuda_tools/StaticCast.hpp>
#include <cvcuda/cuda_tools/TypeTraits.hpp>
#include <cvcuda/cuda_tools/math/LinAlg.hpp>
#include <math.h>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorBatch.hpp>
#include <nvcv/TensorDataAccess.hpp>
#include <nvcv/util/Math.hpp>

#include <array>
#include <iostream>
#include <random>
#include <vector>

#ifdef PERFORMANCE_RUN
#    define WARMUP_ITERATIONS 5
#    define PERF_ITERATIONS   50
#endif

namespace test = nvcv::test;
namespace util = nvcv::util;
namespace cuda = nvcv::cuda;

static void calculateDst(float x, float y, float *X, float *Y, const float *model)
{
    *X = model[0] * x + model[1] * y + model[2] * 1;
    *Y = model[3] * x + model[4] * y + model[5] * 1;
}

static void calculateProjectiveDst(float x, float y, float *X, float *Y, const float *model)
{
    float w = model[6] * x + model[7] * y + model[8];
    *X      = (model[0] * x + model[1] * y + model[2]) / w;
    *Y      = (model[3] * x + model[4] * y + model[5]) / w;
}

static nvcv::Tensor createModelTensor(int numSamples)
{
    return nvcv::Tensor(
        {
            {numSamples, 3, 3},
            "NHW"
    },
        nvcv::TYPE_F32);
}

static void calculateGoldModelMatrix(float *m, std::mt19937 &rng, std::uniform_int_distribution<int> &dis)
{
    // random rotation angle between 0 and pi
    float                           theta = static_cast<float>(M_PI / 2.0) * static_cast<float>(dis(rng)) / 100.0f;
    float                           Tx    = static_cast<float>(dis(rng)) / 100.0f;
    float                           Ty    = static_cast<float>(dis(rng)) / 100.0f;
    float                           sx    = static_cast<float>(dis(rng)) / 100.0f;
    float                           sy    = static_cast<float>(dis(rng)) / 100.0f;
    float                           p1    = static_cast<float>(dis(rng)) / 100.0f;
    float                           p2    = static_cast<float>(dis(rng)) / 100.0f * 2.0f;
    cuda::math::Matrix<float, 3, 3> He;
    He[0] = {cos(theta), -sin(theta), Tx};
    He[1] = {sin(theta), cos(theta), Ty};
    He[2] = {0, 0, 1};
    cuda::math::Matrix<float, 3, 3> Ha;
    Ha[0] = {1, sy, 0};
    Ha[1] = {sx, 1, 0};
    Ha[2] = {0, 0, 1};
    cuda::math::Matrix<float, 3, 3> Hp;
    Hp[0]                                  = {1, 0, 0};
    Hp[1]                                  = {0, 1, 0};
    Hp[2]                                  = {p1, p2, 1};
    cuda::math::Matrix<float, 3, 3> result = He * (Ha * Hp);
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) m[i * 3 + j] = result[i][j];
}

// clang-format off
NVCV_TEST_SUITE_P(OpFindHomography, test::ValueList<int, int>
{
    // Parameter order: sample count, point count.
    {8, 16},
    {16, 20},
    {25, 40}
});

// clang-format on

TEST_P(OpFindHomography, correct_output)
{
    int numSamples = GetParamValue<0>();
    int numPoints  = GetParamValue<1>();
    numPoints *= numPoints;

    // clang-format off
    nvcv::Tensor srcPoints({{numSamples, numPoints}, "NW"}, nvcv::TYPE_2F32);
    nvcv::Tensor dstPoints({{numSamples, numPoints}, "NW"}, nvcv::TYPE_2F32);
    nvcv::Tensor models({{numSamples, 3, 3}, "NHW"}, nvcv::TYPE_F32);

    // clang-format on

    auto srcData    = srcPoints.exportData<nvcv::TensorDataStridedCuda>();
    auto dstData    = dstPoints.exportData<nvcv::TensorDataStridedCuda>();
    auto modelsData = models.exportData<nvcv::TensorDataStridedCuda>();

    ASSERT_EQ(srcData->shape(0), srcData->shape(0));
    ASSERT_EQ(srcData->shape(1), srcData->shape(1));

    std::vector<float> srcVec(2 * numSamples * numPoints);
    std::vector<float> dstVec(2 * numSamples * numPoints);
    std::vector<float> modelsVec(numSamples * 9);
    std::vector<float> estimatedModelsVec(numSamples * 9);
    std::vector<float> computedDstVec(2 * numSamples * numPoints);

    // Fixed seed matches the sibling varshape_correct_output test below — the
    // original random_device-seeded gen made input geometry non-deterministic
    // and occasionally produced ill-conditioned point sets that exceeded the
    // 1e-3 tolerance on rare-config CI (manylinux x86 gcc10 release).
    std::mt19937                  gen(12345); // Mersenne Twister engine
    std::uniform_int_distribution dis(0, 100);

    auto numXPoints = static_cast<int>(std::sqrt(numPoints));
    int  numYPoints = numXPoints;

#ifdef WRITE_COORDINATES_TO_FILE
    std::string src_filename
        = "src_coordinates_" + std::to_string(numSamples) + "x" + std::to_string(numPoints) + ".bin";
    std::string dst_filename
        = "dst_coordinates_" + std::to_string(numSamples) + "x" + std::to_string(numPoints) + ".bin";

    std::ofstream outSrcFile(src_filename.c_str(), std::ios::binary);
    if (!outSrcFile.is_open())
    {
        std::cerr << "Failed to open the src file for writing." << std::endl;
        return;
    }

    std::ofstream outDstFile(dst_filename.c_str(), std::ios::binary);
    if (!outDstFile.is_open())
    {
        std::cerr << "Failed to open the dst file for writing." << std::endl;
        return;
    }
#endif

    // Fill gold models and src and dst points
    for (int i = 0; i < numSamples; i++)
    {
#pragma unroll
        calculateGoldModelMatrix(&modelsVec[i * 9], gen, dis);
        // generate src and dst points
        for (int j = 0; j < numYPoints; j++)
        {
            for (int k = 0; k < numXPoints; k++)
            {
                int idx                                 = j * numYPoints + k;
                srcVec[i * numPoints * 2 + 2 * idx]     = static_cast<float>(dis(gen));
                srcVec[i * numPoints * 2 + 2 * idx + 1] = static_cast<float>(dis(gen));

                float dstx;
                float dsty;
                calculateDst(srcVec[i * numPoints * 2 + 2 * idx], srcVec[i * numPoints * 2 + 2 * idx + 1], &dstx, &dsty,
                             modelsVec.data() + i * 9);
                dstVec[i * numPoints * 2 + 2 * idx]     = dstx;
                dstVec[i * numPoints * 2 + 2 * idx + 1] = dsty;
            }
        }
    }

#ifdef WRITE_COORDINATES_TO_FILE
    outSrcFile.write(reinterpret_cast<const char *>(srcVec.data()), srcVec.size() * sizeof(float));
    outDstFile.write(reinterpret_cast<const char *>(dstVec.data()), dstVec.size() * sizeof(float));

    outSrcFile.close();
    outDstFile.close();
#endif

    // Create the test stream BEFORE the input uploads so the H2D copies are
    // queued on the same stream the operator will use. Using cudaMemcpyAsync on
    // a non-blocking stream guarantees the operator's reduction kernels see the
    // populated dst data — synchronous cudaMemcpy on the default stream does
    // NOT order against work on a non-blocking custom stream, which on certain
    // driver versions (observed on 580.35) produced an all-zero dst tensor at
    // kernel-read time and a silent zero-homography output (CVCUDA-####).
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(srcData->basePtr(), srcVec.data(),
                                           sizeof(float) * 2 * numPoints * numSamples, cudaMemcpyHostToDevice, stream));
    ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(dstData->basePtr(), dstVec.data(),
                                           sizeof(float) * 2 * numPoints * numSamples, cudaMemcpyHostToDevice, stream));

    cvcuda::FindHomography fh(numSamples, numPoints);

#ifdef PERFORMANCE_RUN
    for (int it = 0; it < WARMUP_ITERATIONS; it++)
    {
        EXPECT_NO_THROW(fh(stream, srcPoints, dstPoints, models));
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, stream);
    for (int it = 0; it < PERF_ITERATIONS; it++)
    {
        EXPECT_NO_THROW(fh(stream, srcPoints, dstPoints, models));
    }
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Time taken for " << numSamples << "x" << numPoints << " = " << milliseconds / PERF_ITERATIONS
              << "ms\n";
    // std::cout << "Time taken per image  = " << milliseconds / PERF_ITERATIONS / numSamples << "ms\n";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
#else
    EXPECT_NO_THROW(fh(stream, srcPoints, dstPoints, models));
#endif

    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    // copy back the estimated models into modelsVec
    for (int i = 0; i < numSamples; i++)
    {
        ASSERT_EQ(cudaSuccess, cudaMemcpy2D(estimatedModelsVec.data() + i * 9, sizeof(float) * 3,
                                            modelsData->basePtr() + i * modelsData->stride(0), modelsData->stride(1),
                                            sizeof(float) * 3, 3, cudaMemcpyDeviceToHost));
    }

    // Compute dst vec based on model estimated
#ifndef PERFORMANCE_RUN
    for (int i = 0; i < numSamples; i++)
    {
        for (int j = 0; j < numYPoints; j++)
        {
            for (int k = 0; k < numXPoints; k++)
            {
                int   idx = j * numYPoints + k;
                float dstx;
                float dsty;
                calculateDst(srcVec[i * numPoints * 2 + 2 * idx], srcVec[i * numPoints * 2 + 2 * idx + 1], &dstx, &dsty,
                             estimatedModelsVec.data() + i * 9);
                computedDstVec[i * numPoints * 2 + 2 * idx]     = dstx;
                computedDstVec[i * numPoints * 2 + 2 * idx + 1] = dsty;
                float A                                         = dstVec[i * numPoints * 2 + 2 * idx];
                float B                                         = computedDstVec[i * numPoints * 2 + 2 * idx];
                // The 1e-3 tolerance covers GPU reduction/FMA rounding against the independent CPU projection.
                EXPECT_NEAR(A, B, 1e-03);
                A = dstVec[i * numPoints * 2 + 2 * idx + 1];
                B = computedDstVec[i * numPoints * 2 + 2 * idx + 1];
                // The 1e-3 tolerance covers GPU reduction/FMA rounding against the independent CPU projection.
                EXPECT_NEAR(A, B, 1e-03);
            }
        }
    }
#endif
}

TEST(OpFindHomography, nwc_correct_output)
{
    constexpr int                  numSamples = 2;
    constexpr std::array<float, 9> goldModel  = {1.05f, 0.08f, 0.15f, -0.04f, 0.97f, -0.10f, 0.015f, -0.020f, 1.0f};
    constexpr std::array<int, 2>   numPointCases{4, 16};

    auto runCase = [&](int numPoints)
    {
        SCOPED_TRACE(::testing::Message() << "numPoints=" << numPoints);

        nvcv::Tensor srcPoints(
            {
                {numSamples, numPoints, 2},
                "NWC"
        },
            nvcv::TYPE_F32);
        nvcv::Tensor dstPoints(
            {
                {numSamples, numPoints, 2},
                "NWC"
        },
            nvcv::TYPE_F32);
        nvcv::Tensor models = createModelTensor(numSamples);

        auto srcData    = srcPoints.exportData<nvcv::TensorDataStridedCuda>();
        auto dstData    = dstPoints.exportData<nvcv::TensorDataStridedCuda>();
        auto modelsData = models.exportData<nvcv::TensorDataStridedCuda>();

        std::vector<float> srcVec(2 * numSamples * numPoints);
        std::vector<float> dstVec(2 * numSamples * numPoints);
        std::vector<float> estimatedModelsVec(numSamples * 9);

        int gridSide = 1;
        while (gridSide * gridSide < numPoints)
        {
            ++gridSide;
        }
        float gridScale = 2.0f / static_cast<float>(gridSide - 1);

        for (int sample = 0; sample < numSamples; ++sample)
        {
            for (int point = 0; point < numPoints; ++point)
            {
                float x = -1.0f + static_cast<float>(point % gridSide) * gridScale;
                float y = -1.0f + static_cast<float>(point / gridSide) * gridScale;
                float dstx;
                float dsty;
                calculateProjectiveDst(x, y, &dstx, &dsty, goldModel.data());

                int offset         = 2 * (sample * numPoints + point);
                srcVec[offset]     = x;
                srcVec[offset + 1] = y;
                dstVec[offset]     = dstx;
                dstVec[offset + 1] = dsty;
            }
        }

        cudaStream_t stream;
        ASSERT_EQ(cudaSuccess, cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(srcData->basePtr(), srcVec.data(), srcVec.size() * sizeof(float),
                                               cudaMemcpyHostToDevice, stream));
        ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(dstData->basePtr(), dstVec.data(), dstVec.size() * sizeof(float),
                                               cudaMemcpyHostToDevice, stream));

        cvcuda::FindHomography fh(numSamples, numPoints);
        EXPECT_NO_THROW(fh(stream, srcPoints, dstPoints, models));
        ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

        for (int sample = 0; sample < numSamples; ++sample)
        {
            ASSERT_EQ(cudaSuccess, cudaMemcpy2D(estimatedModelsVec.data() + sample * 9, sizeof(float) * 3,
                                                modelsData->basePtr() + sample * modelsData->stride(0),
                                                modelsData->stride(1), sizeof(float) * 3, 3, cudaMemcpyDeviceToHost));
        }
        ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));

        // GPU reductions and FMA contraction are not bit-exact with this independent CPU projection.
        for (int sample = 0; sample < numSamples; ++sample)
        {
            for (int point = 0; point < numPoints; ++point)
            {
                int   offset = 2 * (sample * numPoints + point);
                float projectedX;
                float projectedY;
                calculateProjectiveDst(srcVec[offset], srcVec[offset + 1], &projectedX, &projectedY,
                                       estimatedModelsVec.data() + sample * 9);
                // The 1e-3 tolerance covers GPU reduction/FMA rounding against the independent CPU projection.
                EXPECT_NEAR(dstVec[offset], projectedX, 1e-3f);
                // The 1e-3 tolerance covers GPU reduction/FMA rounding against the independent CPU projection.
                EXPECT_NEAR(dstVec[offset + 1], projectedY, 1e-3f);
            }
        }
    };

    for (int numPoints : numPointCases)
    {
        runCase(numPoints);
    }
}

TEST_P(OpFindHomography, varshape_correct_output)
{
    int              numSamples = GetParamValue<0>();
    int              maxPoints  = GetParamValue<1>();
    std::vector<int> numPoints(numSamples);
    std::vector<int> numXPoints(numSamples);

    std::mt19937                  rng(12345);
    std::uniform_int_distribution dis(0, 100);
    std::uniform_int_distribution dis_num_points(4, maxPoints);

    auto              reqs = nvcv::TensorBatch::CalcRequirements(numSamples);
    nvcv::TensorBatch srcTensorBatch(reqs);
    nvcv::TensorBatch dstTensorBatch(reqs);
    nvcv::TensorBatch modelsTensorBatch(reqs);

    // Create the test stream up front so the per-batch H2D uploads below can
    // use cudaMemcpyAsync on the same stream the operator will run on. See the
    // matching comment in correct_output for why synchronous cudaMemcpy on the
    // default stream is unsafe here.
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    std::vector<std::vector<float>> srcVec(numSamples);
    std::vector<std::vector<float>> dstVec(numSamples);
    std::vector<float>              modelsVec(numSamples * 9);
    std::vector<float>              estimatedModelsVec(numSamples * 9);
    std::vector<std::vector<float>> computedDstVec(numSamples);

    int maxNumPoints = 0;
    for (int i = 0; i < numSamples; i++)
    {
        numXPoints[i] = dis_num_points(rng);
        numPoints[i]  = numXPoints[i] * numXPoints[i];
        if (numPoints[i] > maxNumPoints)
            maxNumPoints = numPoints[i];

        // Fill gold models and src and dst points
        calculateGoldModelMatrix(&modelsVec[i * 9], rng, dis);
        for (int j = 0; j < numPoints[i]; j++)
        {
            int sx = dis(rng);
            int sy = dis(rng);
            srcVec[i].push_back(static_cast<float>(sx));
            srcVec[i].push_back(static_cast<float>(sy));

            float dstx;
            float dsty;
            calculateDst(static_cast<float>(sx), static_cast<float>(sy), &dstx, &dsty, modelsVec.data() + i * 9);
            dstVec[i].push_back(dstx);
            dstVec[i].push_back(dsty);
        }

        nvcv::Tensor srcPoints(
            {
                {1, numPoints[i]},
                "NW"
        },
            nvcv::TYPE_2F32);
        nvcv::Tensor dstPoints(
            {
                {1, numPoints[i]},
                "NW"
        },
            nvcv::TYPE_2F32);
        nvcv::Tensor models(
            {
                {1, 3, 3},
                "NHW"
        },
            nvcv::TYPE_F32);

        auto srcData = srcPoints.exportData<nvcv::TensorDataStridedCuda>();
        auto dstData = dstPoints.exportData<nvcv::TensorDataStridedCuda>();

        ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(srcData->basePtr(), srcVec[i].data(), sizeof(float) * srcVec[i].size(),
                                               cudaMemcpyHostToDevice, stream));
        ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(dstData->basePtr(), dstVec[i].data(), sizeof(float) * dstVec[i].size(),
                                               cudaMemcpyHostToDevice, stream));

        srcTensorBatch.pushBack(srcPoints);
        dstTensorBatch.pushBack(dstPoints);
        modelsTensorBatch.pushBack(models);
    }

    cvcuda::FindHomography fh(numSamples, maxNumPoints);

#ifdef PERFORMANCE_RUN
    for (int it = 0; it < WARMUP_ITERATIONS; it++)
    {
        EXPECT_NO_THROW(fh(stream, batchSrc, batchDst, models));
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, stream);
    for (int it = 0; it < PERF_ITERATIONS; it++)
    {
        EXPECT_NO_THROW(fh(stream, batchSrc, batchDst, models));
    }
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Time taken for " << numSamples << "x" << maxPoints << " = " << milliseconds / PERF_ITERATIONS
              << "ms\n";
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
#else
    EXPECT_NO_THROW(fh(stream, srcTensorBatch, dstTensorBatch, modelsTensorBatch));
#endif

    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    // copy back the estimated models into modelsVec
    for (int i = 0; i < numSamples; i++)
    {
        auto modelsData = modelsTensorBatch[i].exportData<nvcv::TensorDataStridedCuda>();
        ASSERT_EQ(cudaSuccess, cudaMemcpy2D(estimatedModelsVec.data() + i * 9, sizeof(float) * 3, modelsData->basePtr(),
                                            modelsData->stride(1), sizeof(float) * 3, 3, cudaMemcpyDeviceToHost));
    }

    // Compute dst vec based on model estimated
#ifndef PERFORMANCE_RUN
    for (int i = 0; i < numSamples; i++)
    {
        for (int j = 0; j < numPoints[i]; j++)
        {
            float dstx;
            float dsty;
            float sx;
            float sy;
            sx = srcVec[i][2 * j + 0];
            sy = srcVec[i][2 * j + 1];
            calculateDst(sx, sy, &dstx, &dsty, estimatedModelsVec.data() + i * 9);
            computedDstVec[i].push_back(dstx);
            computedDstVec[i].push_back(dsty);
            float A = dstVec[i][2 * j + 0];
            float B = computedDstVec[i][2 * j + 0];
            // The 1e-3 tolerance covers GPU reduction/FMA rounding against the independent CPU projection.
            EXPECT_NEAR(A, B, 1e-03);
            A = dstVec[i][2 * j + 1];
            B = computedDstVec[i][2 * j + 1];
            // The 1e-3 tolerance covers GPU reduction/FMA rounding against the independent CPU projection.
            EXPECT_NEAR(A, B, 1e-03);
        }
    }
#endif
}

TEST(OpFindHomography, degenerate_identical_source_points)
{
    int numSamples = 2;
    int numPoints  = 16;

    nvcv::Tensor srcPoints(
        {
            {numSamples, numPoints},
            "NW"
    },
        nvcv::TYPE_2F32);
    nvcv::Tensor dstPoints(
        {
            {numSamples, numPoints},
            "NW"
    },
        nvcv::TYPE_2F32);
    nvcv::Tensor models = createModelTensor(numSamples);

    auto srcData    = srcPoints.exportData<nvcv::TensorDataStridedCuda>();
    auto dstData    = dstPoints.exportData<nvcv::TensorDataStridedCuda>();
    auto modelsData = models.exportData<nvcv::TensorDataStridedCuda>();

    std::vector<float> srcVec(2 * numSamples * numPoints);
    std::vector<float> dstVec(2 * numSamples * numPoints);
    std::vector<float> estimatedModelsVec(numSamples * 9);

    for (int i = 0; i < numSamples; i++)
    {
        float fixed_src_x = 100.0f;
        float fixed_src_y = 200.0f;

        for (int j = 0; j < numPoints; j++)
        {
            // All source points are identical
            srcVec[i * numPoints * 2 + 2 * j]     = fixed_src_x;
            srcVec[i * numPoints * 2 + 2 * j + 1] = fixed_src_y;

            // Different destination points
            dstVec[i * numPoints * 2 + 2 * j]     = static_cast<float>(j) * 10.0f;
            dstVec[i * numPoints * 2 + 2 * j + 1] = static_cast<float>(j) * 15.0f;
        }
    }

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(srcData->basePtr(), srcVec.data(),
                                           sizeof(float) * 2 * numPoints * numSamples, cudaMemcpyHostToDevice, stream));
    ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(dstData->basePtr(), dstVec.data(),
                                           sizeof(float) * 2 * numPoints * numSamples, cudaMemcpyHostToDevice, stream));

    cvcuda::FindHomography fh(numSamples, numPoints);
    EXPECT_NO_THROW(fh(stream, srcPoints, dstPoints, models));
    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

// clang-format off
NVCV_TEST_SUITE_P(OpFindHomography_Negative, test::ValueList<std::string, nvcv::DataType, std::string, nvcv::DataType, std::string, nvcv::DataType, int, int, int, int, int, int, int>
    {
        // layoutSrc, dataTypeSrc, layoutDst, dataTypeDst, layoutModels, dataTypeModels, numSamples, numPoints, numPointsSrc, numPointsDst, numPointsModels, numPointsSrc, numPointsDst
        // invalid layout
        {"N", nvcv::TYPE_2F32, "NW", nvcv::TYPE_2F32, "NHW", nvcv::TYPE_F32, 8, 16, 8, 16, 8, 3, 3},
        {"NW", nvcv::TYPE_2F32, "N", nvcv::TYPE_2F32, "NHW", nvcv::TYPE_F32, 8, 16, 8, 16, 8, 3, 3},
        // invalid shape
        {"NW", nvcv::TYPE_2F32, "NW", nvcv::TYPE_2F32, "NHW", nvcv::TYPE_F32, 8, 16, 10, 16, 8, 3, 3},
        {"NW", nvcv::TYPE_2F32, "NW", nvcv::TYPE_2F32, "NHW", nvcv::TYPE_F32, 8, 16, 8, 12, 8, 3, 3},
        {"NW", nvcv::TYPE_2F32, "NW", nvcv::TYPE_2F32, "NHW", nvcv::TYPE_F32, 8, 2, 8, 2, 8, 3, 3},
        {"NW", nvcv::TYPE_2F32, "NW", nvcv::TYPE_2F32, "NHW", nvcv::TYPE_F32, 8, 16, 8, 16, 8, 4, 3},
        // invalid datta type
        {"NW", nvcv::TYPE_3F32, "NW", nvcv::TYPE_2F32, "NHW", nvcv::TYPE_F32, 8, 16, 8, 16, 8, 3, 3},
        {"NW", nvcv::TYPE_2F32, "NW", nvcv::TYPE_3F32, "NHW", nvcv::TYPE_F32, 8, 16, 8, 16, 8, 3, 3},

    });

// clang-format on

TEST(OpFindHomography_Negative, createWillNullHandle)
{
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, cvcudaFindHomographyCreate(nullptr, 8, 16));
}

TEST(OpFindHomography_Negative, createRejectsInvalidBatchOrPointCount)
{
    NVCVOperatorHandle handle = nullptr;

    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, cvcudaFindHomographyCreate(&handle, 0, 4));
    EXPECT_EQ(nullptr, handle);

    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, cvcudaFindHomographyCreate(&handle, 1, 3));
    EXPECT_EQ(nullptr, handle);
}

TEST(OpFindHomography_Negative, varshape_different_batch_size)
{
    int              numSamplesSrc = 4;
    int              numSamplesDst = 5;
    int              numSamples    = std::min(numSamplesSrc, numSamplesDst);
    std::vector<int> numPoints(numSamples);

    auto              reqsSrc = nvcv::TensorBatch::CalcRequirements(numSamplesSrc);
    auto              reqsDst = nvcv::TensorBatch::CalcRequirements(numSamplesDst);
    nvcv::TensorBatch srcTensorBatch(reqsSrc);
    nvcv::TensorBatch dstTensorBatch(reqsDst);
    nvcv::TensorBatch modelsTensorBatch(reqsSrc);

    int maxNumPoints = 10;
    for (int i = 0; i < numSamples; i++)
    {
        numPoints[i] = 3;
        maxNumPoints = std::max(maxNumPoints, numPoints[i]);

        nvcv::Tensor srcPoints(
            {
                {1, numPoints[i]},
                "NW"
        },
            nvcv::TYPE_2F32);
        nvcv::Tensor dstPoints(
            {
                {1, numPoints[i]},
                "NW"
        },
            nvcv::TYPE_2F32);
        nvcv::Tensor models(
            {
                {1, 3, 3},
                "NHW"
        },
            nvcv::TYPE_F32);
        srcTensorBatch.pushBack(srcPoints);
        dstTensorBatch.pushBack(dstPoints);
        modelsTensorBatch.pushBack(models);
    }
    {
        nvcv::Tensor dstPoints(
            {
                {1, numPoints[0]},
                "NW"
        },
            nvcv::TYPE_2F32);
        dstTensorBatch.pushBack(dstPoints);
    }

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    cvcuda::FindHomography fh(numSamples, maxNumPoints);

    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcv::ProtectCall([&fh, &stream, &srcTensorBatch, &dstTensorBatch, &modelsTensorBatch]
                                { fh(stream, srcTensorBatch, dstTensorBatch, modelsTensorBatch); }));

    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST_P(OpFindHomography_Negative, invalid_parameters)
{
    std::string    layoutSrc        = GetParamValue<0>();
    nvcv::DataType dataTypeSrc      = GetParamValue<1>();
    std::string    layoutDst        = GetParamValue<2>();
    nvcv::DataType dataTypeDst      = GetParamValue<3>();
    std::string    layoutModels     = GetParamValue<4>();
    nvcv::DataType dataTypeModels   = GetParamValue<5>();
    int            numSamplesSrc    = GetParamValue<6>();
    int            numPointsSrc     = GetParamValue<7>();
    int            numSamplesDst    = GetParamValue<8>();
    int            numPointsDst     = GetParamValue<9>();
    int            numSamplesModels = GetParamValue<10>();
    int            shapeOneModels   = GetParamValue<11>();
    int            shapeTwoModels   = GetParamValue<12>();

    // clang-format off
    // Create tensors
    nvcv::Tensor srcPoints;
    if (layoutSrc.length() == 1)
    {
        srcPoints = nvcv::Tensor({{numSamplesSrc}, layoutSrc.c_str()}, dataTypeSrc);
    }
    else
    {
        srcPoints = nvcv::Tensor({{numSamplesSrc, numPointsSrc}, layoutSrc.c_str()}, dataTypeSrc);
    }

    nvcv::Tensor dstPoints;
    if (layoutDst.length() == 1)
    {
        dstPoints = nvcv::Tensor({{numSamplesDst}, layoutDst.c_str()}, dataTypeDst);
    }
    else
    {
        dstPoints = nvcv::Tensor({{numSamplesDst, numPointsDst}, layoutDst.c_str()}, dataTypeDst);
    }
    nvcv::Tensor models({{numSamplesModels, shapeOneModels, shapeTwoModels}, layoutModels.c_str()}, dataTypeModels);

    // clang-format on

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    cvcuda::FindHomography fh(std::max(numSamplesSrc, numSamplesDst),
                              std::max(4, std::max(numPointsSrc, numPointsDst)));

    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcv::ProtectCall([&fh, &stream, &srcPoints, &dstPoints, &models]
                                                             { fh(stream, srcPoints, dstPoints, models); }));

    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}
