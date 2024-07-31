/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

static std::default_random_engine g_rng(std::random_device{}());

static void calculateDst(float x, float y, float *X, float *Y, float *model)
{
    *X = model[0] * x + model[1] * y + model[2] * 1;
    *Y = model[3] * x + model[4] * y + model[5] * 1;
}

static void calculateGoldModelMatrix(float *m, std::mt19937 &rng, std::uniform_int_distribution<int> &dis)
{
    // random rotation angle between 0 and pi
    float                           theta = (M_PI / 2.0) * dis(rng) / 100;
    float                           Tx    = (float)dis(rng) / 100;
    float                           Ty    = (float)dis(rng) / 100;
    float                           sx    = (float)dis(rng) / 100;
    float                           sy    = (float)dis(rng) / 100;
    float                           p1    = (float)dis(rng) / 100;
    float                           p2    = (float)dis(rng) / 100 * 2;
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
    // numSamples, numPoints}
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

    std::random_device              rd;
    std::mt19937                    gen(rd()); // Mersenne Twister engine
    std::uniform_int_distribution<> dis(0, 100);

    int numXPoints = static_cast<int>(std::sqrt(numPoints));
    int numYPoints = numXPoints;

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
                srcVec[i * numPoints * 2 + 2 * idx]     = dis(gen);
                srcVec[i * numPoints * 2 + 2 * idx + 1] = dis(gen);

                float dstx, dsty;
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

    ASSERT_EQ(cudaSuccess, cudaMemcpy(srcData->basePtr(), srcVec.data(), sizeof(float) * 2 * numPoints * numSamples,
                                      cudaMemcpyHostToDevice));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(dstData->basePtr(), dstVec.data(), sizeof(float) * 2 * numPoints * numSamples,
                                      cudaMemcpyHostToDevice));

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

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
                float dstx, dsty;
                calculateDst(srcVec[i * numPoints * 2 + 2 * idx], srcVec[i * numPoints * 2 + 2 * idx + 1], &dstx, &dsty,
                             estimatedModelsVec.data() + i * 9);
                computedDstVec[i * numPoints * 2 + 2 * idx]     = dstx;
                computedDstVec[i * numPoints * 2 + 2 * idx + 1] = dsty;
                float A                                         = dstVec[i * numPoints * 2 + 2 * idx];
                float B                                         = computedDstVec[i * numPoints * 2 + 2 * idx];
                EXPECT_NEAR(A, B, 1e-03);
                A = dstVec[i * numPoints * 2 + 2 * idx + 1];
                B = computedDstVec[i * numPoints * 2 + 2 * idx + 1];
                EXPECT_NEAR(A, B, 1e-03);
            }
        }
    }
#endif
}

TEST_P(OpFindHomography, varshape_correct_output)
{
    int              numSamples = GetParamValue<0>();
    int              maxPoints  = GetParamValue<1>();
    std::vector<int> numPoints(numSamples);
    std::vector<int> numXPoints(numSamples);

    std::mt19937                       rng(12345);
    std::uniform_int_distribution<int> dis(0, 100);
    std::uniform_int_distribution<int> dis_num_points(4, maxPoints);

    auto              reqs = nvcv::TensorBatch::CalcRequirements(numSamples);
    nvcv::TensorBatch srcTensorBatch(reqs);
    nvcv::TensorBatch dstTensorBatch(reqs);
    nvcv::TensorBatch modelsTensorBatch(reqs);

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
            srcVec[i].push_back(sx);
            srcVec[i].push_back(sy);

            float dstx, dsty;
            calculateDst(sx, sy, &dstx, &dsty, modelsVec.data() + i * 9);
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

        ASSERT_EQ(cudaSuccess, cudaMemcpy(srcData->basePtr(), srcVec[i].data(), sizeof(float) * srcVec[i].size(),
                                          cudaMemcpyHostToDevice));
        ASSERT_EQ(cudaSuccess, cudaMemcpy(dstData->basePtr(), dstVec[i].data(), sizeof(float) * dstVec[i].size(),
                                          cudaMemcpyHostToDevice));

        srcTensorBatch.pushBack(srcPoints);
        dstTensorBatch.pushBack(dstPoints);
        modelsTensorBatch.pushBack(models);
    }

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

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
            float dstx, dsty;
            float sx, sy;
            sx = srcVec[i][2 * j + 0];
            sy = srcVec[i][2 * j + 1];
            calculateDst(sx, sy, &dstx, &dsty, estimatedModelsVec.data() + i * 9);
            computedDstVec[i].push_back(dstx);
            computedDstVec[i].push_back(dsty);
            float A = dstVec[i][2 * j + 0];
            float B = computedDstVec[i][2 * j + 0];
            EXPECT_NEAR(A, B, 1e-03);
            A = dstVec[i][2 * j + 1];
            B = computedDstVec[i][2 * j + 1];
            EXPECT_NEAR(A, B, 1e-03);
        }
    }
#endif
}
