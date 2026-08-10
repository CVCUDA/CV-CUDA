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

#include <cvcuda/priv/Types.hpp>
#include <cvcuda/priv/legacy/CvCudaLegacy.h>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorDataAccess.hpp>

#include <cstdint>
#include <vector>

namespace legacy = nvcv::legacy::cuda_op;

namespace {

std::vector<uint8_t> DownloadPixels(const nvcv::TensorDataAccessStridedImagePlanar &access)
{
    const auto samples  = static_cast<size_t>(access.numSamples());
    const auto rows     = static_cast<size_t>(access.numRows());
    const auto cols     = static_cast<size_t>(access.numCols());
    const auto channels = static_cast<size_t>(access.numChannels());
    const auto rowBytes = cols * channels;

    std::vector<uint8_t> pixels(samples * rows * rowBytes);
    for (size_t n = 0; n < samples; ++n)
    {
        EXPECT_EQ(
            cudaSuccess,
            cudaMemcpy2D(pixels.data() + n * rows * rowBytes, rowBytes, access.sampleData(static_cast<int64_t>(n)),
                         static_cast<size_t>(access.rowStride()), rowBytes, rows, cudaMemcpyDeviceToHost));
    }
    return pixels;
}

} // namespace

TEST(OpBoxBlurPrivate, SkipCopyKeepsInputAsBlurSource)
{
    constexpr int numSamples = 1;
    constexpr int width      = 128;
    constexpr int height     = 96;

    nvcv::Tensor input(numSamples, {width, height}, nvcv::FMT_RGB8);
    nvcv::Tensor normalOutput(numSamples, {width, height}, nvcv::FMT_RGB8);
    nvcv::Tensor skipCopyOutput(numSamples, {width, height}, nvcv::FMT_RGB8);

    auto inputData    = input.exportData<nvcv::TensorDataStridedCuda>();
    auto normalData   = normalOutput.exportData<nvcv::TensorDataStridedCuda>();
    auto skipCopyData = skipCopyOutput.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_TRUE(inputData);
    ASSERT_TRUE(normalData);
    ASSERT_TRUE(skipCopyData);

    auto inputAccess    = nvcv::TensorDataAccessStridedImagePlanar::Create(*inputData);
    auto normalAccess   = nvcv::TensorDataAccessStridedImagePlanar::Create(*normalData);
    auto skipCopyAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*skipCopyData);
    ASSERT_TRUE(inputAccess);
    ASSERT_TRUE(normalAccess);
    ASSERT_TRUE(skipCopyAccess);

    const auto           bufferSize = static_cast<size_t>(inputAccess->sampleStride() * inputAccess->numSamples());
    std::vector<uint8_t> inputHost(bufferSize);
    for (size_t i = 0; i < inputHost.size(); ++i)
    {
        inputHost[i] = static_cast<uint8_t>((i * 31 + 7) & 0xFF);
    }

    ASSERT_EQ(cudaSuccess, cudaMemcpy(inputData->basePtr(), inputHost.data(), bufferSize, cudaMemcpyHostToDevice));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(skipCopyData->basePtr(), inputHost.data(), bufferSize, cudaMemcpyHostToDevice));

    constexpr NVCVBlurBoxI box{
        {16, 12, 64, 64},
        5
    };
    auto blurBoxes = std::make_shared<cvcuda::priv::NVCVBlurBoxesImpl>(std::vector<std::vector<NVCVBlurBoxI>>{{box}});

    // Corrupt the copied output inside the ROI. A skip-copy blur must still sample the immutable input tensor.
    auto *corruptStart = skipCopyAccess->sampleData(0) + box.box.y * skipCopyAccess->rowStride()
                       + box.box.x * skipCopyAccess->colStride();
    ASSERT_EQ(cudaSuccess, cudaMemset2D(corruptStart, static_cast<size_t>(skipCopyAccess->rowStride()), 0,
                                        static_cast<size_t>((box.box.width - 1) * skipCopyAccess->colStride()),
                                        static_cast<size_t>(box.box.height - 1)));

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    legacy::DataShape maxInput;
    legacy::DataShape maxOutput;
    legacy::BoxBlur   op(maxInput, maxOutput);

    EXPECT_EQ(legacy::ErrorCode::SUCCESS, op.infer(*inputData, *normalData, (NVCVBlurBoxesI)blurBoxes.get(), stream));
    EXPECT_EQ(legacy::ErrorCode::SUCCESS,
              op.infer(*inputData, *skipCopyData, (NVCVBlurBoxesI)blurBoxes.get(), stream, true));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    EXPECT_EQ(DownloadPixels(*normalAccess), DownloadPixels(*skipCopyAccess));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}
