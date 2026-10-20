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

#include <cvcuda/priv/OpBoxBlur.hpp>
#include <cvcuda/priv/Types.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorDataAccess.hpp>

#include <cstdint>
#include <vector>

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

// Out of place, the operator either moves the image with a dense host-side copy and then elides its
// in-kernel copy pass, or falls back to that copy pass; either way the blur samples the immutable
// input and the untouched pixels reach the destination. A batched NHWC destination takes the dense
// copy, an HWC destination is outside it, so the two paths must agree pixel for pixel from a
// destination that starts as neither the input nor the answer.
TEST(OpBoxBlurPrivate, SkipCopyPathMatchesKernelCopyPath)
{
    constexpr int numSamples = 1;
    constexpr int width      = 128;
    constexpr int height     = 96;

    nvcv::Tensor input(numSamples, {width, height}, nvcv::FMT_RGB8);
    nvcv::Tensor kernelCopyOutput(
        nvcv::TensorShape{
            {height, width, 3},
            "HWC"
    },
        nvcv::TYPE_U8);
    nvcv::Tensor skipCopyOutput(numSamples, {width, height}, nvcv::FMT_RGB8);

    auto inputData      = input.exportData<nvcv::TensorDataStridedCuda>();
    auto kernelCopyData = kernelCopyOutput.exportData<nvcv::TensorDataStridedCuda>();
    auto skipCopyData   = skipCopyOutput.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_TRUE(inputData);
    ASSERT_TRUE(kernelCopyData);
    ASSERT_TRUE(skipCopyData);

    auto inputAccess      = nvcv::TensorDataAccessStridedImagePlanar::Create(*inputData);
    auto kernelCopyAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*kernelCopyData);
    auto skipCopyAccess   = nvcv::TensorDataAccessStridedImagePlanar::Create(*skipCopyData);
    ASSERT_TRUE(inputAccess);
    ASSERT_TRUE(kernelCopyAccess);
    ASSERT_TRUE(skipCopyAccess);

    const auto           bufferSize = static_cast<size_t>(inputAccess->sampleStride() * inputAccess->numSamples());
    std::vector<uint8_t> inputHost(bufferSize);
    for (size_t i = 0; i < inputHost.size(); ++i)
    {
        inputHost[i] = static_cast<uint8_t>((i * 31 + 7) & 0xFF);
    }

    ASSERT_EQ(cudaSuccess, cudaMemcpy(inputData->basePtr(), inputHost.data(), bufferSize, cudaMemcpyHostToDevice));

    // Poison both destinations so a copy the operator fails to make cannot be mistaken for one it
    // made: neither buffer starts out holding the input.
    ASSERT_EQ(cudaSuccess,
              cudaMemset(skipCopyData->basePtr(), 0xCD,
                         static_cast<size_t>(skipCopyAccess->sampleStride() * skipCopyAccess->numSamples())));
    ASSERT_EQ(cudaSuccess,
              cudaMemset(kernelCopyData->basePtr(), 0xCD, static_cast<size_t>(kernelCopyAccess->rowStride() * height)));

    constexpr NVCVBlurBoxI box{
        {16, 12, 64, 64},
        5
    };
    auto blurBoxes = std::make_shared<cvcuda::priv::NVCVBlurBoxesImpl>(std::vector<std::vector<NVCVBlurBoxI>>{{box}});

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    cvcuda::priv::BoxBlur op;

    EXPECT_NO_THROW(op(stream, input, kernelCopyOutput, (NVCVBlurBoxesI)blurBoxes.get()));
    EXPECT_NO_THROW(op(stream, input, skipCopyOutput, (NVCVBlurBoxesI)blurBoxes.get()));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    const auto kernelCopyPixels = DownloadPixels(*kernelCopyAccess);
    const auto inputPixels      = DownloadPixels(*inputAccess);
    EXPECT_EQ(kernelCopyPixels, DownloadPixels(*skipCopyAccess));
    EXPECT_NE(kernelCopyPixels, inputPixels) << "The blur box must have changed the image";

    // Pixels outside the blur box must pass through to the output unchanged.
    auto pixelsOutsideBox = [](const std::vector<uint8_t> &src, int w, int h, const NVCVBlurBoxI &b)
    {
        std::vector<uint8_t> out;
        for (int row = 0; row < h; ++row)
            for (int col = 0; col < w; ++col)
                if (col < b.box.x || col >= b.box.x + b.box.width || row < b.box.y || row >= b.box.y + b.box.height)
                {
                    const int off = (row * w + col) * 3;
                    out.insert(out.end(), src.begin() + off, src.begin() + off + 3);
                }
        return out;
    };
    EXPECT_EQ(pixelsOutsideBox(kernelCopyPixels, width, height, box), pixelsOutsideBox(inputPixels, width, height, box))
        << "Pixels outside the blur box must be copied from input unchanged";
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}
