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

// Minimal driver for the ImageArea operator: computes the area of a synthetic
// image and compares it against the expected W * H value. Useful as the entry
// point for gdb / cuda-gdb / compute-sanitizer.

#include <cvcuda/OpImageArea.hpp>
#include <nvcv/Tensor.hpp>

#include <cuda_runtime.h>

#include <cstdio>
#include <exception>

namespace {

int64_t ExpectedArea(int width, int height)
{
    return static_cast<int64_t>(width) * height;
}

bool Check(const char *name, nvcv::Tensor &in, int width, int height)
{
    cudaStream_t stream = nullptr;
    if (cudaStreamCreate(&stream) != cudaSuccess)
    {
        std::fprintf(stderr, "%s: cudaStreamCreate failed\n", name);
        return false;
    }

    nvcv::Tensor out({{1}, "N"}, nvcv::TYPE_S32);

    bool ok = true;
    try
    {
        cvcuda::ImageArea op;
        op(stream, in, out);
        if (cudaStreamSynchronize(stream) != cudaSuccess)
        {
            std::fprintf(stderr, "%s: cudaStreamSynchronize failed\n", name);
            ok = false;
        }
    }
    catch (const std::exception &e)
    {
        std::fprintf(stderr, "%s: %s\n", name, e.what());
        ok = false;
    }

    int32_t got = 0;
    if (ok)
    {
        auto outData = out.exportData<nvcv::TensorDataStridedCuda>();
        if (outData == nullptr)
        {
            std::fprintf(stderr, "%s: cannot export output tensor data\n", name);
            ok = false;
        }
        else if (cudaMemcpy(&got, outData->basePtr(), sizeof(int32_t), cudaMemcpyDeviceToHost) != cudaSuccess)
        {
            std::fprintf(stderr, "%s: cudaMemcpy failed\n", name);
            ok = false;
        }
    }

    const int64_t want = ExpectedArea(width, height);
    const bool    match = ok && got == want;
    std::printf("%-4s %5dx%-5d area=%d expected=%lld %s\n", name, width, height, got,
                static_cast<long long>(want), match ? "OK" : "MISMATCH");

    cudaStreamDestroy(stream);
    return match;
}

} // namespace

int main()
{
    // Interleaved (NHWC) and planar (NCHW) inputs validate both layout paths.
    nvcv::Tensor imgNHWC({{1, 480, 640, 3}, "NHWC"}, nvcv::TYPE_U8);
    nvcv::Tensor imgNCHW({{1, 3, 50, 100}, "NCHW"}, nvcv::TYPE_U8);

    const bool ok = Check("NHWC", imgNHWC, 640, 480) & Check("NCHW", imgNCHW, 100, 50);

    std::printf("%s\n", ok ? "PASSED" : "FAILED");
    return ok ? 0 : 1;
}
