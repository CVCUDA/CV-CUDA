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

#include "BenchClockProbe.hpp"

#include <cuda_runtime.h>

#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <string_view>

namespace benchutils {

namespace {

constexpr std::uint64_t kSettleNs    = 5'000'000;
constexpr std::uint64_t kSampleNs    = 1'000'000;
constexpr int           kThreads     = 256;
constexpr int           kBlocksPerSm = 4;
// A100/H100 frequency steps are 15 MHz; 7 MHz keeps adjacent bins distinct.
constexpr int           kToleranceMHz = 7;

constexpr std::string_view kVerifiedPrefix          = "SM clock verified:";
constexpr std::string_view kMismatchPrefix          = "SM clock mismatch:";
constexpr std::string_view kVerificationErrorPrefix = "SM clock verification error:";
constexpr std::string_view kDeviceErrorPrefix       = "SM clock device error:";

enum class SmClockErrorOrigin
{
    NONE,
    AMBIENT,
    INVALID_SM_COUNT,
    PROBE
};

struct SmClockMeasurement
{
    int                mhz;
    cudaError_t        status;
    SmClockErrorOrigin errorOrigin;
    int                smCount;
};

static void emitSmClockMessage(std::ostream &stream, std::string_view prefix, const std::string &detail)
{
    stream << prefix << ' ' << detail << std::endl;
}

__device__ __forceinline__ std::uint64_t globalTimer()
{
    std::uint64_t value;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
    return value;
}

__device__ __forceinline__ std::uint32_t busyStep(std::uint32_t value)
{
    value = value * 1664525u + 1013904223u;
    asm volatile("" : "+r"(value));
    return value;
}

__device__ double g_smClockMHz;

__global__ void measureSmClockKernel(double *clockMhz)
{
    // Every thread spins so the sampled clock reflects an actively loaded GPU.
    const std::uint64_t settleStartNs = globalTimer();
    std::uint32_t       work          = blockIdx.x * blockDim.x + threadIdx.x + 1;
    std::uint64_t       startNs;
    do
    {
        work    = busyStep(work);
        startNs = globalTimer();
    }
    while (startNs - settleStartNs < kSettleNs);

    const std::uint64_t startCycles = clock64();
    std::uint64_t       endNs;
    do
    {
        work  = busyStep(work);
        endNs = globalTimer();
    }
    while (endNs - startNs < kSampleNs);
    const std::uint64_t endCycles = clock64();

    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        *clockMhz = static_cast<double>(endCycles - startCycles) * 1000.0 / static_cast<double>(endNs - startNs);
    }
}

static bool parseExpectedSmClock(const char *rawExpected, int &expected)
{
    if (rawExpected == nullptr)
    {
        return false;
    }

    char *end              = nullptr;
    errno                  = 0;
    const long parsedValue = std::strtol(rawExpected, &end, 10);
    if (errno != 0 || end == rawExpected)
    {
        return false;
    }
    while (std::isspace(static_cast<unsigned char>(*end)))
    {
        ++end;
    }
    if (*end != '\0' || parsedValue <= 0 || parsedValue > std::numeric_limits<int>::max())
    {
        return false;
    }
    expected = static_cast<int>(parsedValue);
    return true;
}

static SmClockMeasurement measureSmClockMHz()
{
    // Surface asynchronous errors from preceding benchmark work before the probe starts.
    const cudaError_t ambientStatus = cudaDeviceSynchronize();
    if (ambientStatus != cudaSuccess)
    {
        return {-1, ambientStatus, SmClockErrorOrigin::AMBIENT, 0};
    }

    int         device  = 0;
    int         smCount = 0;
    cudaError_t status  = cudaGetDevice(&device);
    if (status == cudaSuccess)
    {
        status = cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);
    }
    if (status != cudaSuccess)
    {
        return {-1, status, SmClockErrorOrigin::PROBE, smCount};
    }
    if (smCount <= 0)
    {
        return {-1, cudaErrorInvalidConfiguration, SmClockErrorOrigin::INVALID_SM_COUNT, smCount};
    }

    void *deviceClockAddress = nullptr;
    status                   = cudaGetSymbolAddress(&deviceClockAddress, g_smClockMHz);
    if (status != cudaSuccess)
    {
        return {-1, status, SmClockErrorOrigin::PROBE, smCount};
    }
    auto *deviceClock = static_cast<double *>(deviceClockAddress);
    status            = cudaMemset(deviceClock, 0, sizeof(*deviceClock));
    if (status != cudaSuccess)
    {
        return {-1, status, SmClockErrorOrigin::PROBE, smCount};
    }

    measureSmClockKernel<<<smCount * kBlocksPerSm, kThreads>>>(deviceClock);
    status = cudaGetLastError();
    if (status != cudaSuccess)
    {
        return {-1, status, SmClockErrorOrigin::PROBE, smCount};
    }

    double hostClock = 0;
    status           = cudaMemcpy(&hostClock, deviceClock, sizeof(hostClock), cudaMemcpyDeviceToHost);
    if (status != cudaSuccess)
    {
        return {-1, status, SmClockErrorOrigin::PROBE, smCount};
    }
    return {static_cast<int>(std::lround(hostClock)), cudaSuccess, SmClockErrorOrigin::NONE, smCount};
}

} // namespace

void verifyExpectedSmClock()
{
    const char *rawExpected = std::getenv("CVCUDA_BENCH_EXPECT_SM_CLOCK_MHZ");
    if (rawExpected == nullptr)
    {
        return;
    }

    int expected = 0;
    if (!parseExpectedSmClock(rawExpected, expected))
    {
        emitSmClockMessage(std::cerr, kVerificationErrorPrefix,
                           "invalid expected clock '" + std::string(rawExpected) + "'");
        std::_Exit(EXIT_FAILURE);
    }

    const SmClockMeasurement measurement = measureSmClockMHz();
    if (measurement.status != cudaSuccess)
    {
        const char *status = cudaGetErrorString(measurement.status);
        switch (measurement.errorOrigin)
        {
        case SmClockErrorOrigin::AMBIENT:
            std::cerr << "Pre-existing CUDA error: " << status << std::endl;
            break;
        case SmClockErrorOrigin::INVALID_SM_COUNT:
            emitSmClockMessage(std::cerr, kDeviceErrorPrefix,
                               "invalid multiprocessor count: " + std::to_string(measurement.smCount));
            break;
        case SmClockErrorOrigin::PROBE:
            emitSmClockMessage(std::cerr, kVerificationErrorPrefix, "device probe failed: " + std::string(status));
            break;
        case SmClockErrorOrigin::NONE:
            emitSmClockMessage(std::cerr, kVerificationErrorPrefix,
                               "unclassified device probe failure: " + std::string(status));
            break;
        }
        std::_Exit(EXIT_FAILURE);
    }
    const int measured = measurement.mhz;
    if (measured <= 0)
    {
        emitSmClockMessage(std::cerr, kVerificationErrorPrefix,
                           "device probe returned " + std::to_string(measured) + " MHz");
        std::_Exit(EXIT_FAILURE);
    }
    if (std::abs(measured - expected) > kToleranceMHz)
    {
        emitSmClockMessage(std::cerr, kMismatchPrefix,
                           "expected " + std::to_string(expected) + " MHz, measured " + std::to_string(measured)
                               + " MHz (tolerance " + std::to_string(kToleranceMHz) + " MHz)");
        std::_Exit(EXIT_FAILURE);
    }
    emitSmClockMessage(std::cout, kVerifiedPrefix, std::to_string(measured) + " MHz");
}

} // namespace benchutils
