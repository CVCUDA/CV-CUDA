// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cuda_runtime.h>
#include <cvcuda/OpConvertTo.hpp>
#include <gtest/gtest.h>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorData.hpp>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace {

constexpr int kSamples  = 2;
constexpr int kChannels = 3;
constexpr int kHeight   = 128;
constexpr int kWidth    = 256;
constexpr int kElements = kSamples * kChannels * kHeight * kWidth;

struct LifetimeStorage
{
    NVCVByte *source{};
    NVCVByte *destination{};

    LifetimeStorage()
    {
        if (cudaMalloc(&source, kElements) != cudaSuccess)
            throw std::runtime_error("source allocation failed");
        if (cudaMalloc(&destination, kElements * sizeof(float)) != cudaSuccess)
        {
            cudaFree(source);
            throw std::runtime_error("destination allocation failed");
        }
    }

    LifetimeStorage(const LifetimeStorage &)            = delete;
    LifetimeStorage &operator=(const LifetimeStorage &) = delete;

    ~LifetimeStorage()
    {
        cudaFree(destination);
        cudaFree(source);
    }
};

nvcv::Tensor WrapLifetimeTensor(NVCVByte *pointer, nvcv::DataType dtype)
{
    nvcv::TensorDataStridedCuda::Buffer buffer{};
    buffer.basePtr    = pointer;
    buffer.strides[3] = dtype.strideBytes();
    buffer.strides[2] = kWidth * buffer.strides[3];
    buffer.strides[1] = kHeight * buffer.strides[2];
    buffer.strides[0] = kChannels * buffer.strides[1];
    return nvcv::TensorWrapData(nvcv::TensorDataStridedCuda{
        nvcv::TensorShape{{kSamples, kChannels, kHeight, kWidth}, "NCHW"},
        dtype, buffer
    });
}

std::vector<uint8_t> LifetimeInput(int seed)
{
    std::vector<uint8_t> input(kElements);
    for (int element = 0; element < kElements; ++element) input[element] = static_cast<uint8_t>(element * 73 + seed);
    return input;
}

void CheckLifetimeGold(const LifetimeStorage &storage, const std::vector<uint8_t> &input, float alpha, float beta,
                       bool secondPass)
{
    std::vector<uint8_t> expected(kElements * sizeof(float));
    std::vector<uint8_t> actual(expected.size());
    for (int element = 0; element < kElements; ++element)
    {
        float value = std::fma(alpha, static_cast<float>(input[element]), beta);
        if (secondPass)
            value = std::fma(0.5f, value, -4.0f);
        std::memcpy(expected.data() + element * sizeof(float), &value, sizeof(value));
    }
    ASSERT_EQ(cudaSuccess, cudaMemcpy(actual.data(), storage.destination, actual.size(), cudaMemcpyDeviceToHost));
    EXPECT_EQ(expected, actual);
    std::vector<uint8_t> sourceAfter(input.size());
    ASSERT_EQ(cudaSuccess, cudaMemcpy(sourceAfter.data(), storage.source, sourceAfter.size(), cudaMemcpyDeviceToHost));
    EXPECT_EQ(input, sourceAfter);
}

} // namespace

TEST(OpConvertToLifetime, shared_operator_two_nonblocking_streams)
{
    LifetimeStorage first;
    LifetimeStorage second;
    const auto      firstInput  = LifetimeInput(19);
    const auto      secondInput = LifetimeInput(127);
    cudaStream_t    firstStream{}, secondStream{};
    ASSERT_EQ(cudaSuccess, cudaStreamCreateWithFlags(&firstStream, cudaStreamNonBlocking));
    std::unique_ptr<std::remove_pointer_t<cudaStream_t>, decltype(&cudaStreamDestroy)> firstStreamOwner(
        firstStream, &cudaStreamDestroy);
    ASSERT_EQ(cudaSuccess, cudaStreamCreateWithFlags(&secondStream, cudaStreamNonBlocking));
    std::unique_ptr<std::remove_pointer_t<cudaStream_t>, decltype(&cudaStreamDestroy)> secondStreamOwner(
        secondStream, &cudaStreamDestroy);
    ASSERT_EQ(cudaSuccess,
              cudaMemcpyAsync(first.source, firstInput.data(), firstInput.size(), cudaMemcpyHostToDevice, firstStream));
    ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(second.source, secondInput.data(), secondInput.size(),
                                           cudaMemcpyHostToDevice, secondStream));
    auto              firstSource       = WrapLifetimeTensor(first.source, nvcv::TYPE_U8);
    auto              firstDestination  = WrapLifetimeTensor(first.destination, nvcv::TYPE_F32);
    auto              secondSource      = WrapLifetimeTensor(second.source, nvcv::TYPE_U8);
    auto              secondDestination = WrapLifetimeTensor(second.destination, nvcv::TYPE_F32);
    cvcuda::ConvertTo op;
    for (int repetition = 0; repetition < 3; ++repetition)
    {
        ASSERT_NO_THROW(op(firstStream, firstSource, firstDestination, 0.123, 0.456));
        ASSERT_NO_THROW(op(secondStream, secondSource, secondDestination, -1.75, 300.5));
    }
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(firstStream));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(secondStream));
    CheckLifetimeGold(first, firstInput, 0.123f, 0.456f, false);
    CheckLifetimeGold(second, secondInput, -1.75f, 300.5f, false);
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(firstStreamOwner.release()));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(secondStreamOwner.release()));
}

TEST(OpConvertToLifetime, submitted_operator_and_wrappers_can_be_destroyed)
{
    LifetimeStorage storage;
    const auto      input = LifetimeInput(71);
    cudaStream_t    stream{};
    ASSERT_EQ(cudaSuccess, cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    std::unique_ptr<std::remove_pointer_t<cudaStream_t>, decltype(&cudaStreamDestroy)> streamOwner(stream,
                                                                                                   &cudaStreamDestroy);
    ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(storage.source, input.data(), input.size(), cudaMemcpyHostToDevice, stream));
    {
        auto              source      = WrapLifetimeTensor(storage.source, nvcv::TYPE_U8);
        auto              destination = WrapLifetimeTensor(storage.destination, nvcv::TYPE_F32);
        cvcuda::ConvertTo op;
        ASSERT_NO_THROW(op(stream, source, destination, 0.123, 0.456));
    }
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    CheckLifetimeGold(storage, input, 0.123f, 0.456f, false);
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(streamOwner.release()));
}

TEST(OpConvertToLifetime, event_ordered_bulk_output_consumed_on_other_stream)
{
    LifetimeStorage storage;
    const auto      input = LifetimeInput(223);
    cudaStream_t    producer{}, consumer{};
    cudaEvent_t     ready{};
    ASSERT_EQ(cudaSuccess, cudaStreamCreateWithFlags(&producer, cudaStreamNonBlocking));
    std::unique_ptr<std::remove_pointer_t<cudaStream_t>, decltype(&cudaStreamDestroy)> producerOwner(
        producer, &cudaStreamDestroy);
    ASSERT_EQ(cudaSuccess, cudaStreamCreateWithFlags(&consumer, cudaStreamNonBlocking));
    std::unique_ptr<std::remove_pointer_t<cudaStream_t>, decltype(&cudaStreamDestroy)> consumerOwner(
        consumer, &cudaStreamDestroy);
    ASSERT_EQ(cudaSuccess, cudaEventCreateWithFlags(&ready, cudaEventDisableTiming));
    std::unique_ptr<std::remove_pointer_t<cudaEvent_t>, decltype(&cudaEventDestroy)> readyOwner(ready,
                                                                                                &cudaEventDestroy);
    ASSERT_EQ(cudaSuccess,
              cudaMemcpyAsync(storage.source, input.data(), input.size(), cudaMemcpyHostToDevice, producer));
    auto              source       = WrapLifetimeTensor(storage.source, nvcv::TYPE_U8);
    auto              intermediate = WrapLifetimeTensor(storage.destination, nvcv::TYPE_F32);
    cvcuda::ConvertTo op;
    ASSERT_NO_THROW(op(producer, source, intermediate, 0.123, 0.456));
    ASSERT_EQ(cudaSuccess, cudaEventRecord(ready, producer));
    ASSERT_EQ(cudaSuccess, cudaStreamWaitEvent(consumer, ready, 0));
    ASSERT_NO_THROW(op(consumer, intermediate, intermediate, 0.5, -4.0));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(consumer));
    CheckLifetimeGold(storage, input, 0.123f, 0.456f, true);
    ASSERT_EQ(cudaSuccess, cudaEventDestroy(readyOwner.release()));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(producerOwner.release()));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(consumerOwner.release()));
}
