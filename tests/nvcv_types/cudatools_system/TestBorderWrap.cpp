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

#include "DeviceBorderWrap.hpp" // to test in the device

#include <common/BorderUtils.hpp> // for test::ReplicateBorderIndex, etc.
#include <common/DeviceBuffer.hpp>
#include <common/Printers.hpp>              // for stream operator, etc.
#include <common/TensorDataUtils.hpp>       // for nvcv::util::CreateTensor, etc.
#include <common/TypedTests.hpp>            // for NVCV_TYPED_TEST_SUITE, etc.
#include <cvcuda/cuda_tools/BorderWrap.hpp> // the object of this test
#include <cvcuda/cuda_tools/MathOps.hpp>    // for operator == to allow EXPECT_EQ
#include <cvcuda/cuda_tools/TensorWrap.hpp> // for Tensor3DWrap, etc.
#include <nvcv/Tensor.hpp>                  // for Tensor, etc.
#include <nvcv/TensorDataAccess.hpp>        // for TensorDataAccessStridedImagePlanar, etc.

#include <algorithm>
#include <array>
#include <numeric>
#include <random>
#include <vector>

namespace cuda  = nvcv::cuda;
namespace test  = nvcv::test;
namespace ttype = nvcv::test::type;

// ---------------------------- Testing IsOutside ------------------------------

#define NVCV_TEST_ROW(SIZE, INDEX, GOLD) ttype::Types<ttype::Value<SIZE>, ttype::Value<INDEX>, ttype::Value<GOLD>>

NVCV_TYPED_TEST_SUITE(IsOutsideTests,
                      ttype::Types<NVCV_TEST_ROW(int{5}, int{3}, false), NVCV_TEST_ROW(int{5}, int{6}, true),
                                   NVCV_TEST_ROW(int{10000000}, int{-1}, true)>);

#undef NVCV_TEST_ROW

TYPED_TEST(IsOutsideTests, correct_boolean)
{
    constexpr auto size  = ttype::GetValue<TypeParam, 0>;
    constexpr auto coord = ttype::GetValue<TypeParam, 1>;
    constexpr auto gold  = ttype::GetValue<TypeParam, 2>;

    EXPECT_EQ(cuda::IsOutside(coord, size), gold);
}

// ----------------------- Testing GetIndexWithBorder --------------------------

#define NVCV_TEST_ROW(BORDERTYPE, BASE, INPUTSIZE, GOLDSIZE, ...)                       \
    ttype::Types<ttype::Value<BORDERTYPE>, ttype::Value<BASE>, ttype::Value<INPUTSIZE>, \
                 ttype::Value<std::array<int, GOLDSIZE>{__VA_ARGS__}>>

NVCV_TYPED_TEST_SUITE(
    GetIndexWithBorderTests,
    ttype::Types<NVCV_TEST_ROW(NVCV_BORDER_REPLICATE, -3, 3, 9, {0, 0, 0, 0, 1, 2, 2, 2, 2}),
                 NVCV_TEST_ROW(NVCV_BORDER_REPLICATE, +10000000, 5, 3, {4, 4, 4}),
                 NVCV_TEST_ROW(NVCV_BORDER_REPLICATE, -10000000, 5, 3, {0, 0, 0}),
                 NVCV_TEST_ROW(NVCV_BORDER_WRAP, -3, 3, 9, {0, 1, 2, 0, 1, 2, 0, 1, 2}),
                 NVCV_TEST_ROW(NVCV_BORDER_WRAP, +10000000, 5, 10, {0, 1, 2, 3, 4, 0, 1, 2, 3, 4}),
                 NVCV_TEST_ROW(NVCV_BORDER_WRAP, -10000000, 5, 10, {0, 1, 2, 3, 4, 0, 1, 2, 3, 4}),
                 NVCV_TEST_ROW(NVCV_BORDER_REFLECT, -3, 3, 9, {2, 1, 0, 0, 1, 2, 2, 1, 0}),
                 NVCV_TEST_ROW(NVCV_BORDER_REFLECT, +10000000, 5, 10, {0, 1, 2, 3, 4, 4, 3, 2, 1, 0}),
                 NVCV_TEST_ROW(NVCV_BORDER_REFLECT, -10000000, 5, 10, {0, 1, 2, 3, 4, 4, 3, 2, 1, 0}),
                 NVCV_TEST_ROW(NVCV_BORDER_REFLECT101, -1, 1, 2, {0, 0}),
                 NVCV_TEST_ROW(NVCV_BORDER_REFLECT101, -3, 3, 9, {1, 2, 1, 0, 1, 2, 1, 0, 1}),
                 NVCV_TEST_ROW(NVCV_BORDER_REFLECT101, +10000000, 5, 9, {0, 1, 2, 3, 4, 3, 2, 1, 0}),
                 NVCV_TEST_ROW(NVCV_BORDER_REFLECT101, -10000000, 5, 9, {0, 1, 2, 3, 4, 3, 2, 1, 0})>);

#undef NVCV_TEST_ROW

TYPED_TEST(GetIndexWithBorderTests, correct_index)
{
    constexpr auto BorderType = ttype::GetValue<TypeParam, 0>;
    constexpr int  kBase      = ttype::GetValue<TypeParam, 1>;
    constexpr int  kInputSize = ttype::GetValue<TypeParam, 2>;
    constexpr auto gold       = ttype::GetValue<TypeParam, 3>;

    std::array<int, gold.size()> test;

    std::iota(test.begin(), test.end(), kBase);

    std::transform( // NOSONAR: std::ranges::transform is C++20.
        test.cbegin(), test.cend(), test.begin(),
        [](const int &coord) { return cuda::GetIndexWithBorder<BorderType>(coord, kInputSize); });

    EXPECT_EQ(test, gold);
}

// --------------------------- Testing BorderWrap ------------------------------

// Shortcuts to easy write each test case

constexpr auto U8      = NVCV_IMAGE_FORMAT_U8;
constexpr auto S16     = NVCV_IMAGE_FORMAT_S16;
constexpr auto _2S16   = NVCV_IMAGE_FORMAT_2S16;
constexpr auto F32     = NVCV_IMAGE_FORMAT_F32;
constexpr auto RGB8    = NVCV_IMAGE_FORMAT_RGB8;
constexpr auto RGBA8   = NVCV_IMAGE_FORMAT_RGBA8;
constexpr auto RGBf32  = NVCV_IMAGE_FORMAT_RGBf32;
constexpr auto RGBAf32 = NVCV_IMAGE_FORMAT_RGBAf32;

#define NVCV_TEST_ROW(WIDTH, HEIGHT, BATCHES, BORDERSIZE, FORMAT, VALUETYPE, BORDERTYPE)                     \
    ttype::Types<ttype::Value<WIDTH>, ttype::Value<HEIGHT>, ttype::Value<BATCHES>, ttype::Value<BORDERSIZE>, \
                 ttype::Value<FORMAT>, VALUETYPE, ttype::Value<BORDERTYPE>>

NVCV_TYPED_TEST_SUITE(BorderWrapNHWTest,
                      ttype::Types<NVCV_TEST_ROW(22, 33, 1, 0, RGBA8, uchar4, NVCV_BORDER_CONSTANT),
                                   NVCV_TEST_ROW(33, 22, 3, 3, _2S16, short2, NVCV_BORDER_CONSTANT),
                                   NVCV_TEST_ROW(11, 44, 5, 55, RGBAf32, float4, NVCV_BORDER_REPLICATE),
                                   NVCV_TEST_ROW(122, 6, 9, 7, RGBf32, float3, NVCV_BORDER_WRAP),
                                   NVCV_TEST_ROW(66, 163, 6, 9, RGB8, uchar3, NVCV_BORDER_REFLECT),
                                   NVCV_TEST_ROW(199, 99, 4, 19, S16, short1, NVCV_BORDER_REFLECT101)>);

TYPED_TEST(BorderWrapNHWTest, correct_fill)
{
    int width      = ttype::GetValue<TypeParam, 0>;
    int height     = ttype::GetValue<TypeParam, 1>;
    int batches    = ttype::GetValue<TypeParam, 2>;
    int borderSize = ttype::GetValue<TypeParam, 3>;

    nvcv::ImageFormat format{ttype::GetValue<TypeParam, 4>};

    using ValueType            = ttype::GetType<TypeParam, 5>;
    constexpr auto kBorderType = ttype::GetValue<TypeParam, 6>;
    using DimType              = int3;

    ValueType borderValue = cuda::SetAll<ValueType>(123);

    int2 bSize{borderSize, borderSize};

    nvcv::Tensor srcTensor = nvcv::util::CreateTensor(batches, width, height, format);
    nvcv::Tensor dstTensor = nvcv::util::CreateTensor(batches, width + borderSize * 2, height + borderSize * 2, format);

    auto srcDev = srcTensor.exportData<nvcv::TensorDataStridedCuda>();
    auto dstDev = dstTensor.exportData<nvcv::TensorDataStridedCuda>();

    ASSERT_NE(srcDev, nullptr);
    ASSERT_NE(dstDev, nullptr);

    auto srcAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*srcDev);
    auto dstAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*dstDev);

    ASSERT_TRUE(srcAccess);
    ASSERT_TRUE(dstAccess);

    DimType srcSize;
    DimType dstSize;

    srcSize.x = srcAccess->numCols();
    dstSize.x = dstAccess->numCols();

    srcSize.y = srcAccess->numRows();
    dstSize.y = dstAccess->numRows();

    srcSize.z = static_cast<int>(srcAccess->numSamples());
    dstSize.z = static_cast<int>(dstAccess->numSamples());

    long3 srcStrides{srcAccess->sampleStride(), srcAccess->rowStride(), srcAccess->colStride()};
    long3 dstStrides{dstAccess->sampleStride(), dstAccess->rowStride(), dstAccess->colStride()};

    if (srcDev->rank() == 3)
    {
        srcStrides.x = srcAccess->numRows() * srcAccess->rowStride();
        dstStrides.x = dstAccess->numRows() * dstAccess->rowStride();
    }

    auto srcSizeBytes = static_cast<int>(srcStrides.x * srcSize.z);
    auto dstSizeBytes = static_cast<int>(dstStrides.x * dstSize.z);

    std::vector<uint8_t> srcVec(srcSizeBytes);

    std::default_random_engine             randEng{0};
    std::uniform_int_distribution<uint8_t> srcRand{0u, 255u};
    std::ranges::generate(srcVec, [&srcRand, &randEng]() { return srcRand(randEng); });

    ASSERT_EQ(cudaSuccess, cudaMemcpy(srcDev->basePtr(), srcVec.data(), srcVec.size(), cudaMemcpyHostToDevice));

    auto srcWrap = cuda::CreateBorderWrapNHW<const ValueType, kBorderType>(*srcDev, borderValue);
    auto dstWrap = cuda::CreateTensorWrapNHW<ValueType>(*dstDev);

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    DeviceRunFillBorder(dstWrap, srcWrap, dstSize, bSize, stream);

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    std::vector<uint8_t> test(dstSizeBytes);
    std::vector<uint8_t> gold(dstSizeBytes);

    // Get test fill border
    ASSERT_EQ(cudaSuccess, cudaMemcpy(test.data(), dstDev->basePtr(), test.size(), cudaMemcpyDeviceToHost));

    // Run gold fill border
    for (int z = 0; z < dstSize.z; ++z)
    {
        int2 srcCoord;

        for (int y = 0; y < dstSize.y; ++y)
        {
            srcCoord.y = y - borderSize;

            for (int x = 0; x < dstSize.x; ++x)
            {
                srcCoord.x = x - borderSize;

                bool isInside = test::IsInside(srcCoord, {width, height}, kBorderType);

                *reinterpret_cast<ValueType *>(&gold[z * dstStrides.x + y * dstStrides.y + x * dstStrides.z])
                    = isInside ? *reinterpret_cast<ValueType *>(
                          &srcVec[z * srcStrides.x + srcCoord.y * srcStrides.y + srcCoord.x * srcStrides.z])
                               : borderValue;
            }
        }
    }

    EXPECT_EQ(test, gold);
}

TEST(BorderWrapNonConstValueTypeTest, it_can_be_read)
{
    using TensorWrap = cuda::TensorWrap<int, -1>;
    using BorderWrap = cuda::BorderWrap<TensorWrap, NVCV_BORDER_CONSTANT, true>;

    EXPECT_TRUE((std::is_same_v<typename BorderWrap::TensorWrapper, TensorWrap>));

    EXPECT_EQ(BorderWrap::kNumDimensions, 1);
    EXPECT_EQ(BorderWrap::kNumActiveDimensions, 1);

    std::array<int, 10> input;
    std::iota(input.begin(), input.end(), 0);

    TensorWrap tensorWrap(input.data(), (int)sizeof(int));
    BorderWrap borderWrap(tensorWrap, -1, 10);

    for (int i = -1; i < 10; ++i)
    {
        EXPECT_EQ(borderWrap[int1{i}], i);
    }
}

TEST(BorderWrapNonConstValueTypeTest, it_can_be_written)
{
    using TensorWrap = cuda::TensorWrap<int, -1>;
    using BorderWrap = cuda::BorderWrap<TensorWrap, NVCV_BORDER_REPLICATE, true>;

    EXPECT_TRUE((std::is_same_v<typename BorderWrap::TensorWrapper, TensorWrap>));

    EXPECT_EQ(BorderWrap::kNumDimensions, 1);
    EXPECT_EQ(BorderWrap::kNumActiveDimensions, 1);

    std::array<int, 10> input;
    std::iota(input.begin(), input.end(), 0);

    TensorWrap tensorWrap(input.data(), (int)sizeof(int));
    BorderWrap borderWrap(tensorWrap, 0, 10);

    borderWrap[-1] = -1;

    EXPECT_EQ(borderWrap[0], -1);
}

// ---------------------- Testing BorderWrap for __half ------------------------

// Host-side smoke test: __half and half3 are not structural types, so they cannot go through
// the ttype::Value<>-driven typed suites above; this proves the wrap layer works with the
// fp16 value types added in HalfTypes.hpp.

TEST(BorderWrapHalfValueTypeTest, it_can_be_read_with_constant_border)
{
    using TensorWrap = cuda::TensorWrap<__half, -1>;
    using BorderWrap = cuda::BorderWrap<TensorWrap, NVCV_BORDER_CONSTANT, true>;

    std::array<__half, 4> input;
    for (std::size_t i = 0; i < input.size(); ++i)
    {
        input[i] = __float2half(static_cast<float>(i));
    }

    TensorWrap tensorWrap(input.data(), (int)sizeof(__half));
    BorderWrap borderWrap(tensorWrap, __float2half(-1.5f), 4);

    for (int i = 0; i < 4; ++i)
    {
        EXPECT_EQ(__half2float(borderWrap[int1{i}]), static_cast<float>(i));
    }

    EXPECT_EQ(__half2float(borderWrap[int1{-1}]), -1.5f);
    EXPECT_EQ(__half2float(borderWrap[int1{4}]), -1.5f);
}

TEST(BorderWrapHalfValueTypeTest, it_can_be_read_with_replicate_border)
{
    using TensorWrap = cuda::TensorWrap<half3, -1>;
    using BorderWrap = cuda::BorderWrap<TensorWrap, NVCV_BORDER_REPLICATE, true>;

    std::array<half3, 3> input;
    for (std::size_t i = 0; i < input.size(); ++i)
    {
        input[i] = cuda::SetAll<half3>(__float2half(static_cast<float>(i)));
    }

    TensorWrap tensorWrap(input.data(), (int)sizeof(half3));
    BorderWrap borderWrap(tensorWrap, half3{}, 3);

    EXPECT_EQ(borderWrap[int1{1}], input[1]);
    EXPECT_EQ(borderWrap[int1{-2}], input[0]);
    EXPECT_EQ(borderWrap[int1{5}], input[2]);
}

// ---------------------------------------------------------------------------------------------
// value(): the by-value constant-border read
// ---------------------------------------------------------------------------------------------

// The host cannot exercise the arch-selected form: ReadConstantBorder branches on
// __CUDA_ARCH__, so a host build compiles one arm and never runs the one that ships. This runs
// the comparison on the device, over several element types for the constant-border specialization
// that exposes value().
template<typename T, NVCVBorderType B>
static void ExpectValueMatchesSubscriptOnDevice()
{
    using TensorWrap = cuda::TensorWrap<const T, -1>;
    using BorderWrap = cuda::BorderWrap<TensorWrap, B, true>;

    constexpr int kNumElems = 4;

    std::array<int, 7> probes{0, 1, 3, -1, -5, 4, 9};

    test::DeviceBuffer<T>   inputBuf{kNumElems};
    test::DeviceBuffer<int> coordsBuf{probes.size()};
    test::DeviceBuffer<int> resultBuf{1};

    T   *devInput  = inputBuf.get();
    int *devCoords = coordsBuf.get();
    int *devResult = resultBuf.get();
    ASSERT_NE(nullptr, devInput);
    ASSERT_NE(nullptr, devCoords);
    ASSERT_NE(nullptr, devResult);

    std::array<T, kNumElems> host;
    for (int i = 0; i < kNumElems; ++i)
    {
        host[i] = cuda::SetAll<T>(static_cast<cuda::BaseType<T>>(i + 1));
    }
    ASSERT_EQ(cudaSuccess, cudaMemcpy(devInput, host.data(), sizeof(host), cudaMemcpyHostToDevice));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(devCoords, probes.data(), probes.size() * sizeof(int), cudaMemcpyHostToDevice));

    TensorWrap tensorWrap(devInput, (int)sizeof(T));
    BorderWrap borderWrap(tensorWrap, cuda::SetAll<T>(static_cast<cuda::BaseType<T>>(42)), kNumElems);

    ASSERT_EQ(cudaSuccess, test::SeedResult(resultBuf.get()));

    test::StreamGuard streamGuard;
    ASSERT_NE(nullptr, streamGuard.get());
    cudaStream_t stream = streamGuard.get();
    DeviceRunValueMatchesSubscript(borderWrap, devCoords, (int)probes.size(), devResult, stream);
    ASSERT_EQ(cudaSuccess, cudaGetLastError());
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    int mismatch = 0;
    ASSERT_EQ(cudaSuccess, cudaMemcpy(&mismatch, devResult, sizeof(int), cudaMemcpyDeviceToHost));
    ASSERT_NE(test::kResultUnwritten, mismatch) << "the parity kernel did not run";
    EXPECT_EQ(0, mismatch) << "border " << (int)B << " at probe " << mismatch;
}

// Both constant-border forms, run on the device. ValueMatchesSubscriptOnDevice* above can only
// ever execute the arm this GPU's __CUDA_ARCH__ selects, so on an sm_80 or sm_90 runner the
// by-reference arm -- the one Blackwell ships -- is compiled and never run. Calling the two
// helpers directly runs both on whatever GPU the suite lands on.
template<typename T>
static void ExpectConstantBorderFormsAgreeOnDevice()
{
    test::DeviceBuffer<T>   insideBuf{1};
    test::DeviceBuffer<int> resultBuf{1};
    ASSERT_NE(nullptr, insideBuf.get());
    ASSERT_NE(nullptr, resultBuf.get());

    const T inside = cuda::SetAll<T>(static_cast<cuda::BaseType<T>>(5));
    const T border = cuda::SetAll<T>(static_cast<cuda::BaseType<T>>(42));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(insideBuf.get(), &inside, sizeof(T), cudaMemcpyHostToDevice));

    ASSERT_EQ(cudaSuccess, test::SeedResult(resultBuf.get()));

    test::StreamGuard streamGuard;
    ASSERT_NE(nullptr, streamGuard.get());
    cudaStream_t stream = streamGuard.get();
    DeviceRunConstantBorderFormsAgree<T>(insideBuf.get(), nullptr, border, resultBuf.get(), stream);
    ASSERT_EQ(cudaSuccess, cudaGetLastError());
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    int mismatch = 0;
    ASSERT_EQ(cudaSuccess, cudaMemcpy(&mismatch, resultBuf.get(), sizeof(int), cudaMemcpyDeviceToHost));
    ASSERT_NE(test::kResultUnwritten, mismatch) << "the parity kernel did not run";
    EXPECT_EQ(0, mismatch) << "the two constant-border forms disagreed on the "
                           << (mismatch == 1 ? "in-bounds probe" : "out-of-bounds probe");
}

// Parity with a 4-D coordinate. doGetPtrForCoord resolves N == 4 through its own arm, and that is
// the arm InterpolationWrap::doGetValue reaches whenever it wraps a 4-D tensor; the 1-D probes
// never enter it.
template<typename T, NVCVBorderType B>
static void ExpectValueMatchesSubscriptOnDevice4D()
{
    using TensorWrap = cuda::Tensor4DWrap<const T>;
    using BorderWrap = cuda::BorderWrap<TensorWrap, B, false, false, true, true>;

    constexpr int kW = 4;
    constexpr int kH = 3;
    constexpr int kP = 2;
    constexpr int kN = 2;

    // Inside, both edges, and outside in x and y; the two inactive dimensions stay valid.
    std::vector<int2> spatial;
    for (int y : {0, 1, 2, -1, -3, 3, 6})
    {
        for (int x : {0, 3, -1, -4, 4, 8})
        {
            spatial.push_back(int2{x, y});
        }
    }

    std::vector<int4> probes;
    for (int n = 0; n < kN; ++n)
    {
        for (int p = 0; p < kP; ++p)
        {
            for (const int2 &xy : spatial)
            {
                probes.push_back(int4{xy.x, xy.y, p, n});
            }
        }
    }

    test::DeviceBuffer<T>    inputBuf{kW * kH * kP * kN};
    test::DeviceBuffer<int4> probesBuf{probes.size()};
    test::DeviceBuffer<int>  resultBuf{1};
    ASSERT_NE(nullptr, inputBuf.get());
    ASSERT_NE(nullptr, probesBuf.get());
    ASSERT_NE(nullptr, resultBuf.get());

    std::array<T, kW * kH * kP * kN> host;
    std::size_t                      next = 0;
    for (T &element : host)
    {
        ++next;
        element = cuda::SetAll<T>(static_cast<cuda::BaseType<T>>(next));
    }
    ASSERT_EQ(cudaSuccess, cudaMemcpy(inputBuf.get(), host.data(), sizeof(host), cudaMemcpyHostToDevice));
    ASSERT_EQ(cudaSuccess,
              cudaMemcpy(probesBuf.get(), probes.data(), probes.size() * sizeof(int4), cudaMemcpyHostToDevice));

    TensorWrap tensorWrap(inputBuf.get(), (int)(kP * kH * kW * sizeof(T)), (int)(kH * kW * sizeof(T)),
                          (int)(kW * sizeof(T)));
    BorderWrap borderWrap(tensorWrap, cuda::SetAll<T>(static_cast<cuda::BaseType<T>>(42)), kH, kW);

    ASSERT_EQ(cudaSuccess, test::SeedResult(resultBuf.get()));

    test::StreamGuard streamGuard;
    ASSERT_NE(nullptr, streamGuard.get());
    cudaStream_t stream = streamGuard.get();
    DeviceRunValueMatchesSubscript4D(borderWrap, probesBuf.get(), (int)probes.size(), resultBuf.get(), stream);
    ASSERT_EQ(cudaSuccess, cudaGetLastError());
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    int mismatch = 0;
    ASSERT_EQ(cudaSuccess, cudaMemcpy(&mismatch, resultBuf.get(), sizeof(int), cudaMemcpyDeviceToHost));
    ASSERT_NE(test::kResultUnwritten, mismatch) << "the parity kernel did not run";
    EXPECT_EQ(0, mismatch) << "border " << (int)B << " at probe " << mismatch;
}

// The device parity kernels resolve both arms through the same doGetPtrForCoord, so they prove the
// two read forms agree, not that the ladder is right. The FillBorder suites above check the ladder
// against an independent host gold for every border mode, and this pins the same expectation
// directly on value() for the 4-D arm those suites do not drive.
TEST(BorderWrapTest, Value4DResolvesTheLadderAndBorder)
{
    using TensorWrap = cuda::Tensor4DWrap<const uchar3>;
    using BorderWrap = cuda::BorderWrap<TensorWrap, NVCV_BORDER_CONSTANT, false, false, true, true>;

    constexpr int kW = 4;
    constexpr int kH = 3;
    constexpr int kP = 2;
    constexpr int kN = 2;

    std::array<uchar3, kW * kH * kP * kN> input;
    for (std::size_t i = 0; i < input.size(); ++i)
    {
        input[i] = cuda::SetAll<uchar3>(static_cast<unsigned char>(i + 1));
    }
    const uchar3 border{17, 18, 19};

    TensorWrap tensorWrap(input.data(), (int)(kP * kH * kW * sizeof(uchar3)), (int)(kH * kW * sizeof(uchar3)),
                          (int)(kW * sizeof(uchar3)));
    BorderWrap borderWrap(tensorWrap, border, kH, kW);

    // int4 is (x, y, plane, sample), and only the two innermost dimensions are active. Walked as
    // one flat loop rather than four nested ones: the coverage is identical and four levels of
    // nesting is past what the static analysis allows.
    for (std::size_t flat = 0; flat < input.size(); ++flat)
    {
        const auto x = static_cast<int>(flat % kW);
        const auto y = static_cast<int>((flat / kW) % kH);
        const auto p = static_cast<int>((flat / (kW * kH)) % kP);
        const auto n = static_cast<int>(flat / (kW * kH * kP));

        EXPECT_EQ(borderWrap.value(int4{x, y, p, n}), input[flat]) << "at " << n << ',' << p << ',' << y << ',' << x;
    }

    // Outside either active dimension the constant border must yield exactly the border value.
    EXPECT_EQ(borderWrap.value(int4{-1, 0, 0, 0}), border);
    EXPECT_EQ(borderWrap.value(int4{kW, 0, 0, 0}), border);
    EXPECT_EQ(borderWrap.value(int4{0, -1, 0, 0}), border);
    EXPECT_EQ(borderWrap.value(int4{0, kH, 1, 1}), border);
}

TEST(BorderWrapTest, ValueMatchesSubscriptOnDeviceWith4DCoords)
{
    ExpectValueMatchesSubscriptOnDevice4D<uchar3, NVCV_BORDER_CONSTANT>();
    ExpectValueMatchesSubscriptOnDevice4D<uchar1, NVCV_BORDER_CONSTANT>();
    ExpectValueMatchesSubscriptOnDevice4D<uchar4, NVCV_BORDER_CONSTANT>();
    ExpectValueMatchesSubscriptOnDevice4D<float4, NVCV_BORDER_CONSTANT>();
}

TEST(BorderWrapTest, ReadConstantBorderFormsAgreeOnDevice)
{
    ExpectConstantBorderFormsAgreeOnDevice<uchar1>();
    ExpectConstantBorderFormsAgreeOnDevice<uchar3>();
    ExpectConstantBorderFormsAgreeOnDevice<uchar4>();
    ExpectConstantBorderFormsAgreeOnDevice<short2>();
    ExpectConstantBorderFormsAgreeOnDevice<float3>();
    ExpectConstantBorderFormsAgreeOnDevice<float4>();
}

// The spill this API exists to avoid is element-width dependent, so cover more than uchar3.
TEST(BorderWrapTest, ValueMatchesSubscriptOnDeviceForEveryElementType)
{
    ExpectValueMatchesSubscriptOnDevice<uchar3, NVCV_BORDER_CONSTANT>();
    ExpectValueMatchesSubscriptOnDevice<uchar1, NVCV_BORDER_CONSTANT>();
    ExpectValueMatchesSubscriptOnDevice<uchar4, NVCV_BORDER_CONSTANT>();
    ExpectValueMatchesSubscriptOnDevice<short2, NVCV_BORDER_CONSTANT>();
    ExpectValueMatchesSubscriptOnDevice<float3, NVCV_BORDER_CONSTANT>();
    ExpectValueMatchesSubscriptOnDevice<float4, NVCV_BORDER_CONSTANT>();
}

// Out of bounds, a constant border must yield exactly the configured border value.
TEST(BorderWrapTest, ValueReturnsBorderValueOutsideConstantBorder)
{
    using TensorWrap = cuda::TensorWrap<const uchar3, -1>;
    using BorderWrap = cuda::BorderWrap<TensorWrap, NVCV_BORDER_CONSTANT, true>;

    std::array<uchar3, 2> input{
        uchar3{1, 2, 3},
        uchar3{4, 5, 6}
    };
    const uchar3 border{17, 18, 19};

    TensorWrap tensorWrap(input.data(), (int)sizeof(uchar3));
    BorderWrap borderWrap(tensorWrap, border, (int)input.size());

    EXPECT_EQ(borderWrap.value(int1{0}), input[0]);
    EXPECT_EQ(borderWrap.value(int1{1}), input[1]);
    EXPECT_EQ(borderWrap.value(int1{-1}), border);
    EXPECT_EQ(borderWrap.value(int1{2}), border);
}
