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

#include <common/TensorDataUtils.hpp>
#include <common/ValueTests.hpp>
#include <cvcuda/OpMorphology.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorDataAccess.hpp>

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace test = nvcv::test;

using uchar = unsigned char;

namespace {

bool NeedsWorkspace(NVCVMorphologyType morphType, int iteration)
{
    return morphType == NVCVMorphologyType::NVCV_OPEN || morphType == NVCVMorphologyType::NVCV_CLOSE || iteration > 1;
}

// Reinterpreting random bytes as halves yields NaNs on ~3% of draws, and NaN min/max ordering is
// operand-order dependent; F16 inputs are therefore generated as finite quantized half values.
void FillRandomInput(std::vector<uint8_t> &vec, const nvcv::ImageFormat &format,
                     std::default_random_engine &rng) // NOSONAR: deterministic test data, not security-sensitive.
{
    if (test::IsF16Format(format))
    {
        test::FillRandomHalfBytes(vec, rng);
    }
    else
    {
        std::uniform_int_distribution<uint32_t> rand(0u, 255u);
        std::ranges::generate(vec, [&rand, &rng]() { return static_cast<uint8_t>(rand(rng)); });
    }
}

} // namespace

class MorphologyTestError : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

template<size_t rows, size_t cols>
using TestImage = std::array<std::array<uchar, cols>, rows>;

static int ScaledSize(int size, double scale)
{
    return static_cast<int>(size * scale);
}

static void hostMorphDilateErode(std::vector<uint8_t> &hDst, const long3 &dstStrides, const std::vector<uint8_t> &hSrc,
                                 const long3 &srcStrides, const int3 &shape, const nvcv::ImageFormat &format,
                                 const nvcv::Size2D &kernelSize, int2 &kernelAnchor, int iterations,
                                 const NVCVBorderType &borderMode, NVCVMorphologyType type)
{
    std::vector<uint8_t> tmpDst;
    tmpDst.reserve(hSrc.size());
    for (int i = 0; i < iterations; i++)
    {
        if (i == 0)
        {
            test::Morph(hDst, dstStrides, hSrc, srcStrides, shape, format, kernelSize, kernelAnchor, borderMode, type);
        }
        else
        {
            tmpDst = hDst;
            test::Morph(hDst, dstStrides, tmpDst, dstStrides, shape, format, kernelSize, kernelAnchor, borderMode,
                        type);
        }
    }
}

static void hostMorph(std::vector<uint8_t> &hDst, const long3 &dstStrides, const std::vector<uint8_t> &hSrc,
                      const long3 &srcStrides, const int3 &shape, const nvcv::ImageFormat &format,
                      const nvcv::Size2D &kernelSize, int2 &kernelAnchor, int iterations,
                      const NVCVBorderType &borderMode, NVCVMorphologyType type)
{
    switch (type)
    {
    case NVCVMorphologyType::NVCV_DILATE:
    case NVCVMorphologyType::NVCV_ERODE:
    {
        hostMorphDilateErode(hDst, dstStrides, hSrc, srcStrides, shape, format, kernelSize, kernelAnchor, iterations,
                             borderMode, type);
        break;
    }
    case NVCVMorphologyType::NVCV_OPEN:
    case NVCVMorphologyType::NVCV_CLOSE:
    {
        NVCVMorphologyType   first  = (type == NVCVMorphologyType::NVCV_OPEN ? NVCVMorphologyType::NVCV_ERODE
                                                                             : NVCVMorphologyType::NVCV_DILATE);
        NVCVMorphologyType   second = (type == NVCVMorphologyType::NVCV_OPEN ? NVCVMorphologyType::NVCV_DILATE
                                                                             : NVCVMorphologyType::NVCV_ERODE);
        std::vector<uint8_t> tmpDst(hSrc.size());
        for (int i = 0; i < iterations; i++)
        {
            if (i == 0)
            {
                test::Morph(tmpDst, dstStrides, hSrc, srcStrides, shape, format, kernelSize, kernelAnchor, borderMode,
                            first);
            }
            else
            {
                test::Morph(tmpDst, dstStrides, hDst, srcStrides, shape, format, kernelSize, kernelAnchor, borderMode,
                            first);
            }
            test::Morph(hDst, dstStrides, tmpDst, srcStrides, shape, format, kernelSize, kernelAnchor, borderMode,
                        second);
        }
        break;
    }

    default:
        throw MorphologyTestError("Unsupported morph type");
        break;
    }
}

TEST(OpMorphology, cpu_gold_open_close_constant_input)
{
    const std::vector<uint8_t> input(20, 128);
    const long3                pitches{20, 5, 1};
    for (NVCVMorphologyType type : {NVCV_OPEN, NVCV_CLOSE})
    {
        SCOPED_TRACE(type);
        std::vector<uint8_t> result(input.size());
        int2                 anchor{1, 1};
        hostMorph(result, pitches, input, pitches, int3{5, 4, 1}, nvcv::FMT_U8, nvcv::Size2D{3, 3}, anchor, 2,
                  NVCV_BORDER_REPLICATE, type);
        EXPECT_EQ(result, input);
    }
}

// checks pixels only in the logical image region.
template<class T>
static bool imagePixelValuesSame(nvcv::util::TensorImageData &a, nvcv::util::TensorImageData &b, int x, int y)
{
    for (int c = 0; c < a.numC(); ++c)
    {
        if (*a.item<T>(x, y, c) != *b.item<T>(x, y, c))
        {
            return false;
        }
    }

    return true;
}

// checks pixels only in the logical image region.
template<class T>
static bool imageRegionValuesSame(nvcv::util::TensorImageData &a, nvcv::util::TensorImageData &b)
{
    int minWidth  = a.size().w > b.size().w ? b.size().w : a.size().w;
    int minHeight = a.size().h > b.size().h ? b.size().h : a.size().h;

    if (a.bytesPerC() != b.bytesPerC() || a.imageCHW() != b.imageCHW() || a.numC() != b.numC())
        return false;

    for (int x = 0; x < minWidth; ++x)
    {
        for (int y = 0; y < minHeight; ++y)
        {
            if (!imagePixelValuesSame<T>(a, b, x, y))
            {
                return false;
            }
        }
    }

    return true;
}

template<class T, size_t rows, size_t cols>
void SetPixelToTestVector(const TestImage<rows, cols> &inputVals, nvcv::util::TensorImageData &data, int x, int y)
{
    for (int c = 0; c < data.numC(); ++c)
    {
        *data.item<T>(x, y, c) = (T)inputVals[y][x];
    }
}

template<class T, size_t rows, size_t cols>
void SetTensorToTestVector(const TestImage<rows, cols> &inputVals, int width, int height, const nvcv::Tensor &tensor,
                           int sample)
{
    nvcv::util::TensorImageData data(tensor.exportData(), sample);

    for (int x = 0; x < width; ++x)
    {
        for (int y = 0; y < height; ++y)
        {
            SetPixelToTestVector<T, rows, cols>(inputVals, data, x, y);
        }
    }

    EXPECT_NO_THROW(nvcv::util::SetTensorFromVector<T>(tensor.exportData(), data.getVector(), sample));
}

template<class T, size_t rows, size_t cols>
bool PixelMatchesTestVector(const TestImage<rows, cols> &checkVals, nvcv::util::TensorImageData &data, int x, int y)
{
    for (int c = 0; c < data.numC(); ++c)
    {
        if (*data.item<T>(x, y, c) != (T)checkVals[y][x])
        {
            return false;
        }
    }

    return true;
}

template<class T, size_t rows, size_t cols>
bool MatchTensorToTestVector(const TestImage<rows, cols> &checkVals, int width, int height, const nvcv::Tensor &tensor,
                             int sample)
{
    nvcv::util::TensorImageData data(tensor.exportData(), sample);
    for (int x = 0; x < width; ++x)
    {
        for (int y = 0; y < height; ++y)
        {
            if (!PixelMatchesTestVector<T, rows, cols>(checkVals, data, x, y))
            {
                return false;
            }
        }
    }

    return true;
}

template<class T, size_t rows, size_t cols>
void checkTestVectors(cudaStream_t &stream, nvcv::Tensor &inTensor, nvcv::Tensor &outTensor,
                      nvcv::OptionalTensorConstRef workspace, const TestImage<rows, cols> &input,
                      const TestImage<rows, cols> &output, int width, int height, const nvcv::Size2D &maskSize,
                      const int2 &anchor, int iteration, NVCVMorphologyType type, NVCVBorderType borderMode,
                      int batches)
{
    for (int i = 0; i < batches; ++i)
    {
        SetTensorToTestVector<uchar, rows, cols>(input, width, height, inTensor, i);
    }

    cvcuda::Morphology morphOp;
    morphOp(stream, inTensor, outTensor, workspace, type, maskSize, anchor, iteration, borderMode);

    if (cudaSuccess != cudaStreamSynchronize(stream))
        throw MorphologyTestError("Cuda Sync failed");

    for (int i = 0; i < batches; ++i)
    {
        if (MatchTensorToTestVector<uchar, rows, cols>(output, width, height, outTensor, i) != true)
        {
            throw MorphologyTestError("Op returned unexpected result");
        }
    }
}

TEST(OpMorphology, morph_check_dilate_kernel)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    constexpr int width   = 5;
    constexpr int height  = 5;
    int           batches = 3;

    nvcv::ImageFormat format{NVCV_IMAGE_FORMAT_U8};

    nvcv::Tensor inTensor        = nvcv::util::CreateTensor(batches, width, height, format);
    nvcv::Tensor outTensor       = nvcv::util::CreateTensor(batches, width, height, format);
    nvcv::Tensor workspaceTensor = nvcv::util::CreateTensor(batches, width, height, format);

    int2               anchor{-1, -1};
    nvcv::Size2D       maskSize(3, 3);
    int                iteration  = 1;
    NVCVMorphologyType type       = NVCVMorphologyType::NVCV_DILATE;
    NVCVBorderType     borderMode = NVCVBorderType::NVCV_BORDER_CONSTANT;

    {
        // clang-format off
        const TestImage<height, width> inImg = {{
                        {0,0,0,0,0},
                        {0,0,0,0,0},
                        {0,0,1,0,0},
                        {0,0,0,0,0},
                        {0,0,0,0,0}
                    }};

        const TestImage<height, width> expImg = {{
                        {0,0,0,0,0},
                        {0,1,1,1,0},
                        {0,1,1,1,0},
                        {0,1,1,1,0},
                        {0,0,0,0,0}
                    }};
        // clang-format on
        EXPECT_NO_THROW((checkTestVectors<uchar, width, height>(
            stream, inTensor, outTensor, nvcv::OptionalTensorConstRef{nvcv::NullOpt}, inImg, expImg, width, height,
            maskSize, anchor, iteration, type, borderMode, batches)));
    }

    // iteration = 2
    {
        // clang-format off
        iteration = 2;
        const TestImage<height, width> inImg = {{
                        {0,0,0,0,0},
                        {0,0,0,0,0},
                        {0,0,1,0,0},
                        {0,0,0,0,0},
                        {0,0,0,0,0}
                    }};

        const TestImage<height, width> expImg = {{
                        {1,1,1,1,1},
                        {1,1,1,1,1},
                        {1,1,1,1,1},
                        {1,1,1,1,1},
                        {1,1,1,1,1}
                    }};
        // clang-format on
        EXPECT_NO_THROW((checkTestVectors<uchar, width, height>(
            stream, inTensor, outTensor, nvcv::OptionalTensorConstRef{workspaceTensor}, inImg, expImg, width, height,
            maskSize, anchor, iteration, type, borderMode, batches)));
    }

    {
        // overlap
        iteration = 1;
        // clang-format off
        const TestImage<height, width> inImg = {{
                        {1,0,0,0,2},
                        {0,0,0,0,0},
                        {0,0,5,0,0},
                        {0,0,0,0,0},
                        {4,0,0,0,3}
                    }};

        const TestImage<height, width> expImg = {{
                        {1,1,0,2,2},
                        {1,5,5,5,2},
                        {0,5,5,5,0},
                        {4,5,5,5,3},
                        {4,4,0,3,3}
                    }};
        // clang-format on
        EXPECT_NO_THROW((checkTestVectors<uchar, width, height>(
            stream, inTensor, outTensor, nvcv::OptionalTensorConstRef{nvcv::NullOpt}, inImg, expImg, width, height,
            maskSize, anchor, iteration, type, borderMode, batches)));
    }

    {
        // mask
        iteration = 1;
        // clang-format off
        maskSize.w = 1;
        maskSize.h = 2;
        const TestImage<height, width> inImg = {{
                        {1,0,0,0,2},
                        {0,0,0,0,0},
                        {0,0,5,0,0},
                        {0,0,0,0,0},
                        {4,0,0,0,3}
                    }};

        const TestImage<height, width> expImg = {{
                        {1,0,0,0,2},
                        {1,0,0,0,2},
                        {0,0,5,0,0},
                        {0,0,5,0,0},
                        {4,0,0,0,3}
                    }};
        // clang-format on
        EXPECT_NO_THROW((checkTestVectors<uchar, width, height>(
            stream, inTensor, outTensor, nvcv::OptionalTensorConstRef{nvcv::NullOpt}, inImg, expImg, width, height,
            maskSize, anchor, iteration, type, borderMode, batches)));
        maskSize.w = 3;
        maskSize.h = 3;
    }

    {
        // anchor
        iteration = 1;
        // clang-format off
        anchor.x = 0;
        anchor.y = 0;

        const TestImage<height, width> inImg = {{
                        {0,0,0,0,0},
                        {0,0,0,0,0},
                        {0,0,1,0,0},
                        {0,0,0,0,0},
                        {0,0,0,0,0}
                    }};

        const TestImage<height, width> expImg = {{
                        {1,1,1,0,0},
                        {1,1,1,0,0},
                        {1,1,1,0,0},
                        {0,0,0,0,0},
                        {0,0,0,0,0}
                    }};
        // clang-format on
        EXPECT_NO_THROW((checkTestVectors<uchar, width, height>(
            stream, inTensor, outTensor, nvcv::OptionalTensorConstRef{nvcv::NullOpt}, inImg, expImg, width, height,
            maskSize, anchor, iteration, type, borderMode, batches)));
        anchor.x = -1;
        anchor.y = -1;
    }

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

// The var-shape interleaved and planar callers seed their running maximum and constant border from
// MorphIdentityValue independently of the tensor path above, so each needs its own guard against
// the identity regressing to a positive value.
static void TestVarShapeFloatDilationPreservesAllNegativeInput(nvcv::ImageFormat format)
{
    constexpr int            width  = 4;
    constexpr int            height = 1;
    const std::vector<float> input{-4.f, -3.f, -2.f, -1.f};
    const std::vector<float> expected{-3.f, -2.f, -1.f, -1.f};

    std::vector<nvcv::Image> inImages;
    std::vector<nvcv::Image> outImages;
    inImages.emplace_back(nvcv::Size2D{width, height}, format);
    outImages.emplace_back(nvcv::Size2D{width, height}, format);

    auto inData  = inImages[0].exportData<nvcv::ImageDataStridedCuda>();
    auto outData = outImages[0].exportData<nvcv::ImageDataStridedCuda>();
    ASSERT_TRUE(inData && outData);

    const int numPlanes = inData->numPlanes();
    for (int p = 0; p < numPlanes; ++p)
    {
        ASSERT_EQ(cudaSuccess, cudaMemcpy(inData->plane(p).basePtr, input.data(), input.size() * sizeof(float),
                                          cudaMemcpyHostToDevice));
    }

    nvcv::ImageBatchVarShape inBatch(1);
    inBatch.pushBack(inImages.begin(), inImages.end());
    nvcv::ImageBatchVarShape outBatch(1);
    outBatch.pushBack(outImages.begin(), outImages.end());

    auto masks   = test::planar::MakePerImageTensor(1, nvcv::TYPE_2S32, int2{3, 1});
    auto anchors = test::planar::MakePerImageTensor(1, nvcv::TYPE_2S32, int2{-1, -1});

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    cvcuda::Morphology op;
    ASSERT_NO_THROW(op(stream, inBatch, outBatch, nvcv::OptionalImageBatchVarShapeConstRef{nvcv::NullOpt}, NVCV_DILATE,
                       masks, anchors, 1, NVCV_BORDER_CONSTANT));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    for (int p = 0; p < numPlanes; ++p)
    {
        std::vector<float> actual(input.size());
        ASSERT_EQ(cudaSuccess, cudaMemcpy(actual.data(), outData->plane(p).basePtr, actual.size() * sizeof(float),
                                          cudaMemcpyDeviceToHost));
        EXPECT_EQ(expected, actual) << "plane " << p;
    }

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpMorphology, varshape_float_dilation_preserves_all_negative_input)
{
    TestVarShapeFloatDilationPreservesAllNegativeInput(nvcv::FMT_F32);
}

TEST(OpMorphologyPlanar, varshape_float_dilation_preserves_all_negative_input)
{
    TestVarShapeFloatDilationPreservesAllNegativeInput(nvcv::FMT_RGBf32p);
}

TEST(OpMorphology, morph_check_erode_kernel)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    constexpr int width   = 5;
    constexpr int height  = 5;
    int           batches = 3;

    nvcv::ImageFormat format{NVCV_IMAGE_FORMAT_U8};

    nvcv::Tensor inTensor  = nvcv::util::CreateTensor(batches, width, height, format);
    nvcv::Tensor outTensor = nvcv::util::CreateTensor(batches, width, height, format);

    int2               anchor{-1, -1};
    nvcv::Size2D       maskSize(3, 3);
    int                iteration  = 1;
    NVCVMorphologyType type       = NVCVMorphologyType::NVCV_ERODE;
    NVCVBorderType     borderMode = NVCVBorderType::NVCV_BORDER_CONSTANT;

    {
        // clang-format off
        const TestImage<height, width> inImg = {{
                        {0,0,0,0,0},
                        {0,1,1,1,0},
                        {0,1,1,1,0},
                        {0,1,1,1,0},
                        {0,0,0,0,0}
                    }};

        const TestImage<height, width> expImg = {{
                        {0,0,0,0,0},
                        {0,0,0,0,0},
                        {0,0,1,0,0},
                        {0,0,0,0,0},
                        {0,0,0,0,0}
                    }};
        // clang-format on
        EXPECT_NO_THROW((checkTestVectors<uchar, width, height>(
            stream, inTensor, outTensor, nvcv::OptionalTensorConstRef{nvcv::NullOpt}, inImg, expImg, width, height,
            maskSize, anchor, iteration, type, borderMode, batches)));
    }

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpMorphology, morph_check_dilate_kernel_even)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    constexpr int width   = 6;
    constexpr int height  = 6;
    int           batches = 3;

    nvcv::ImageFormat format{NVCV_IMAGE_FORMAT_U8};

    nvcv::Tensor inTensor  = nvcv::util::CreateTensor(batches, width, height, format);
    nvcv::Tensor outTensor = nvcv::util::CreateTensor(batches, width, height, format);

    int2               anchor{-1, -1};
    nvcv::Size2D       maskSize(3, 3);
    int                iteration  = 1;
    NVCVMorphologyType type       = NVCVMorphologyType::NVCV_DILATE;
    NVCVBorderType     borderMode = NVCVBorderType::NVCV_BORDER_CONSTANT;

    {
        // clang-format off
        const TestImage<height, width> inImg = {{
                        {1,0,0,0,0,2},
                        {0,0,0,0,0,0},
                        {0,0,5,0,0,0},
                        {0,0,0,0,0,0},
                        {0,0,0,0,0,0},
                        {4,0,0,0,0,3}
                    }};

        const TestImage<height, width> expImg = {{
                        {1,1,0,0,2,2},
                        {1,5,5,5,2,2},
                        {0,5,5,5,0,0},
                        {0,5,5,5,0,0},
                        {4,4,0,0,3,3},
                        {4,4,0,0,3,3}
                    }};

        EXPECT_NO_THROW((checkTestVectors<uchar,width, height>(stream, inTensor, outTensor, nvcv::OptionalTensorConstRef{nvcv::NullOpt}, inImg, expImg, width, height, maskSize,anchor,iteration, type, borderMode, batches)));
    }
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

// Dilation seeds its running maximum with the identity of max(), which must be the *lowest*
// representable value of the element type. Both morphology paths used std::numeric_limits<T>::min()
// instead, which for a floating-point T is the smallest positive normal (~1.18e-38 for float), not
// the most negative value. Unsigned types hid the defect (min() == lowest() == 0) and the f16 arm
// already used HalfLowest(), so only f32 was affected: wherever every sample in the mask window was
// negative, the seed won every comparison and dilate emitted that positive constant -- a value
// present nowhere in the input.
//
// The image here is entirely negative and the border is replicated, so no constant-border value
// participates and the seed is the only way a positive number can reach the output. The expectation
// is computed locally, deliberately not through test::Morph: that CPU gold seeded its fold the same
// wrong way, so it agreed with the wrong answer and the whole parametrized F32 suite stayed green.
// It is corrected in the same change, but a regression test for a shared-seed defect must not
// depend on the thing that shared the seed.
TEST(OpMorphology, dilate_f32_all_negative_neighborhood)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    constexpr int width   = 7;
    constexpr int height  = 5;
    constexpr int batches = 1;

    nvcv::ImageFormat format{NVCV_IMAGE_FORMAT_F32};

    nvcv::Tensor inTensor  = nvcv::util::CreateTensor(batches, width, height, format);
    nvcv::Tensor outTensor = nvcv::util::CreateTensor(batches, width, height, format);

    auto inData  = inTensor.exportData<nvcv::TensorDataStridedCuda>();
    auto outData = outTensor.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_NE(inData, nullptr);
    ASSERT_NE(outData, nullptr);

    auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*inData);
    ASSERT_TRUE(inAccess);
    auto outAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*outData);
    ASSERT_TRUE(outAccess);

    auto srcValue = [](int x, int y) { return -1.0f - static_cast<float>(y * width + x); };

    constexpr size_t     hostRowBytes = width * sizeof(float);
    std::vector<float>   inVec(static_cast<size_t>(width) * height);
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            inVec[y * width + x] = srcValue(x, y);
        }
    }
    ASSERT_EQ(cudaSuccess, cudaMemcpy2D(inData->basePtr(), inAccess->rowStride(), inVec.data(), hostRowBytes,
                                        hostRowBytes, height, cudaMemcpyHostToDevice));

    cvcuda::Morphology morphOp;
    EXPECT_NO_THROW(morphOp(stream, inTensor, outTensor, nvcv::OptionalTensorConstRef{nvcv::NullOpt},
                            NVCVMorphologyType::NVCV_DILATE, nvcv::Size2D{3, 3}, int2{-1, -1}, 1,
                            NVCV_BORDER_REPLICATE));

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    std::vector<float> testVec(static_cast<size_t>(width) * height);
    ASSERT_EQ(cudaSuccess, cudaMemcpy2D(testVec.data(), hostRowBytes, outData->basePtr(), outAccess->rowStride(),
                                        hostRowBytes, height, cudaMemcpyDeviceToHost));

    auto dilate3x3 = [&](int x, int y)
    {
        float maxVal = srcValue(x, y);
        for (int dy = -1; dy <= 1; ++dy)
        {
            for (int dx = -1; dx <= 1; ++dx)
            {
                maxVal
                    = std::max(maxVal, srcValue(std::clamp(x + dx, 0, width - 1), std::clamp(y + dy, 0, height - 1)));
            }
        }
        return maxVal;
    };

    std::vector<float> goldVec(static_cast<size_t>(width) * height);
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            goldVec[y * width + x] = dilate3x3(x, y);
        }
    }

    EXPECT_EQ(testVec, goldVec);
}

TEST(OpMorphology, tensor_identity_covers_f16_and_f32_dilate_erode)
{
    constexpr int            width   = 7;
    constexpr int            height  = 5;
    constexpr int            batches = 1;
    const nvcv::Size2D       maskSize{3, 3};
    const int2               anchor{1, 1};
    const NVCVBorderType     borderMode = NVCV_BORDER_CONSTANT;
    const int3               shape{width, height, batches};
    cudaStream_t             stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));
    cvcuda::Morphology morphOp;

    auto runFormat = [&](nvcv::ImageFormat format)
    {
        nvcv::Tensor inTensor  = nvcv::util::CreateTensor(batches, width, height, format);
        nvcv::Tensor outTensor = nvcv::util::CreateTensor(batches, width, height, format);

        auto inData  = inTensor.exportData<nvcv::TensorDataStridedCuda>();
        auto outData = outTensor.exportData<nvcv::TensorDataStridedCuda>();
        ASSERT_NE(inData, nullptr);
        ASSERT_NE(outData, nullptr);

        auto inAccess  = nvcv::TensorDataAccessStridedImagePlanar::Create(*inData);
        auto outAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*outData);
        ASSERT_TRUE(inAccess);
        ASSERT_TRUE(outAccess);

        long3 inStrides{inAccess->sampleStride(), inAccess->rowStride(), inAccess->colStride()};
        long3 outStrides{outAccess->sampleStride(), outAccess->rowStride(), outAccess->colStride()};
        if (inData->rank() == 3)
        {
            inStrides.x  = inAccess->numRows() * inAccess->rowStride();
            outStrides.x = outAccess->numRows() * outAccess->rowStride();
        }

        const auto inBytes  = static_cast<size_t>(inStrides.x * inAccess->numSamples());
        const auto outBytes = static_cast<size_t>(outStrides.x * outAccess->numSamples());
        std::vector<uint8_t> input(inBytes);
        std::default_random_engine rng(0); // NOSONAR: deterministic test data, not security-sensitive.
        if (format == nvcv::FMT_F32)
        {
            constexpr float value = -4.f;
            for (size_t offset = 0; offset < input.size(); offset += sizeof(value))
            {
                std::memcpy(input.data() + offset, &value, sizeof(value));
            }
        }
        else
        {
            FillRandomInput(input, format, rng);
        }
        ASSERT_EQ(cudaSuccess, cudaMemcpy(inData->basePtr(), input.data(), input.size(), cudaMemcpyHostToDevice));

        auto copyFloatPixels = [&](std::vector<uint8_t> &expected)
        {
            for (int batch = 0; batch < batches; ++batch)
            {
                for (int y = 0; y < height; ++y)
                {
                    for (int x = 0; x < width; ++x)
                    {
                        std::memcpy(expected.data() + batch * outStrides.x + y * outStrides.y + x * outStrides.z,
                                    input.data() + batch * inStrides.x + y * inStrides.y + x * inStrides.z,
                                    sizeof(float));
                    }
                }
            }
        };

        auto runMorphType = [&](NVCVMorphologyType morphType)
        {
            std::vector<uint8_t> expected(outBytes, 0xA5);
            int2                 kernelAnchor = anchor;
            if (format == nvcv::FMT_F32)
            {
                copyFloatPixels(expected);
            }
            else
            {
                hostMorph(expected, outStrides, input, inStrides, shape, format, maskSize, kernelAnchor, 1,
                          borderMode, morphType);
            }

            ASSERT_EQ(cudaSuccess, cudaMemset(outData->basePtr(), 0xA5, outBytes));
            EXPECT_NO_THROW(morphOp(stream, inTensor, outTensor, nvcv::OptionalTensorConstRef{nvcv::NullOpt},
                                    morphType, maskSize, anchor, 1, borderMode));
            ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

            std::vector<uint8_t> actual(outBytes);
            ASSERT_EQ(cudaSuccess, cudaMemcpy(actual.data(), outData->basePtr(), actual.size(), cudaMemcpyDeviceToHost));
            EXPECT_EQ(expected, actual) << (test::IsF16Format(format) ? "F16" : "F32") << " morphology type "
                                        << static_cast<int>(morphType);
        };

        for (const auto morphType : {NVCV_DILATE, NVCV_ERODE})
        {
            ASSERT_NO_FATAL_FAILURE(runMorphType(morphType));
        }
    };

    for (const nvcv::ImageFormat format : {nvcv::FMT_F16, nvcv::FMT_F32})
    {
        ASSERT_NO_FATAL_FAILURE(runFormat(format));
    }

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

// A one-pixel negative RGBA sample makes the centered constant-border identity directly observable
// for both min and max reductions while exercising scalar and packed element dispatch.
TEST(OpMorphologyPlanar, tensor_rgba_identity_covers_f16_f32_dilate_erode)
{
    auto runCase = [](nvcv::ImageFormat planarFmt, nvcv::ImageFormat interleavedFmt,
                      NVCVMorphologyType morphType)
    {
        const int elemSize = planarFmt.planePixelStrideBytes(0);
        auto      makeInput = [elemSize](int, int width, int height, int channels)
        {
            const std::vector<float> values(static_cast<size_t>(width) * height * channels, -4.f);
            if (elemSize == static_cast<int>(sizeof(__half)))
            {
                return test::FloatToHalfBytes(values);
            }

            std::vector<uint8_t> bytes(values.size() * sizeof(float));
            std::memcpy(bytes.data(), values.data(), bytes.size());
            return bytes;
        };
        const std::vector<uint8_t> expected = makeInput(0, 1, 1, planarFmt.numChannels());
        auto compare = [&expected](const std::vector<uint8_t> &interleaved, const std::vector<uint8_t> &planar)
        {
            EXPECT_EQ(expected, interleaved);
            EXPECT_EQ(expected, planar);
        };

        test::planar::RunTensorParity(
            planarFmt, interleavedFmt, 1, 1, 1, 1, 1,
            [morphType](cudaStream_t stream, const nvcv::Tensor &src, const nvcv::Tensor &dst, nvcv::ImageFormat)
            {
                cvcuda::Morphology op;
                EXPECT_NO_THROW(op(stream, src, dst, nvcv::OptionalTensorConstRef{nvcv::NullOpt}, morphType,
                                    nvcv::Size2D{3, 3}, int2{1, 1}, 1, NVCV_BORDER_CONSTANT));
            },
            makeInput, compare);
    };

    for (const auto morphType : {NVCV_DILATE, NVCV_ERODE})
    {
        runCase(nvcv::FMT_RGBAf16p, nvcv::FMT_RGBAf16, morphType);
        runCase(nvcv::FMT_RGBAf32p, nvcv::FMT_RGBAf32, morphType);
    }
}

TEST(OpMorphologyPlanar, varshape_identity_covers_f16_f32_dilate_erode)
{
    auto runCase = [](nvcv::ImageFormat planarFmt, nvcv::ImageFormat interleavedFmt,
                      NVCVMorphologyType morphType)
    {
        test::planar::RunVarShapeParity(
            planarFmt, interleavedFmt, 17, 13, 17, 13, 1,
            [morphType](cudaStream_t stream, const nvcv::ImageBatchVarShape &src,
                        const nvcv::ImageBatchVarShape &dst, nvcv::ImageFormat)
            {
                auto masks   = test::planar::MakePerImageTensor(1, nvcv::TYPE_2S32, int2{3, 3});
                auto anchors = test::planar::MakePerImageTensor(1, nvcv::TYPE_2S32, int2{1, 1});

                cvcuda::Morphology op;
                EXPECT_NO_THROW(op(stream, src, dst, nvcv::OptionalImageBatchVarShapeConstRef{nvcv::NullOpt},
                                    morphType, masks, anchors, 1, NVCV_BORDER_CONSTANT));
            });
    };

    for (const auto morphType : {NVCV_DILATE, NVCV_ERODE})
    {
        runCase(nvcv::FMT_RGBf16p, nvcv::FMT_RGBf16, morphType);
        runCase(nvcv::FMT_RGBf32p, nvcv::FMT_RGBf32, morphType);
    }
}

// clang-format off
NVCV_TEST_SUITE_P(OpMorphology, test::ValueList<int, int, int, NVCVImageFormat, int, int, NVCVBorderType, NVCVMorphologyType, int>
{
    // width, height, batches,               format,  maskWidth, maskHeight,             borderMode,  morphType,    iteration
    {      5,      5,       1, NVCV_IMAGE_FORMAT_U8,          2,          2,   NVCV_BORDER_CONSTANT, NVCV_ERODE,            3},
    {      5,      5,       1, NVCV_IMAGE_FORMAT_U8,         -1,         -1,   NVCV_BORDER_CONSTANT, NVCV_DILATE,           1},
    {      5,      5,       1, NVCV_IMAGE_FORMAT_RGBAf32,     3,          3,   NVCV_BORDER_CONSTANT, NVCV_DILATE,           1},
    {     25,     45,       2, NVCV_IMAGE_FORMAT_U8,          3,          3,   NVCV_BORDER_CONSTANT, NVCV_DILATE,           2},
    {    125,     35,       1, NVCV_IMAGE_FORMAT_RGBA8,       4,          4,   NVCV_BORDER_CONSTANT, NVCV_ERODE,            1},
    {     52,     45,       1, NVCV_IMAGE_FORMAT_U16,         3,          3,   NVCV_BORDER_CONSTANT, NVCV_ERODE,            1},
    {    325,     45,       3, NVCV_IMAGE_FORMAT_RGB8,        3,          2,   NVCV_BORDER_CONSTANT, NVCV_DILATE,           10},
    {     25,     45,       1, NVCV_IMAGE_FORMAT_U8,          3,          3,   NVCV_BORDER_CONSTANT, NVCV_ERODE,            1},
    {     25,     45,       2, NVCV_IMAGE_FORMAT_U8,          3,          3,   NVCV_BORDER_CONSTANT, NVCV_DILATE,           3},
    {     25,     45,       2, NVCV_IMAGE_FORMAT_U8,          3,          3,   NVCV_BORDER_CONSTANT, NVCV_OPEN,             3},
    {     25,     45,       1, NVCV_IMAGE_FORMAT_U8,          3,          3,   NVCV_BORDER_CONSTANT, NVCV_CLOSE,            1},
    {     13,      5,       1, NVCV_IMAGE_FORMAT_U8,          2,          2,   NVCV_BORDER_CONSTANT, NVCV_OPEN,             3},
    {    345,      5,       1, NVCV_IMAGE_FORMAT_RGBAf32,     3,          3,   NVCV_BORDER_CONSTANT, NVCV_CLOSE,            1},
    {    217,    451,       2, NVCV_IMAGE_FORMAT_U8,          3,          3,   NVCV_BORDER_CONSTANT, NVCV_CLOSE,            2},
    {    125,     35,       1, NVCV_IMAGE_FORMAT_RGBA8,       4,          4,   NVCV_BORDER_CONSTANT, NVCV_OPEN,             1},
    {     52,     87,       1, NVCV_IMAGE_FORMAT_U16,         3,          3,   NVCV_BORDER_CONSTANT, NVCV_OPEN,             1},
    {    325,    800,       3, NVCV_IMAGE_FORMAT_RGB8,        3,          2,   NVCV_BORDER_CONSTANT, NVCV_OPEN,            10},
    {     25,     44,       1, NVCV_IMAGE_FORMAT_U8,          3,          3,   NVCV_BORDER_CONSTANT, NVCV_CLOSE,            1},
    {     21,    435,       2, NVCV_IMAGE_FORMAT_U8,          3,          3,   NVCV_BORDER_CONSTANT, NVCV_CLOSE,            3},
    {      5,      5,       1, NVCV_IMAGE_FORMAT_U8,          2,          2,   NVCV_BORDER_REPLICATE, NVCV_ERODE,           3},
    {     25,     45,       2, NVCV_IMAGE_FORMAT_U8,          3,          3,   NVCV_BORDER_REPLICATE, NVCV_DILATE,          2},
    {     25,     45,       2, NVCV_IMAGE_FORMAT_U8,          3,          3,   NVCV_BORDER_REPLICATE, NVCV_OPEN,            3},
    {     25,     44,       2, NVCV_IMAGE_FORMAT_U8,          3,          3,   NVCV_BORDER_REPLICATE, NVCV_CLOSE,           2},
    {      5,      5,       1, NVCV_IMAGE_FORMAT_U8,          2,          2,   NVCV_BORDER_REFLECT, NVCV_ERODE,             3},
    {     25,     45,       2, NVCV_IMAGE_FORMAT_U8,          3,          3,   NVCV_BORDER_REFLECT, NVCV_DILATE,            2},
    {     25,     45,       2, NVCV_IMAGE_FORMAT_U8,          3,          3,   NVCV_BORDER_REFLECT, NVCV_OPEN,              3},
    {     25,     44,       2, NVCV_IMAGE_FORMAT_U8,          3,          3,   NVCV_BORDER_REFLECT, NVCV_CLOSE,             2},
    {      5,      5,       1, NVCV_IMAGE_FORMAT_U8,          2,          2,   NVCV_BORDER_WRAP, NVCV_ERODE,                3},
    {     25,     45,       2, NVCV_IMAGE_FORMAT_U8,          3,          3,   NVCV_BORDER_WRAP, NVCV_DILATE,               2},
    {     25,     45,       2, NVCV_IMAGE_FORMAT_U8,          3,          3,   NVCV_BORDER_WRAP, NVCV_OPEN,                 3},
    {     25,     44,       2, NVCV_IMAGE_FORMAT_U8,          3,          3,   NVCV_BORDER_WRAP, NVCV_CLOSE,                2},
    {      5,      5,       1, NVCV_IMAGE_FORMAT_U8,          2,          2,   NVCV_BORDER_REFLECT101, NVCV_ERODE,          3},
    {     25,     45,       2, NVCV_IMAGE_FORMAT_U8,          3,          3,   NVCV_BORDER_REFLECT101, NVCV_DILATE,         2},
    {     25,     45,       2, NVCV_IMAGE_FORMAT_U8,          3,          3,   NVCV_BORDER_REFLECT101, NVCV_OPEN,           3},
    {     25,     44,       2, NVCV_IMAGE_FORMAT_U8,          3,          3,   NVCV_BORDER_REFLECT101, NVCV_CLOSE,          2},
    {      5,      5,       1, NVCV_IMAGE_FORMAT_RGBAf16,     3,          3,   NVCV_BORDER_CONSTANT, NVCV_DILATE,           1},
    {     25,     45,       2, NVCV_IMAGE_FORMAT_RGBf16,      3,          3,   NVCV_BORDER_REPLICATE, NVCV_ERODE,           1},
    {    345,      5,       1, NVCV_IMAGE_FORMAT_RGBAf16,     3,          3,   NVCV_BORDER_CONSTANT, NVCV_CLOSE,            1},
    {     25,     45,       2, NVCV_IMAGE_FORMAT_F16,         3,          3,   NVCV_BORDER_REFLECT, NVCV_OPEN,              2}
});

// clang-format on

TEST_P(OpMorphology, morph_noop)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    int                width      = GetParamValue<0>();
    int                height     = GetParamValue<1>();
    int                batches    = GetParamValue<2>();
    NVCVBorderType     borderMode = GetParamValue<6>();
    NVCVMorphologyType morphType  = GetParamValue<7>();

    // do not check noop on open/close since it needs workspace.
    if (morphType == NVCVMorphologyType::NVCV_OPEN || morphType == NVCVMorphologyType::NVCV_CLOSE)
        return;

    nvcv::ImageFormat format{NVCV_IMAGE_FORMAT_U8};

    nvcv::Tensor inTensor  = nvcv::util::CreateTensor(batches, width, height, format);
    nvcv::Tensor outTensor = nvcv::util::CreateTensor(batches, width, height, format);

    EXPECT_NO_THROW(nvcv::util::SetTensorToRandomValue<uint8_t>(inTensor.exportData(), 0, 0xFF));
    EXPECT_NO_THROW(nvcv::util::SetTensorTo<uint8_t>(outTensor.exportData(), 0));

    cvcuda::Morphology morphOp;
    int2               anchor{0, 0};

    nvcv::Size2D maskSize(1, 1);
    int          iteration = 0;

    EXPECT_NO_THROW(morphOp(stream, inTensor, outTensor, nvcv::OptionalTensorConstRef{nvcv::NullOpt}, morphType,
                            maskSize, anchor, iteration, borderMode));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    for (int i = 0; i < batches; ++i)
    {
        nvcv::util::TensorImageData cvTensorIn(inTensor.exportData());
        nvcv::util::TensorImageData cvTensorOut(outTensor.exportData());
        EXPECT_TRUE(imageRegionValuesSame<uint8_t>(cvTensorIn, cvTensorOut));
    }

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST_P(OpMorphology, morph_random)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    int width   = GetParamValue<0>();
    int height  = GetParamValue<1>();
    int batches = GetParamValue<2>();

    nvcv::ImageFormat format{GetParamValue<3>()};

    nvcv::Size2D maskSize;
    maskSize.w                    = GetParamValue<4>();
    maskSize.h                    = GetParamValue<5>();
    NVCVBorderType     borderMode = GetParamValue<6>();
    NVCVMorphologyType morphType  = GetParamValue<7>();
    int                iteration  = GetParamValue<8>();

    int3 shape{width, height, batches};

    nvcv::Tensor inTensor        = nvcv::util::CreateTensor(batches, width, height, format);
    nvcv::Tensor outTensor       = nvcv::util::CreateTensor(batches, width, height, format);
    nvcv::Tensor workspaceTensor = nvcv::util::CreateTensor(batches, width, height, format);

    auto inData  = inTensor.exportData<nvcv::TensorDataStridedCuda>();
    auto outData = outTensor.exportData<nvcv::TensorDataStridedCuda>();

    ASSERT_NE(inData, nullptr);
    ASSERT_NE(outData, nullptr);

    auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*inData);
    ASSERT_TRUE(inAccess);

    auto outAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*outData);
    ASSERT_TRUE(outAccess);

    long3 inStrides{inAccess->sampleStride(), inAccess->rowStride(), inAccess->colStride()};
    long3 outStrides{outAccess->sampleStride(), outAccess->rowStride(), outAccess->colStride()};

    if (inData->rank() == 3)
    {
        inStrides.x  = inAccess->numRows() * inAccess->rowStride();
        outStrides.x = outAccess->numRows() * outAccess->rowStride();
    }

    long inBufSize  = inStrides.x * inAccess->numSamples();
    long outBufSize = outStrides.x * outAccess->numSamples();

    std::vector<uint8_t> inVec(inBufSize);

    std::default_random_engine randEng(0); // NOSONAR: deterministic test data, not security-sensitive.

    FillRandomInput(inVec, format, randEng);

    // copy random input to device
    ASSERT_EQ(cudaSuccess, cudaMemcpy(inData->basePtr(), inVec.data(), inBufSize, cudaMemcpyHostToDevice));

    // run operator
    cvcuda::Morphology morphOp;
    int2               anchor{-1, -1};

    EXPECT_NO_THROW(morphOp(stream, inTensor, outTensor, nvcv::OptionalTensorConstRef{workspaceTensor}, morphType,
                            maskSize, anchor, iteration, borderMode));

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    std::vector<uint8_t> goldVec(outBufSize);
    std::vector<uint8_t> testVec(outBufSize);

    // copy output back to host
    ASSERT_EQ(cudaSuccess, cudaMemcpy(testVec.data(), outData->basePtr(), outBufSize, cudaMemcpyDeviceToHost));

    // generate gold result
    if (maskSize.w == -1 || maskSize.h == -1)
    {
        maskSize.w = 3;
        maskSize.h = 3;
    }
    int2 kernelAnchor{maskSize.w / 2, maskSize.h / 2};
    hostMorph(goldVec, outStrides, inVec, inStrides, shape, format, maskSize, kernelAnchor, iteration, borderMode,
              morphType);

    EXPECT_EQ(testVec, goldVec);
}

// Planar (NCHW/CHW) layout support

// Parameters: width, height, maskWidth, maskHeight, borderMode, morphType, iteration, numImages, planarFmt,
// interleavedFmt
// clang-format off
NVCV_TEST_SUITE_P(OpMorphologyPlanar,
                  test::ValueList<int, int, int, int, NVCVBorderType, NVCVMorphologyType, int, int,
                                  nvcv::ImageFormat, nvcv::ImageFormat>{
    {64, 48, 3, 3,    NVCV_BORDER_CONSTANT, NVCV_ERODE,  1, 2,  nvcv::FMT_RGB8p,    nvcv::FMT_RGB8},
    {35, 29, 3, 3,   NVCV_BORDER_REPLICATE, NVCV_DILATE, 1, 1,  nvcv::FMT_RGB8p,    nvcv::FMT_RGB8},
    {37, 31, 5, 3,      NVCV_BORDER_REFLECT, NVCV_DILATE, 1, 1,  nvcv::FMT_RGB8p,    nvcv::FMT_RGB8},
    {39, 33, 5, 3,   NVCV_BORDER_REFLECT101, NVCV_ERODE,  2, 1,  nvcv::FMT_RGB8p,    nvcv::FMT_RGB8},
    {61, 47, 5, 3,      NVCV_BORDER_REFLECT, NVCV_DILATE, 2, 1,  nvcv::FMT_RGB8p,    nvcv::FMT_RGB8},
    {34, 30, 3, 3,       NVCV_BORDER_CONSTANT, NVCV_CLOSE, 1, 1, nvcv::FMT_RGBAf32p, nvcv::FMT_RGBAf32},
    {36, 32, 5, 3,         NVCV_BORDER_REFLECT, NVCV_DILATE, 2, 1, nvcv::FMT_RGBf32p, nvcv::FMT_RGBf32},
    {38, 34, 5, 3,      NVCV_BORDER_REPLICATE, NVCV_ERODE,  2, 1, nvcv::FMT_RGBf32p, nvcv::FMT_RGBf32},
    {32, 28, 3, 5, NVCV_BORDER_REFLECT101, NVCV_CLOSE,  1, 1, nvcv::FMT_RGBAf32p, nvcv::FMT_RGBAf32},
    {36, 32, 5, 3,         NVCV_BORDER_REFLECT, NVCV_DILATE, 2, 1, nvcv::FMT_RGBf16p, nvcv::FMT_RGBf16},
    {38, 34, 5, 3,      NVCV_BORDER_REPLICATE, NVCV_ERODE,  2, 1, nvcv::FMT_RGBf16p, nvcv::FMT_RGBf16},
    {32, 28, 3, 5, NVCV_BORDER_REFLECT101, NVCV_CLOSE,  1, 1, nvcv::FMT_RGBAf16p, nvcv::FMT_RGBAf16},
});

// clang-format on

TEST_P(OpMorphologyPlanar, tensor_matches_interleaved)
{
    const int                width  = GetParamValue<0>();
    const int                height = GetParamValue<1>();
    const nvcv::Size2D       maskSize{GetParamValue<2>(), GetParamValue<3>()};
    const NVCVBorderType     borderMode     = GetParamValue<4>();
    const NVCVMorphologyType morphType      = GetParamValue<5>();
    const int                iteration      = GetParamValue<6>();
    const int                numImages      = GetParamValue<7>();
    const nvcv::ImageFormat  planarFmt      = GetParamValue<8>();
    const nvcv::ImageFormat  interleavedFmt = GetParamValue<9>();
    const int2               anchor{-1, -1};

    test::planar::RunTensorParity(
        planarFmt, interleavedFmt, width, height, width, height, numImages,
        [width, height, maskSize, borderMode, morphType, iteration, numImages, anchor](
            cudaStream_t stream, const nvcv::Tensor &src, const nvcv::Tensor &dst, nvcv::ImageFormat fmt)
        {
            nvcv::Tensor                 workspace{nullptr};
            nvcv::OptionalTensorConstRef workspaceRef{nvcv::NullOpt};
            if (NeedsWorkspace(morphType, iteration))
            {
                workspace    = nvcv::util::CreateTensor(numImages, width, height, fmt);
                workspaceRef = nvcv::OptionalTensorConstRef{std::cref(workspace)};
            }

            cvcuda::Morphology op;
            EXPECT_NO_THROW(op(stream, src, dst, workspaceRef, morphType, maskSize, anchor, iteration, borderMode));
        });
}

TEST_P(OpMorphologyPlanar, varshape_matches_interleaved)
{
    const int                width  = GetParamValue<0>();
    const int                height = GetParamValue<1>();
    const int2               maskSize{GetParamValue<2>(), GetParamValue<3>()};
    const NVCVBorderType     borderMode     = GetParamValue<4>();
    const NVCVMorphologyType morphType      = GetParamValue<5>();
    const int                iteration      = GetParamValue<6>();
    const int                numImages      = GetParamValue<7>();
    const nvcv::ImageFormat  planarFmt      = GetParamValue<8>();
    const nvcv::ImageFormat  interleavedFmt = GetParamValue<9>();
    const int2               anchor{-1, -1};

    test::planar::RunVarShapeParity(
        planarFmt, interleavedFmt, width, height, width, height, numImages,
        [maskSize, borderMode, morphType, iteration, numImages, anchor](
            cudaStream_t stream, const nvcv::ImageBatchVarShape &src, const nvcv::ImageBatchVarShape &dst,
            nvcv::ImageFormat fmt)
        {
            auto masks   = test::planar::MakePerImageTensor(numImages, nvcv::TYPE_2S32, maskSize);
            auto anchors = test::planar::MakePerImageTensor(numImages, nvcv::TYPE_2S32, anchor);

            nvcv::ImageBatchVarShape                 workspace{nullptr};
            std::vector<nvcv::Image>                 workspaceImages;
            nvcv::OptionalImageBatchVarShapeConstRef workspaceRef{nvcv::NullOpt};
            if (NeedsWorkspace(morphType, iteration))
            {
                workspace = nvcv::ImageBatchVarShape(numImages);
                for (int i = 0; i < numImages; ++i)
                {
                    workspaceImages.emplace_back(src[i].size(), fmt);
                }
                workspace.pushBack(workspaceImages.begin(), workspaceImages.end());
                workspaceRef = nvcv::OptionalImageBatchVarShapeConstRef{std::cref(workspace)};
            }

            cvcuda::Morphology op;
            EXPECT_NO_THROW(op(stream, src, dst, workspaceRef, morphType, masks, anchors, iteration, borderMode));
        });
}

static void ExpectU8MorphAlignment(const nvcv::ImageDataStridedCuda::Buffer &buffer, int offset, bool aligned)
{
    if (!aligned)
    {
        EXPECT_NE(0, buffer.planes[0].rowStride % alignof(uchar4));
        return;
    }
    EXPECT_EQ(0, buffer.planes[0].rowStride % alignof(uchar4));
    for (int c = 0; c < buffer.numPlanes; ++c)
    {
        EXPECT_EQ(static_cast<uintptr_t>(offset),
                  reinterpret_cast<uintptr_t>(buffer.planes[c].basePtr) % alignof(uchar4));
    }
}

static void FillU8MorphInput(std::vector<uint8_t> &input, const nvcv::ImageDataStridedCuda::Buffer &buffer, int sample)
{
    const auto &plane     = buffer.planes[0];
    const auto  planeSize = static_cast<size_t>(plane.rowStride) * plane.height;
    for (int c = 0; c < buffer.numPlanes; ++c)
    {
        for (int i = 0; i < plane.width * plane.height; ++i)
        {
            const int y = i / plane.width;
            const int x = i % plane.width;
            input[c * planeSize + y * plane.rowStride + x]
                = static_cast<uint8_t>(sample * 43 + c * 67 + y * 29 + x * 17 + 11);
        }
    }
}

static void RunU8MorphGold(int width, int height, NVCVMorphologyType morphType, int iteration, bool offsetBases = false,
                           bool mixedAnchors = false, bool staggeredDstPitches = false,
                           nvcv::ImageFormat format = nvcv::FMT_RGB8p, bool tensor = false)
{
    if (staggeredDstPitches)
    {
        ASSERT_EQ(morphType, NVCV_ERODE);
        ASSERT_EQ(iteration, 1);
    }

    const int                        channels  = format.numChannels();
    constexpr int                    numImages = 2;
    constexpr nvcv::Size2D           maskSize{3, 3};
    constexpr NVCVBorderType         borderMode = NVCV_BORDER_REPLICATE;
    const int                        alignedRow = (width + 3) & ~3;
    std::array<int, numImages>       rowStrides{alignedRow, alignedRow + (offsetBases ? 0 : 1)};
    const std::array<int, numImages> srcOffsets{offsetBases ? 1 : 0, 0};
    const std::array<int, numImages> dstOffsets{0, offsetBases ? 1 : 0};
    const std::vector<int2>          hostAnchors{
        mixedAnchors ? int2{0, 2}
          : int2{1, 1},
          int2{1, 1}
    };

    cudaStream_t rawStream{};
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&rawStream));
    std::unique_ptr<std::remove_pointer_t<cudaStream_t>, cudaError_t (*)(cudaStream_t)> stream(rawStream,
                                                                                               &cudaStreamDestroy);

    using DeviceBuffer = std::unique_ptr<NVCVByte, cudaError_t (*)(void *)>;
    std::vector<DeviceBuffer>                   srcAllocations;
    std::vector<DeviceBuffer>                   dstAllocations;
    std::array<size_t, numImages>               allocationSizes{};
    std::array<std::vector<uint8_t>, numImages> expected;
    {
        std::vector<nvcv::Image> srcImages;
        std::vector<nvcv::Image> dstImages;
        srcImages.reserve(numImages);
        dstImages.reserve(numImages);

        for (int b = 0; b < numImages; ++b)
        {
            const int    rowStride    = rowStrides[b];
            const size_t planeSize    = static_cast<size_t>(rowStride) * height;
            const size_t bytes        = planeSize * channels;
            const int    dstIndex     = staggeredDstPitches ? 1 - b : b;
            const int    dstRowStride = rowStrides[dstIndex];
            const size_t dstPlaneSize = static_cast<size_t>(dstRowStride) * height;
            const size_t dstBytes     = dstPlaneSize * channels;
            allocationSizes[b]        = dstBytes + dstOffsets[b];

            NVCVByte *srcAllocation{};
            ASSERT_EQ(cudaSuccess, cudaMalloc(reinterpret_cast<void **>(&srcAllocation), bytes + srcOffsets[b]));
            srcAllocations.emplace_back(srcAllocation, &cudaFree);
            NVCVByte *dstAllocation{};
            ASSERT_EQ(cudaSuccess, cudaMalloc(reinterpret_cast<void **>(&dstAllocation), allocationSizes[b]));
            dstAllocations.emplace_back(dstAllocation, &cudaFree);

            nvcv::ImageDataStridedCuda::Buffer srcBuffer{};
            srcBuffer.numPlanes = channels;
            for (int c = 0; c < channels; ++c)
            {
                srcBuffer.planes[c].width     = width;
                srcBuffer.planes[c].height    = height;
                srcBuffer.planes[c].rowStride = rowStride;
                srcBuffer.planes[c].basePtr   = srcAllocations[b].get() + srcOffsets[b] + c * planeSize;
            }
            auto dstBuffer = srcBuffer;
            for (int c = 0; c < channels; ++c)
            {
                dstBuffer.planes[c].rowStride = dstRowStride;
                dstBuffer.planes[c].basePtr   = dstAllocations[b].get() + dstOffsets[b] + c * dstPlaneSize;
            }

            ExpectU8MorphAlignment(srcBuffer, srcOffsets[b], b == 0 || offsetBases);
            ExpectU8MorphAlignment(dstBuffer, dstOffsets[b], dstIndex == 0 || offsetBases);

            srcImages.emplace_back(nvcv::ImageWrapData(nvcv::ImageDataStridedCuda{format, srcBuffer}));
            dstImages.emplace_back(nvcv::ImageWrapData(nvcv::ImageDataStridedCuda{format, dstBuffer}));

            std::vector<uint8_t> src(bytes, 0xA5);
            FillU8MorphInput(src, srcBuffer, b);

            expected[b].assign(dstBytes, 0xD7);
            long3 strides{static_cast<long>(planeSize), rowStride, 1};
            long3 dstStrides{static_cast<long>(dstPlaneSize), dstRowStride, 1};
            int2  anchor = hostAnchors[b];
            hostMorph(expected[b], dstStrides, src, strides, int3{width, height, channels}, nvcv::FMT_U8, maskSize,
                      anchor, iteration, borderMode, morphType);
            expected[b].insert(expected[b].begin(), dstOffsets[b], 0xD7);

            ASSERT_EQ(cudaSuccess, cudaMemset(srcAllocations[b].get(), 0xA5, bytes + srcOffsets[b]));
            ASSERT_EQ(cudaSuccess,
                      cudaMemcpy(srcAllocations[b].get() + srcOffsets[b], src.data(), bytes, cudaMemcpyHostToDevice));
            ASSERT_EQ(cudaSuccess, cudaMemset(dstAllocations[b].get(), 0xD7, allocationSizes[b]));
        }

        nvcv::ImageBatchVarShape srcBatch(numImages);
        nvcv::ImageBatchVarShape dstBatch(numImages);
        srcBatch.pushBack(srcImages.begin(), srcImages.end());
        dstBatch.pushBack(dstImages.begin(), dstImages.end());

        auto masks   = test::planar::MakePerImageTensor(numImages, nvcv::TYPE_2S32, int2{3, 3});
        auto anchors = test::planar::MakePerImageTensor(numImages, nvcv::TYPE_2S32, int2{1, 1});
        test::planar::UploadTensorValues(anchors, hostAnchors);

        nvcv::ImageBatchVarShape                 workspace{nullptr};
        nvcv::OptionalImageBatchVarShapeConstRef workspaceRef{nvcv::NullOpt};
        if (NeedsWorkspace(morphType, iteration))
        {
            workspace = nvcv::ImageBatchVarShape(numImages);
            for (int b = 0; b < numImages; ++b) workspace.pushBack(nvcv::Image(nvcv::Size2D{width, height}, format));
            workspaceRef = nvcv::OptionalImageBatchVarShapeConstRef{std::cref(workspace)};
        }

        cvcuda::Morphology op;
        if (tensor)
        {
            ASSERT_FALSE(NeedsWorkspace(morphType, iteration));
            ASSERT_NO_THROW({
                for (int b = 0; b < numImages; ++b)
                {
                    auto src = nvcv::TensorWrapImage(srcImages[b]);
                    auto dst = nvcv::TensorWrapImage(dstImages[b]);
                    op(stream.get(), src, dst, nvcv::OptionalTensorConstRef{nvcv::NullOpt}, morphType, maskSize,
                       hostAnchors[b], iteration, borderMode);
                }
            });
        }
        else
        {
            ASSERT_NO_THROW(
                op(stream.get(), srcBatch, dstBatch, workspaceRef, morphType, masks, anchors, iteration, borderMode));
        }
        ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream.get()));

        for (int b = 0; b < numImages; ++b)
        {
            std::vector<uint8_t> got(allocationSizes[b]);
            ASSERT_EQ(cudaSuccess, cudaMemcpy(got.data(), dstAllocations[b].get(), got.size(), cudaMemcpyDeviceToHost));
            EXPECT_EQ(expected[b], got);
        }
    }
}

TEST(OpMorphologyPlanar, varshape_rgb8p_dilate_fast_path_matches_cpu_gold)
{
    RunU8MorphGold(35, 9, NVCV_DILATE, 1);
}

TEST(OpMorphologyPlanar, varshape_rgb8p_erode_vector_boundaries_match_cpu_gold)
{
    for (int width : {1, 3, 4, 5, 63, 64, 65})
    {
        SCOPED_TRACE(width);
        RunU8MorphGold(width, width < 6 ? 2 : 9, NVCV_ERODE, 1);
    }
}

TEST(OpMorphologyPlanar, varshape_rgb8p_open_close_workspace_matches_cpu_gold)
{
    for (NVCVMorphologyType morphType : {NVCV_OPEN, NVCV_CLOSE})
    {
        SCOPED_TRACE(morphType);
        for (int iteration : {1, 2, 3})
        {
            SCOPED_TRACE(iteration);
            RunU8MorphGold(65, 9, morphType, iteration);
        }
    }
}

TEST(OpMorphologyPlanar, scalar_and_planar_u8_alignment_matches_cpu_gold)
{
    for (bool tensor : {false, true})
        for (auto format : {nvcv::FMT_U8, nvcv::FMT_RGB8p, nvcv::FMT_RGBA8p})
        {
            SCOPED_TRACE(::testing::Message() << "tensor=" << tensor << ", format=" << format);
            for (int width : {3, 4, 63, 64, 65})
            {
                RunU8MorphGold(width, 9, NVCV_DILATE, 1, false, false, false, format, tensor);
                RunU8MorphGold(width, 9, NVCV_ERODE, 1, false, false, false, format, tensor);
            }
            RunU8MorphGold(65, 9, NVCV_ERODE, 1, true, false, false, format, tensor);
            RunU8MorphGold(65, 9, NVCV_ERODE, 1, false, true, false, format, tensor);
            RunU8MorphGold(65, 9, NVCV_ERODE, 1, false, false, true, format, tensor);
        }
}

template<typename T>
static void FillMorphologySpecialValues(std::vector<uint8_t> &input, int pattern)
{
    for (size_t i = 0; i < input.size() / sizeof(T); ++i)
    {
        const std::array<float, 6> values{std::numeric_limits<float>::quiet_NaN(),
                                          std::numeric_limits<float>::infinity(),
                                          -std::numeric_limits<float>::infinity(),
                                          -7.0f,
                                          3.0f,
                                          11.0f};
        // Separate zero signs avoid the host/device min/max tie-rule difference while checking
        // that either sign survives unchanged; mixed NaN/finite taps exercise reduction order.
        float                      value = values[(i / 3 + i % 3) % values.size()];
        if (pattern == 1)
            value = values[0];
        else if (pattern == 2)
            value = -0.0f;
        else if (pattern == 3)
            value = 0.0f;
        const auto encoded = static_cast<T>(value);
        std::memcpy(input.data() + i * sizeof(T), &encoded, sizeof(T));
    }
}

static void MaybeFillMorphologySpecialValues(std::vector<uint8_t> &input, int elemSize, int pattern)
{
    if (pattern == 0)
        return;
    if (elemSize == sizeof(float))
        FillMorphologySpecialValues<float>(input, pattern);
    else
        FillMorphologySpecialValues<__half>(input, pattern);
}

// Both layouts are checked against the CPU primitive, so agreement between two device paths
// cannot hide the same error in their shared kernel.
static void RunMorphologyLayoutGold(bool varShape, nvcv::ImageFormat planarFmt, nvcv::ImageFormat interleavedFmt,
                                    nvcv::Size2D size, int maskWidth, NVCVBorderType border,
                                    NVCVMorphologyType morphType, int iteration, int specialPattern = 0,
                                    bool mixedWindows = false)
{
    constexpr int                     numImages = 2;
    const int                         elemSize  = planarFmt.planePixelStrideBytes(0);
    const int2                        anchor{maskWidth / 2, maskWidth / 2};
    const nvcv::Size2D                mask{maskWidth, maskWidth};
    std::vector<std::vector<uint8_t>> gold;
    auto makeInput = [elemSize, planarFmt, specialPattern, &gold, anchor, interleavedFmt, mask, iteration, border,
                      morphType, mixedWindows](int sample, int width, int height, int channels)
    {
        std::vector<uint8_t> input(static_cast<size_t>(width) * height * channels * elemSize);
        test::planar::FillDeterministicValues(input, sample * 101 + 1, planarFmt.planeDataType(0));
        MaybeFillMorphologySpecialValues(input, elemSize, specialPattern);
        const long3 strides{height * width * channels * elemSize, width * channels * elemSize, channels * elemSize};
        gold.emplace_back(input.size());
        auto sampleMask     = mask;
        int2 resolvedAnchor = anchor;
        if (mixedWindows)
        {
            sampleMask.w += sample;
            sampleMask.h += sample;
            resolvedAnchor = sample == 0 ? int2{0, 0} : int2{sampleMask.w - 1, sampleMask.h - 1};
        }
        hostMorph(gold.back(), strides, input, strides, int3{width, height, 1}, interleavedFmt, sampleMask,
                  resolvedAnchor, iteration, border, morphType);
        return input;
    };
    size_t sample  = 0;
    auto   compare = [&sample, &gold](const std::vector<uint8_t> &interleaved, const std::vector<uint8_t> &planar)
    {
        ASSERT_LT(sample, gold.size());
        EXPECT_EQ(gold[sample], interleaved);
        EXPECT_EQ(gold[sample], planar);
        ++sample;
    };
    if (varShape)
    {
        // Different extents put both the image-table lookup and partial blocks under test.
        const std::vector<nvcv::Size2D> sizes{
            size,
            {size.w + 2, size.h + 1}
        };
        test::planar::RunVarShapeParity(
            planarFmt, interleavedFmt, sizes, sizes,
            [maskWidth, anchor, morphType, iteration, &sizes, border, mixedWindows](
                cudaStream_t stream, const nvcv::ImageBatchVarShape &src, const nvcv::ImageBatchVarShape &dst,
                nvcv::ImageFormat format)
            {
                auto masks = test::planar::MakePerImageTensor(numImages, nvcv::TYPE_2S32, int2{maskWidth, maskWidth});
                auto anchors
                    = test::planar::MakePerImageTensor(numImages, nvcv::TYPE_2S32, mixedWindows ? int2{0, 0} : anchor);
                if (mixedWindows)
                {
                    const int2 secondMask{maskWidth + 1, maskWidth + 1};
                    const int2 secondAnchor{maskWidth, maskWidth};
                    auto       maskData   = masks.exportData<nvcv::TensorDataStridedCuda>();
                    auto       anchorData = anchors.exportData<nvcv::TensorDataStridedCuda>();
                    ASSERT_EQ(cudaSuccess, cudaMemcpy(maskData->basePtr() + maskData->stride(0), &secondMask,
                                                      sizeof(secondMask), cudaMemcpyHostToDevice));
                    ASSERT_EQ(cudaSuccess, cudaMemcpy(anchorData->basePtr() + anchorData->stride(0), &secondAnchor,
                                                      sizeof(secondAnchor), cudaMemcpyHostToDevice));
                }
                nvcv::ImageBatchVarShape                 workspace{nullptr};
                nvcv::OptionalImageBatchVarShapeConstRef workspaceRef{nvcv::NullOpt};
                if (NeedsWorkspace(morphType, iteration))
                {
                    workspace = nvcv::ImageBatchVarShape(numImages);
                    for (auto imageSize : sizes) workspace.pushBack(nvcv::Image(imageSize, format));
                    workspaceRef = nvcv::OptionalImageBatchVarShapeConstRef{std::cref(workspace)};
                }
                cvcuda::Morphology op;
                ASSERT_NO_THROW(op(stream, src, dst, workspaceRef, morphType, masks, anchors, iteration, border));
                ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
            },
            makeInput, compare);
    }
    else
    {
        test::planar::RunTensorParity(
            planarFmt, interleavedFmt, size.w, size.h, size.w, size.h, numImages,
            [morphType, iteration, size, mask, anchor, border](cudaStream_t stream, const nvcv::Tensor &src,
                                                               const nvcv::Tensor &dst, nvcv::ImageFormat format)
            {
                nvcv::Tensor                 workspace{nullptr};
                nvcv::OptionalTensorConstRef workspaceRef{nvcv::NullOpt};
                if (NeedsWorkspace(morphType, iteration))
                {
                    workspace    = nvcv::util::CreateTensor(numImages, size.w, size.h, format);
                    workspaceRef = nvcv::OptionalTensorConstRef{std::cref(workspace)};
                }
                cvcuda::Morphology op;
                ASSERT_NO_THROW(op(stream, src, dst, workspaceRef, morphType, mask, anchor, iteration, border));
                ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
            },
            makeInput, compare);
    }
    EXPECT_EQ(sample, numImages);
}

// clang-format off
NVCV_TEST_SUITE_P(OpMorphologyLayoutGold, test::ValueList<nvcv::ImageFormat, nvcv::ImageFormat, bool>
{
    {nvcv::FMT_RGB8p,    nvcv::FMT_RGB8,    false},
    {nvcv::FMT_RGBA8p,   nvcv::FMT_RGBA8,   false},
    {nvcv::FMT_RGBf32p,  nvcv::FMT_RGBf32,  false},
    {nvcv::FMT_RGBAf32p, nvcv::FMT_RGBAf32, false},
    {nvcv::FMT_RGBf16p,  nvcv::FMT_RGBf16,  false},
    {nvcv::FMT_RGB8p,    nvcv::FMT_RGB8,    true},
    {nvcv::FMT_RGBA8p,   nvcv::FMT_RGBA8,   true},
    {nvcv::FMT_RGBf32p,  nvcv::FMT_RGBf32,  true},
    {nvcv::FMT_RGBAf32p, nvcv::FMT_RGBAf32, true},
    {nvcv::FMT_RGBf16p,  nvcv::FMT_RGBf16,  true},
    {nvcv::FMT_RGBAf16p, nvcv::FMT_RGBAf16, false},
    {nvcv::FMT_RGBAf16p, nvcv::FMT_RGBAf16, true},
    {nvcv::FMT_RGBf16p.dataKind(nvcv::DataKind::UNSIGNED), nvcv::FMT_RGBf16.dataKind(nvcv::DataKind::UNSIGNED), false},
    {nvcv::FMT_RGBAf16p.dataKind(nvcv::DataKind::UNSIGNED), nvcv::FMT_RGBAf16.dataKind(nvcv::DataKind::UNSIGNED), false},
    {nvcv::FMT_RGBf16p.dataKind(nvcv::DataKind::UNSIGNED), nvcv::FMT_RGBf16.dataKind(nvcv::DataKind::UNSIGNED), true},
    {nvcv::FMT_RGBAf16p.dataKind(nvcv::DataKind::UNSIGNED), nvcv::FMT_RGBAf16.dataKind(nvcv::DataKind::UNSIGNED), true},
});

// clang-format on

TEST_P(OpMorphologyLayoutGold, block_edges_and_iterations_match_cpu)
{
    const auto planarFmt      = GetParamValue<0>();
    const auto interleavedFmt = GetParamValue<1>();
    const bool varShape       = GetParamValue<2>();
    for (auto morphType : {NVCV_DILATE, NVCV_ERODE, NVCV_OPEN, NVCV_CLOSE})
    {
        SCOPED_TRACE(morphType);
        for (auto border : {NVCV_BORDER_CONSTANT, NVCV_BORDER_REPLICATE, NVCV_BORDER_REFLECT, NVCV_BORDER_WRAP,
                            NVCV_BORDER_REFLECT101})
        {
            SCOPED_TRACE(border);
            for (auto size : {
                     nvcv::Size2D{ 1,  1},
                     { 3,  2},
                     {63, 15},
                     {64, 16},
                     {65, 17},
            })
            {
                SCOPED_TRACE(::testing::Message() << size.w << "x" << size.h);
                RunMorphologyLayoutGold(varShape, planarFmt, interleavedFmt, size, 3, border, morphType, 1);
            }
            RunMorphologyLayoutGold(varShape, planarFmt, interleavedFmt, {65, 17}, 5, border, morphType, 1);
            RunMorphologyLayoutGold(varShape, planarFmt, interleavedFmt, {65, 17}, 7, border, morphType, 1);
            RunMorphologyLayoutGold(varShape, planarFmt, interleavedFmt, {65, 17}, 5, border, morphType, 2);
        }
    }
}

TEST(OpMorphology, varshape_mixed_windows_match_cpu_gold)
{
    for (auto morphType : {NVCV_ERODE, NVCV_DILATE, NVCV_OPEN, NVCV_CLOSE})
        for (auto size : {
                 nvcv::Size2D{ 3,  2},
                 nvcv::Size2D{65, 17}
        })
            RunMorphologyLayoutGold(true, nvcv::FMT_RGB8p, nvcv::FMT_RGB8, size, 5, NVCV_BORDER_CONSTANT, morphType, 1,
                                    0, true);
}

static void RunMorphologySpecialGold(bool varShape, nvcv::ImageFormat planarFmt, nvcv::ImageFormat interleavedFmt,
                                     NVCVMorphologyType morphType)
{
    for (int pattern : {1, 2, 3, 4})
    {
        SCOPED_TRACE(::testing::Message() << "varShape=" << varShape << ", format=" << planarFmt << ", op=" << morphType
                                          << ", pattern=" << pattern);
        RunMorphologyLayoutGold(varShape, planarFmt, interleavedFmt, {65, 9}, 3, NVCV_BORDER_REPLICATE, morphType, 1,
                                pattern);
    }
}

TEST(OpMorphologyPlanar, floating_special_values_match_cpu)
{
    for (bool varShape : {false, true})
        for (const auto &[planarFmt, interleavedFmt] : {
                 std::pair{nvcv::FMT_RGBf16p, nvcv::FMT_RGBf16},
                 std::pair{nvcv::FMT_RGBf32p, nvcv::FMT_RGBf32}
        })
            for (auto morphType : {NVCV_DILATE, NVCV_ERODE})
                RunMorphologySpecialGold(varShape, planarFmt, interleavedFmt, morphType);
}

TEST(OpMorphologyPlanar, tensor_rejects_two_channel)
{
    const nvcv::Size2D       maskSize{3, 3};
    const int2               anchor{-1, -1};
    const int                iteration = 1;
    const NVCVMorphologyType morphType = NVCVMorphologyType::NVCV_ERODE;
    const NVCVBorderType     border    = NVCV_BORDER_CONSTANT;

    test::planar::ExpectPlanarTensorRejected(
        {1, 2, 16, 16}, {1, 2, 16, 16},
        [maskSize, anchor](cudaStream_t stream, const nvcv::Tensor &src, const nvcv::Tensor &dst)
        {
            cvcuda::Morphology op;
            op(stream, src, dst, nvcv::OptionalTensorConstRef{nvcv::NullOpt}, morphType, maskSize, anchor, iteration,
               border);
        });
}

// clang-format off
NVCV_TEST_SUITE_P(OpMorphologyVarShape, test::ValueList<int, int, int, NVCVImageFormat, int, int, NVCVBorderType, NVCVMorphologyType, int>
{
    // width, height, batches,                    format,  maskWidth, maskHeight,            borderMode,   morphType, iteration
    {      5,      5,       1,      NVCV_IMAGE_FORMAT_U8,          3,         3,    NVCV_BORDER_CONSTANT, NVCV_ERODE,          2},
    {      5,      5,       1,      NVCV_IMAGE_FORMAT_RGBAf32,     3,         3,    NVCV_BORDER_CONSTANT, NVCV_DILATE,         3},
    {     25,     45,       2,      NVCV_IMAGE_FORMAT_U8,          2,         2,    NVCV_BORDER_CONSTANT, NVCV_DILATE,         2},
    {    125,     35,       1,      NVCV_IMAGE_FORMAT_RGBA8,       3,         3,    NVCV_BORDER_CONSTANT, NVCV_ERODE,          7},
    {     52,     45,       1,      NVCV_IMAGE_FORMAT_U16,         1,         2,    NVCV_BORDER_CONSTANT, NVCV_ERODE,          1},
    {    325,     45,       3,      NVCV_IMAGE_FORMAT_RGB8,        3,         4,    NVCV_BORDER_CONSTANT, NVCV_DILATE,         1},
    {     25,     45,       4,      NVCV_IMAGE_FORMAT_U8,          3,         3,    NVCV_BORDER_CONSTANT, NVCV_ERODE,          1},
    {     25,     45,       2,      NVCV_IMAGE_FORMAT_U8,          -1,       -1,    NVCV_BORDER_CONSTANT, NVCV_DILATE,         1},
    {      5,      5,       1,      NVCV_IMAGE_FORMAT_U8,          3,         3,    NVCV_BORDER_CONSTANT, NVCV_OPEN,           2},
    {      5,      5,       1,      NVCV_IMAGE_FORMAT_RGBAf32,     3,         3,    NVCV_BORDER_CONSTANT, NVCV_CLOSE,          3},
    {     25,     45,       2,      NVCV_IMAGE_FORMAT_U8,          2,         2,    NVCV_BORDER_CONSTANT, NVCV_CLOSE,          2},
    {    125,     35,       1,      NVCV_IMAGE_FORMAT_RGBA8,       3,         3,    NVCV_BORDER_CONSTANT, NVCV_OPEN,           7},
    {     52,     45,      21,      NVCV_IMAGE_FORMAT_U16,         1,         2,    NVCV_BORDER_CONSTANT, NVCV_CLOSE,          1},
    {    325,     45,       3,      NVCV_IMAGE_FORMAT_RGB8,        3,         4,    NVCV_BORDER_CONSTANT, NVCV_CLOSE,          1},
    {     25,    456,       4,      NVCV_IMAGE_FORMAT_U8,          3,         3,    NVCV_BORDER_CONSTANT, NVCV_OPEN,           1},
    {     55,     45,       2,      NVCV_IMAGE_FORMAT_U8,         -1,        -1,    NVCV_BORDER_CONSTANT, NVCV_OPEN,           1},
    {      5,      5,       4,      NVCV_IMAGE_FORMAT_U8,          2,          2,   NVCV_BORDER_REPLICATE, NVCV_ERODE,         3},
    {     25,     45,       2,      NVCV_IMAGE_FORMAT_U8,          3,          3,   NVCV_BORDER_REPLICATE, NVCV_DILATE,        2},
    {     25,     45,       2,      NVCV_IMAGE_FORMAT_U8,          3,          3,   NVCV_BORDER_REPLICATE, NVCV_OPEN,          3},
    {     25,     44,       2,      NVCV_IMAGE_FORMAT_U8,          3,          3,   NVCV_BORDER_REPLICATE, NVCV_CLOSE,         2},
    {      5,      5,       1,      NVCV_IMAGE_FORMAT_U8,          2,          2,   NVCV_BORDER_REFLECT, NVCV_ERODE,           3},
    {     25,     45,       2,      NVCV_IMAGE_FORMAT_U8,          3,          3,   NVCV_BORDER_REFLECT, NVCV_DILATE,          2},
    {     25,     45,       2,      NVCV_IMAGE_FORMAT_U8,          3,          3,   NVCV_BORDER_REFLECT, NVCV_OPEN,            3},
    {     25,     44,       2,      NVCV_IMAGE_FORMAT_U8,          3,          3,   NVCV_BORDER_REFLECT, NVCV_CLOSE,           2},
    {      5,      5,       4,      NVCV_IMAGE_FORMAT_U8,          2,          2,   NVCV_BORDER_WRAP, NVCV_ERODE,              3},
    {     25,     45,       2,      NVCV_IMAGE_FORMAT_U8,          3,          3,   NVCV_BORDER_WRAP, NVCV_DILATE,             2},
    {     25,     45,       2,      NVCV_IMAGE_FORMAT_U8,          3,          3,   NVCV_BORDER_WRAP, NVCV_OPEN,               3},
    {     25,     44,       2,      NVCV_IMAGE_FORMAT_U8,          3,          3,   NVCV_BORDER_WRAP, NVCV_CLOSE,              2},
    {      5,      5,       4,      NVCV_IMAGE_FORMAT_U8,          2,          2,   NVCV_BORDER_REFLECT101, NVCV_ERODE,        3},
    {     25,     45,       2,      NVCV_IMAGE_FORMAT_U8,          3,          3,   NVCV_BORDER_REFLECT101, NVCV_DILATE,       2},
    {     25,     45,       2,      NVCV_IMAGE_FORMAT_U8,          3,          3,   NVCV_BORDER_REFLECT101, NVCV_OPEN,         3},
    {     25,     44,       2,      NVCV_IMAGE_FORMAT_U8,          3,          3,   NVCV_BORDER_REFLECT101, NVCV_CLOSE,        2},
    {      5,      5,       1,      NVCV_IMAGE_FORMAT_RGBAf16,     3,          3,   NVCV_BORDER_CONSTANT, NVCV_DILATE,         3},
    {     25,     45,       2,      NVCV_IMAGE_FORMAT_RGBf16,      3,          3,   NVCV_BORDER_REPLICATE, NVCV_ERODE,         1},
    {      5,      5,       1,      NVCV_IMAGE_FORMAT_RGBAf16,     3,          3,   NVCV_BORDER_CONSTANT, NVCV_CLOSE,          3},
    {     25,     45,       2,      NVCV_IMAGE_FORMAT_F16,         3,          3,   NVCV_BORDER_REFLECT, NVCV_OPEN,            2}
});

// clang-format on

TEST_P(OpMorphologyVarShape, varshape_correct_output)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    int width   = GetParamValue<0>();
    int height  = GetParamValue<1>();
    int batches = GetParamValue<2>();

    nvcv::ImageFormat format{GetParamValue<3>()};

    int maskSizeX = GetParamValue<4>();
    int maskSizeY = GetParamValue<5>();

    int anchorX = -1;
    int anchorY = -1;

    NVCVBorderType     borderMode = GetParamValue<6>();
    NVCVMorphologyType morphType  = GetParamValue<7>();
    int                iteration  = GetParamValue<8>();

    // Create input varshape
    std::default_random_engine    rng;
    std::uniform_int_distribution udistWidth(ScaledSize(width, 0.8), ScaledSize(width, 1.1));
    std::uniform_int_distribution udistHeight(ScaledSize(height, 0.8), ScaledSize(height, 1.1));

    std::vector<nvcv::Image> imgSrc;

    std::vector<std::vector<uint8_t>> srcVec(batches);
    std::vector<int>                  srcVecRowStride(batches);

    //setup the input images
    for (int i = 0; i < batches; ++i)
    {
        imgSrc.emplace_back(nvcv::Size2D{udistWidth(rng), udistHeight(rng)}, format);

        int srcRowStride   = imgSrc[i].size().w * format.planePixelStrideBytes(0);
        srcVecRowStride[i] = srcRowStride;

        srcVec[i].resize(imgSrc[i].size().h * srcRowStride);
        FillRandomInput(srcVec[i], format, rng);

        auto imgData = imgSrc[i].exportData<nvcv::ImageDataStridedCuda>();
        ASSERT_NE(imgData, nvcv::NullOpt);

        // Copy input data to the GPU
        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy2DAsync(imgData->plane(0).basePtr, imgData->plane(0).rowStride, srcVec[i].data(),
                                    srcRowStride, srcRowStride, imgSrc[i].size().h, cudaMemcpyHostToDevice, stream));
    }

    nvcv::ImageBatchVarShape batchSrc(batches);
    batchSrc.pushBack(imgSrc.begin(), imgSrc.end());

    // Create output varshape
    std::vector<nvcv::Image> imgDst;
    std::vector<nvcv::Image> imgWorkspace;
    for (int i = 0; i < batches; ++i)
    {
        imgDst.emplace_back(imgSrc[i].size(), imgSrc[i].format());
        imgWorkspace.emplace_back(imgSrc[i].size(), imgSrc[i].format());
    }
    nvcv::ImageBatchVarShape batchDst(batches);
    batchDst.pushBack(imgDst.begin(), imgDst.end());

    nvcv::ImageBatchVarShape batchWorkspace(batches);
    batchWorkspace.pushBack(imgWorkspace.begin(), imgWorkspace.end());

    // Create kernel mask size tensor
    nvcv::Tensor maskTensor({{batches}, "N"}, nvcv::TYPE_2S32);
    {
        auto dev = maskTensor.exportData<nvcv::TensorDataStridedCuda>();
        ASSERT_NE(dev, nullptr);

        std::vector<int2> vec(batches, int2{maskSizeX, maskSizeY});

        ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(dev->basePtr(), vec.data(), vec.size() * sizeof(int2),
                                               cudaMemcpyHostToDevice, stream));
    }

    // Create Anchor tensor
    nvcv::Tensor anchorTensor({{batches}, "N"}, nvcv::TYPE_2S32);
    {
        auto dev = anchorTensor.exportData<nvcv::TensorDataStridedCuda>();
        ASSERT_NE(dev, nullptr);

        std::vector<int2> vec(batches, int2{anchorX, anchorY});

        ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(dev->basePtr(), vec.data(), vec.size() * sizeof(int2),
                                               cudaMemcpyHostToDevice, stream));
    }

    // Run operator set the max batches
    cvcuda::Morphology morphOp;

    EXPECT_NO_THROW(morphOp(stream, batchSrc, batchDst, nvcv::OptionalImageBatchVarShapeConstRef{batchWorkspace},
                            morphType, maskTensor, anchorTensor, iteration, borderMode));

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    // Check test data against gold
    for (int i = 0; i < batches; ++i)
    {
        SCOPED_TRACE(i);

        const auto srcData = imgSrc[i].exportData<nvcv::ImageDataStridedCuda>();
        ASSERT_EQ(srcData->numPlanes(), 1);

        const auto dstData = imgDst[i].exportData<nvcv::ImageDataStridedCuda>();
        ASSERT_EQ(dstData->numPlanes(), 1);

        int dstRowStride = srcVecRowStride[i];

        int3  shape{srcData->plane(0).width, srcData->plane(0).height, 1};
        long3 pitches{shape.y * dstRowStride, dstRowStride, format.planePixelStrideBytes(0)};

        std::vector<uint8_t> testVec(shape.y * pitches.y);

        // Copy output data to Host
        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy2D(testVec.data(), dstRowStride, dstData->plane(0).basePtr, dstData->plane(0).rowStride,
                               dstRowStride, shape.y, cudaMemcpyDeviceToHost));

        // Generate gold result

        if (maskSizeX == -1 || maskSizeY == -1)
        {
            maskSizeX = 3;
            maskSizeY = 3;
        }
        nvcv::Size2D         maskSize{maskSizeX, maskSizeY};
        int2                 kernelAnchor{maskSize.w / 2, maskSize.h / 2};
        std::vector<uint8_t> goldVec(shape.y * pitches.y);

        //generate gold result
        hostMorph(goldVec, pitches, srcVec[i], pitches, shape, format, maskSize, kernelAnchor, iteration, borderMode,
                  morphType);

        EXPECT_EQ(testVec, goldVec);
    }
}

TEST_P(OpMorphologyVarShape, varshape_noop)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    int width   = GetParamValue<0>();
    int height  = GetParamValue<1>();
    int batches = GetParamValue<2>();

    nvcv::ImageFormat format{GetParamValue<3>()};

    int maskSizeX = GetParamValue<4>();
    int maskSizeY = GetParamValue<5>();

    int                anchorX    = -1;
    int                anchorY    = -1;
    int                iteration  = 0; // this will bypass and do a copy
    NVCVBorderType     borderMode = GetParamValue<6>();
    NVCVMorphologyType morphType  = GetParamValue<7>();

    // do not check noop on open/close since it needs workspace.
    if (morphType == NVCVMorphologyType::NVCV_OPEN || morphType == NVCVMorphologyType::NVCV_CLOSE)
        return;

    // Create input varshape
    std::default_random_engine    rng;
    std::uniform_int_distribution udistWidth(ScaledSize(width, 0.8), ScaledSize(width, 1.1));
    std::uniform_int_distribution udistHeight(ScaledSize(height, 0.8), ScaledSize(height, 1.1));

    std::vector<nvcv::Image> imgSrc;

    std::vector<std::vector<uint8_t>> srcVec(batches);
    std::vector<int>                  srcVecRowStride(batches);

    //setup the input images
    for (int i = 0; i < batches; ++i)
    {
        imgSrc.emplace_back(nvcv::Size2D{udistWidth(rng), udistHeight(rng)}, format);

        int srcRowStride   = imgSrc[i].size().w * format.planePixelStrideBytes(0);
        srcVecRowStride[i] = srcRowStride;

        std::uniform_int_distribution<uint8_t> udist(0, 255);

        srcVec[i].resize(imgSrc[i].size().h * srcRowStride);
        std::ranges::generate(srcVec[i], [&udist, &rng]() { return udist(rng); });

        auto imgData = imgSrc[i].exportData<nvcv::ImageDataStridedCuda>();
        ASSERT_NE(imgData, nvcv::NullOpt);

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

    // Create kernel mask size tensor
    nvcv::Tensor maskTensor({{batches}, "N"}, nvcv::TYPE_2S32);
    {
        auto dev = maskTensor.exportData<nvcv::TensorDataStridedCuda>();
        ASSERT_NE(dev, nullptr);

        std::vector<int2> vec(batches, int2{maskSizeX, maskSizeY});

        ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(dev->basePtr(), vec.data(), vec.size() * sizeof(int2),
                                               cudaMemcpyHostToDevice, stream));
    }

    // Create Anchor tensor
    nvcv::Tensor anchorTensor({{batches}, "N"}, nvcv::TYPE_2S32);
    {
        auto dev = anchorTensor.exportData<nvcv::TensorDataStridedCuda>();
        ASSERT_NE(dev, nullptr);

        std::vector<int2> vec(batches, int2{anchorX, anchorY});

        ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(dev->basePtr(), vec.data(), vec.size() * sizeof(int2),
                                               cudaMemcpyHostToDevice, stream));
    }

    // Run operator set the max batches
    cvcuda::Morphology morphOp;

    EXPECT_NO_THROW(morphOp(stream, batchSrc, batchDst, nvcv::OptionalImageBatchVarShapeConstRef{nvcv::NullOpt},
                            morphType, maskTensor, anchorTensor, iteration, borderMode));

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    // Check test data against gold
    for (int i = 0; i < batches; ++i)
    {
        SCOPED_TRACE(i);

        const auto srcData = imgSrc[i].exportData<nvcv::ImageDataStridedCuda>();
        ASSERT_EQ(srcData->numPlanes(), 1);

        const auto dstData = imgDst[i].exportData<nvcv::ImageDataStridedCuda>();
        ASSERT_EQ(dstData->numPlanes(), 1);

        int dstRowStride = srcVecRowStride[i];

        int3  shape{srcData->plane(0).width, srcData->plane(0).height, 1};
        long3 pitches{shape.y * dstRowStride, dstRowStride, format.planePixelStrideBytes(0)};

        std::vector<uint8_t> testVec(shape.y * pitches.y);
        std::vector<uint8_t> goldVec(shape.y * pitches.y); // should be the same as source with iteration == 0

        // Copy output data to Host
        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy2D(testVec.data(), dstRowStride, dstData->plane(0).basePtr, dstData->plane(0).rowStride,
                               dstRowStride, shape.y, cudaMemcpyDeviceToHost));

        // Copy output data to Host
        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy2D(goldVec.data(), srcVecRowStride[i], srcData->plane(0).basePtr,
                               srcData->plane(0).rowStride, srcVecRowStride[i], shape.y, cudaMemcpyDeviceToHost));

        EXPECT_EQ(testVec, goldVec);
    }
}

// Regression: when workspace != nullptr and iteration == 0, the multi-pass driver previously
// wrote to workspace rather than outData and left the output untouched.
TEST(OpMorphology, noop_with_workspace_tensor)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    constexpr int width   = 16;
    constexpr int height  = 16;
    constexpr int batches = 2;

    nvcv::ImageFormat format{NVCV_IMAGE_FORMAT_U8};

    nvcv::Tensor inTensor        = nvcv::util::CreateTensor(batches, width, height, format);
    nvcv::Tensor outTensor       = nvcv::util::CreateTensor(batches, width, height, format);
    nvcv::Tensor workspaceTensor = nvcv::util::CreateTensor(batches, width, height, format);

    EXPECT_NO_THROW(nvcv::util::SetTensorToRandomValue<uint8_t>(inTensor.exportData(), 0, 0xFF));
    EXPECT_NO_THROW(nvcv::util::SetTensorTo<uint8_t>(outTensor.exportData(), 0));

    cvcuda::Morphology morphOp;
    int2               anchor{-1, -1};
    nvcv::Size2D       maskSize(3, 3);

    // The nested-jump count comes from gtest assertion macros inside the loop.
    for (auto morphType : {NVCVMorphologyType::NVCV_DILATE, NVCVMorphologyType::NVCV_ERODE}) // NOSONAR
    {
        // Poison the output so a broken call that leaves it untouched is detectable.
        EXPECT_NO_THROW(nvcv::util::SetTensorTo<uint8_t>(outTensor.exportData(), 0xCD));
        EXPECT_NO_THROW(morphOp(stream, inTensor, outTensor, nvcv::OptionalTensorConstRef{workspaceTensor}, morphType,
                                maskSize, anchor, 0, NVCV_BORDER_CONSTANT));

        ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

        for (int i = 0; i < batches; ++i)
        {
            nvcv::util::TensorImageData cvTensorIn(inTensor.exportData());
            nvcv::util::TensorImageData cvTensorOut(outTensor.exportData());
            EXPECT_TRUE(imageRegionValuesSame<uint8_t>(cvTensorIn, cvTensorOut));
        }
    }

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

// Regression: same bug in the var-shape dilate/erode driver.
TEST(OpMorphology, noop_with_workspace_varshape)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    constexpr int width   = 16;
    constexpr int height  = 16;
    constexpr int batches = 2;

    nvcv::ImageFormat format{NVCV_IMAGE_FORMAT_U8};

    std::vector<nvcv::Image> imgSrc;
    std::vector<nvcv::Image> imgDst;
    std::vector<nvcv::Image> imgWs;
    for (int i = 0; i < batches; ++i)
    {
        imgSrc.emplace_back(nvcv::Size2D{width, height}, format);
        imgDst.emplace_back(nvcv::Size2D{width, height}, format);
        imgWs.emplace_back(nvcv::Size2D{width, height}, format);
    }

    nvcv::ImageBatchVarShape batchSrc(batches);
    nvcv::ImageBatchVarShape batchDst(batches);
    nvcv::ImageBatchVarShape batchWs(batches);
    batchSrc.pushBack(imgSrc.begin(), imgSrc.end());
    batchDst.pushBack(imgDst.begin(), imgDst.end());
    batchWs.pushBack(imgWs.begin(), imgWs.end());

    // Fill input with known data, zero the output.
    std::default_random_engine             rng(42); // NOSONAR: deterministic test data, not security-sensitive.
    std::uniform_int_distribution<uint8_t> udist(0, 255);
    std::vector<std::vector<uint8_t>>      srcVec(batches);
    for (int i = 0; i < batches; ++i)
    {
        int rowStride = width; // U8 single channel
        srcVec[i].resize(static_cast<size_t>(height) * rowStride);
        std::ranges::generate(srcVec[i], [&]() { return udist(rng); });

        auto inData = imgSrc[i].exportData<nvcv::ImageDataStridedCuda>();
        ASSERT_NE(inData, nullptr);
        ASSERT_EQ(cudaSuccess, cudaMemcpy2DAsync(inData->plane(0).basePtr, inData->plane(0).rowStride, srcVec[i].data(),
                                                 rowStride, rowStride, height, cudaMemcpyHostToDevice, stream));

        auto outData = imgDst[i].exportData<nvcv::ImageDataStridedCuda>();
        ASSERT_NE(outData, nullptr);
        ASSERT_EQ(cudaSuccess, cudaMemset2DAsync(outData->plane(0).basePtr, outData->plane(0).rowStride, 0, rowStride,
                                                 height, stream));
    }

    nvcv::Tensor maskTensor({{batches}, "N"}, nvcv::TYPE_2S32);
    {
        auto dev = maskTensor.exportData<nvcv::TensorDataStridedCuda>();
        ASSERT_NE(dev, nullptr);
        std::vector<int2> vec(batches, int2{3, 3});
        ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(dev->basePtr(), vec.data(), vec.size() * sizeof(int2),
                                               cudaMemcpyHostToDevice, stream));
    }
    nvcv::Tensor anchorTensor({{batches}, "N"}, nvcv::TYPE_2S32);
    {
        auto dev = anchorTensor.exportData<nvcv::TensorDataStridedCuda>();
        ASSERT_NE(dev, nullptr);
        std::vector<int2> vec(batches, int2{-1, -1});
        ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(dev->basePtr(), vec.data(), vec.size() * sizeof(int2),
                                               cudaMemcpyHostToDevice, stream));
    }

    cvcuda::Morphology morphOp;

    // The nested-jump count comes from gtest assertion macros inside the loop.
    for (auto morphType : {NVCVMorphologyType::NVCV_DILATE, NVCVMorphologyType::NVCV_ERODE}) // NOSONAR
    {
        // Poison each destination image so a broken call that leaves it untouched is detectable.
        for (int i = 0; i < batches; ++i)
        {
            auto outData = imgDst[i].exportData<nvcv::ImageDataStridedCuda>();
            ASSERT_NE(outData, nullptr);
            ASSERT_EQ(cudaSuccess, cudaMemset2DAsync(outData->plane(0).basePtr, outData->plane(0).rowStride, 0xCD,
                                                     width, height, stream));
        }
        EXPECT_NO_THROW(morphOp(stream, batchSrc, batchDst, nvcv::OptionalImageBatchVarShapeConstRef{batchWs},
                                morphType, maskTensor, anchorTensor, 0, NVCV_BORDER_CONSTANT));

        ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

        for (int i = 0; i < batches; ++i)
        {
            int                  rowStride = width;
            std::vector<uint8_t> outVec(static_cast<size_t>(height) * rowStride);
            auto                 dstData = imgDst[i].exportData<nvcv::ImageDataStridedCuda>();
            ASSERT_NE(dstData, nullptr);
            ASSERT_EQ(cudaSuccess,
                      cudaMemcpy2D(outVec.data(), rowStride, dstData->plane(0).basePtr, dstData->plane(0).rowStride,
                                   rowStride, height, cudaMemcpyDeviceToHost));
            EXPECT_EQ(outVec, srcVec[i]);
        }
    }

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpMorphology_Negative, createNull)
{
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, cvcudaMorphologyCreate(nullptr));
}

TEST(OpMorphology_Negative, operator_negative)
{
    NVCVBorderType    borderMode = NVCV_BORDER_CONSTANT;
    nvcv::ImageFormat format{NVCV_IMAGE_FORMAT_U8};
    nvcv::Tensor      inTensor  = nvcv::util::CreateTensor(1, 24, 24, format);
    nvcv::Tensor      outTensor = nvcv::util::CreateTensor(1, 24, 24, format);

    cvcuda::Morphology morphOp;
    int2               anchor{0, 0};

    nvcv::Size2D maskSize(1, 1);

    // testSet0: iteration < 0
    EXPECT_THROW(morphOp(nullptr, inTensor, outTensor, nvcv::OptionalTensorConstRef{nvcv::NullOpt}, NVCV_ERODE,
                         maskSize, anchor, -1, borderMode),
                 nvcv::Exception);

    // testSet1: NVCV_DILATE and NVCV_ERODE && iteration > 1 && null workspace
    std::vector<NVCVMorphologyType> testSet1{NVCV_DILATE, NVCV_ERODE};
    for (auto morphType : testSet1) // NOSONAR
    {
        EXPECT_THROW(morphOp(nullptr, inTensor, outTensor, nvcv::OptionalTensorConstRef{nvcv::NullOpt}, morphType,
                             maskSize, anchor, 2, borderMode),
                     nvcv::Exception);
    }

    // testSet2: NVCV_CLOSE and NVCV_OPEN && null workspace
    std::vector<NVCVMorphologyType> testSet2{NVCV_CLOSE, NVCV_OPEN};
    for (auto morphType : testSet2) // NOSONAR
    {
        EXPECT_THROW(morphOp(nullptr, inTensor, outTensor, nvcv::OptionalTensorConstRef{nvcv::NullOpt}, morphType,
                             maskSize, anchor, 1, borderMode),
                     nvcv::Exception);
    }

    // testSet4: input format is not equal to output format
    {
        nvcv::Tensor outTensorInvalid = nvcv::util::CreateTensor(2, 24, 24, format);
        EXPECT_THROW(morphOp(nullptr, inTensor, outTensorInvalid, nvcv::OptionalTensorConstRef{nvcv::NullOpt},
                             NVCV_ERODE, maskSize, anchor, 0, borderMode),
                     nvcv::Exception);
    }
#ifndef ENABLE_SANITIZER
    // testSet: invalid morph type
    EXPECT_THROW(morphOp(nullptr, inTensor, outTensor, nvcv::OptionalTensorConstRef{nvcv::NullOpt},
                         static_cast<NVCVMorphologyType>(255), maskSize, anchor, 0, borderMode),
                 nvcv::Exception);
    // testSet: invalid border mode
    EXPECT_THROW(morphOp(nullptr, inTensor, outTensor, nvcv::OptionalTensorConstRef{nvcv::NullOpt}, NVCV_ERODE,
                         maskSize, anchor, 0, static_cast<NVCVBorderType>(255)),
                 nvcv::Exception);
#endif

    // testSet: unsupported 2-channel planar format
    {
#define NVCV_IMAGE_FORMAT_2U8 NVCV_DETAIL_MAKE_NONCOLOR_FMT1(PL, UNSIGNED, XY00, ASSOCIATED, X8_Y8)
        nvcv::ImageFormat formatInvalid{NVCV_IMAGE_FORMAT_2U8};
        nvcv::Tensor      inTensorInvalid  = nvcv::util::CreateTensor(1, 24, 24, formatInvalid);
        nvcv::Tensor      outTensorInvalid = nvcv::util::CreateTensor(1, 24, 24, formatInvalid);
        EXPECT_THROW(morphOp(nullptr, inTensorInvalid, outTensorInvalid, nvcv::OptionalTensorConstRef{nvcv::NullOpt},
                             NVCV_ERODE, maskSize, anchor, 0, borderMode),
                     nvcv::Exception);
#undef NVCV_IMAGE_FORMAT_2U8
    }
}

// These scalar formats are valid tensor types but are outside Morphology's supported dtype set;
// use a real 3x3 operation so the call reaches dtype classification before it rejects the input.
TEST(OpMorphology_Negative, rejects_unsupported_scalar_dtypes)
{
    cvcuda::Morphology morphOp;
    const nvcv::Size2D maskSize{3, 3};
    const int2         anchor{-1, -1};

    auto expectRejected = [&](nvcv::ImageFormat format)
    {
        nvcv::Tensor inTensor  = nvcv::util::CreateTensor(1, 24, 24, format);
        nvcv::Tensor outTensor = nvcv::util::CreateTensor(1, 24, 24, format);

        EXPECT_THROW(morphOp(nullptr, inTensor, outTensor, nvcv::OptionalTensorConstRef{nvcv::NullOpt}, NVCV_ERODE,
                             maskSize, anchor, 1, NVCV_BORDER_CONSTANT),
                     nvcv::Exception);
    };

    for (const auto format : {nvcv::FMT_F64, nvcv::FMT_S8, nvcv::FMT_S16, nvcv::FMT_S32})
    {
        expectRejected(format);
    }
}

// These scalar formats exercise dtype classification rejects before Morphology's supported dtype list.
TEST(OpMorphology_Negative, rejects_unclassifiable_scalar_dtypes)
{
    cvcuda::Morphology                     morphOp;
    const nvcv::Size2D                     maskSize{3, 3};
    const int2                             anchor{-1, -1};
    const std::array<nvcv::ImageFormat, 4> formats{
        nvcv::ImageFormat{NVCV_DETAIL_MAKE_NONCOLOR_FMT1(PL, FLOAT, X000, ASSOCIATED, X8)},
        nvcv::ImageFormat{NVCV_DETAIL_MAKE_NONCOLOR_FMT1(PL, SIGNED, X000, ASSOCIATED, X64)},
        nvcv::ImageFormat{nvcv::FMT_U32},
        nvcv::ImageFormat{nvcv::FMT_C64},
    };

    auto expectRejected = [&](const nvcv::ImageFormat &format)
    {
        nvcv::Tensor inTensor  = nvcv::util::CreateTensor(1, 24, 24, format);
        nvcv::Tensor outTensor = nvcv::util::CreateTensor(1, 24, 24, format);

        EXPECT_THROW(morphOp(nullptr, inTensor, outTensor, nvcv::OptionalTensorConstRef{nvcv::NullOpt}, NVCV_ERODE,
                             maskSize, anchor, 1, NVCV_BORDER_CONSTANT),
                     nvcv::Exception);
    };

    for (const nvcv::ImageFormat &format : formats)
    {
        expectRejected(format);
    }
}

// A packed dtype with unequal channel widths must be rejected before its first channel can be
// classified as a supported scalar type.
TEST(OpMorphology_Negative, rejects_mixed_width_tensor_dtype)
{
    cvcuda::Morphology   morphOp;
    const nvcv::DataType mixedWidth{nvcv::DataKind::UNSIGNED, nvcv::Packing::X5Y5Z6};

    nvcv::Tensor inTensor(
        {
            {1, 24, 24, 3},
            "NHWC"
    },
        mixedWidth);
    nvcv::Tensor outTensor(
        {
            {1, 24, 24, 3},
            "NHWC"
    },
        mixedWidth);

    EXPECT_THROW(morphOp(nullptr, inTensor, outTensor, nvcv::OptionalTensorConstRef{nvcv::NullOpt}, NVCV_ERODE,
                         nvcv::Size2D{3, 3}, int2{-1, -1}, 1, NVCV_BORDER_CONSTANT),
                 nvcv::Exception);
}

TEST(OpMorphology_Negative, rejects_varshape_mixed_plane_data_types)
{
    const nvcv::ImageFormat mixedPlaneFormat{NVCV_DETAIL_MAKE_NONCOLOR_FMT2(PL, UNSIGNED, XY00, ASSOCIATED, X8, X16)};
    const auto              masks   = test::planar::MakePerImageTensor(1, nvcv::TYPE_2S32, int2{3, 3});
    const auto              anchors = test::planar::MakePerImageTensor(1, nvcv::TYPE_2S32, int2{-1, -1});

    cvcuda::Morphology morphOp;
    test::planar::ExpectVarShapeRejected(
        {mixedPlaneFormat}, {mixedPlaneFormat},
        [&morphOp, &masks, &anchors](cudaStream_t stream, const nvcv::ImageBatchVarShape &batchSrc,
                                     const nvcv::ImageBatchVarShape &batchDst)
        {
            morphOp(stream, batchSrc, batchDst, nvcv::OptionalImageBatchVarShapeConstRef{nvcv::NullOpt}, NVCV_ERODE,
                    masks, anchors, 1, NVCV_BORDER_CONSTANT);
        });
}

TEST(OpMorphology_Negative, rejects_varshape_semiplanar_format)
{
    const nvcv::ImageFormat semiplanarFormat{NVCV_DETAIL_MAKE_NONCOLOR_FMT2(PL, UNSIGNED, XYZ0, ASSOCIATED, X8, X8)};
    const auto              masks   = test::planar::MakePerImageTensor(1, nvcv::TYPE_2S32, int2{3, 3});
    const auto              anchors = test::planar::MakePerImageTensor(1, nvcv::TYPE_2S32, int2{-1, -1});

    cvcuda::Morphology morphOp;
    test::planar::ExpectVarShapeRejected(
        {semiplanarFormat}, {semiplanarFormat},
        [&morphOp, &masks, &anchors](cudaStream_t stream, const nvcv::ImageBatchVarShape &batchSrc,
                                     const nvcv::ImageBatchVarShape &batchDst)
        {
            morphOp(stream, batchSrc, batchDst, nvcv::OptionalImageBatchVarShapeConstRef{nvcv::NullOpt}, NVCV_ERODE,
                    masks, anchors, 1, NVCV_BORDER_CONSTANT);
        });
}

TEST(OpMorphology_Negative, operator_varshape_negative)
{
    NVCVBorderType    borderMode = NVCV_BORDER_CONSTANT;
    nvcv::ImageFormat format{NVCV_IMAGE_FORMAT_U8};
    const int         batches = 2;

    std::vector<nvcv::Image> imgSrc;
    nvcv::ImageBatchVarShape batchSrc(batches);
    for (int i = 0; i < batches; ++i)
    {
        imgSrc.emplace_back(nvcv::Size2D{24, 24}, format);
    }
    batchSrc.pushBack(imgSrc.begin(), imgSrc.end());

    std::vector<nvcv::Image> imgDst;
    std::vector<nvcv::Image> imgWorkspace;
    nvcv::ImageBatchVarShape batchDst(batches);
    nvcv::ImageBatchVarShape batchWorkspace(batches);
    for (int i = 0; i < batches; ++i)
    {
        imgDst.emplace_back(imgSrc[i].size(), imgSrc[i].format());
        imgWorkspace.emplace_back(imgSrc[i].size(), imgSrc[i].format());
    }
    batchDst.pushBack(imgDst.begin(), imgDst.end());
    batchWorkspace.pushBack(imgWorkspace.begin(), imgWorkspace.end());

    // Create kernel mask size tensor
    nvcv::Tensor maskTensor({{batches}, "N"}, nvcv::TYPE_2S32);
    {
        auto dev = maskTensor.exportData<nvcv::TensorDataStridedCuda>();
        ASSERT_NE(dev, nullptr);

        std::vector<int2> vec(batches, int2{1, 1});

        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy(dev->basePtr(), vec.data(), vec.size() * sizeof(int2), cudaMemcpyHostToDevice));
    }

    // Create Anchor tensor
    nvcv::Tensor anchorTensor({{batches}, "N"}, nvcv::TYPE_2S32);
    {
        auto dev = anchorTensor.exportData<nvcv::TensorDataStridedCuda>();
        ASSERT_NE(dev, nullptr);

        std::vector<int2> vec(batches, int2{0, 0});

        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy(dev->basePtr(), vec.data(), vec.size() * sizeof(int2), cudaMemcpyHostToDevice));
    }

    cvcuda::Morphology morphOp;

    // testSet0: iteration < 0
    EXPECT_THROW(morphOp(nullptr, batchSrc, batchDst, nvcv::OptionalImageBatchVarShapeConstRef{batchWorkspace},
                         NVCV_ERODE, maskTensor, anchorTensor, -1, borderMode),
                 nvcv::Exception);

    // testSet1: NVCV_DILATE and NVCV_ERODE && iteration > 1 && null workspace
    std::vector<NVCVMorphologyType> testSet1{NVCV_DILATE, NVCV_ERODE};
    for (auto morphType : testSet1) // NOSONAR
    {
        EXPECT_THROW(morphOp(nullptr, batchSrc, batchDst, nvcv::OptionalImageBatchVarShapeConstRef{nvcv::NullOpt},
                             morphType, maskTensor, anchorTensor, 2, borderMode),
                     nvcv::Exception);
    }

    // testSet2: NVCV_CLOSE and NVCV_OPEN && null workspace
    std::vector<NVCVMorphologyType> testSet2{NVCV_CLOSE, NVCV_OPEN};
    for (auto morphType : testSet2) // NOSONAR
    {
        EXPECT_THROW(morphOp(nullptr, batchSrc, batchDst, nvcv::OptionalImageBatchVarShapeConstRef{nvcv::NullOpt},
                             morphType, maskTensor, anchorTensor, 1, borderMode),
                     nvcv::Exception);
    }

    // testSet4: input format is not equal to output format
    {
        std::vector<nvcv::Image> imgDstInvalid;
        std::vector<nvcv::Image> imgWorkspaceInvalid;
        nvcv::ImageBatchVarShape batchDstInvalid(1);
        nvcv::ImageBatchVarShape batchWorkspaceInvalid(1);
        imgDstInvalid.emplace_back(imgSrc[0].size(), imgSrc[0].format());
        imgWorkspaceInvalid.emplace_back(imgSrc[0].size(), imgSrc[0].format());
        batchDstInvalid.pushBack(imgDstInvalid.begin(), imgDstInvalid.end());
        batchWorkspaceInvalid.pushBack(imgWorkspaceInvalid.begin(), imgWorkspaceInvalid.end());

        EXPECT_THROW(
            morphOp(nullptr, batchSrc, batchDstInvalid, nvcv::OptionalImageBatchVarShapeConstRef{batchWorkspaceInvalid},
                    NVCV_ERODE, maskTensor, anchorTensor, 1, borderMode),
            nvcv::Exception);
    }

#ifndef ENABLE_SANITIZER
    // testSet5 : invalid morph type
    EXPECT_THROW(morphOp(nullptr, batchSrc, batchDst, nvcv::OptionalImageBatchVarShapeConstRef{batchWorkspace},
                         static_cast<NVCVMorphologyType>(255), maskTensor, anchorTensor, 0, borderMode),
                 nvcv::Exception);
    // invalid border mode
    EXPECT_THROW(morphOp(nullptr, batchSrc, batchDst, nvcv::OptionalImageBatchVarShapeConstRef{batchWorkspace},
                         NVCV_ERODE, maskTensor, anchorTensor, 0, static_cast<NVCVBorderType>(255)),
                 nvcv::Exception);
#endif

    // testSet6: unsupported 2-channel planar format
    {
#define NVCV_IMAGE_FORMAT_2U8 NVCV_DETAIL_MAKE_NONCOLOR_FMT1(PL, UNSIGNED, XY00, ASSOCIATED, X8_Y8)
        nvcv::ImageFormat        formatInvalid{NVCV_IMAGE_FORMAT_2U8};
        std::vector<nvcv::Image> imgSrcInvalid;
        nvcv::ImageBatchVarShape batchSrcInvalid(batches);
        for (int i = 0; i < batches; ++i)
        {
            imgSrcInvalid.emplace_back(nvcv::Size2D{24, 24}, formatInvalid);
        }
        batchSrcInvalid.pushBack(imgSrcInvalid.begin(), imgSrcInvalid.end());

        std::vector<nvcv::Image> imgDstInvalid;
        std::vector<nvcv::Image> imgWorkspaceInvalid;
        nvcv::ImageBatchVarShape batchDstInvalid(batches);
        nvcv::ImageBatchVarShape batchWorkspaceInvalid(batches);
        for (int i = 0; i < batches; ++i)
        {
            imgDstInvalid.emplace_back(imgSrcInvalid[i].size(), imgSrcInvalid[i].format());
            imgWorkspaceInvalid.emplace_back(imgSrcInvalid[i].size(), imgSrcInvalid[i].format());
        }
        batchDstInvalid.pushBack(imgDstInvalid.begin(), imgDstInvalid.end());
        batchWorkspaceInvalid.pushBack(imgWorkspaceInvalid.begin(), imgWorkspaceInvalid.end());

        EXPECT_THROW(morphOp(nullptr, batchSrcInvalid, batchDstInvalid,
                             nvcv::OptionalImageBatchVarShapeConstRef{batchWorkspaceInvalid}, NVCV_ERODE, maskTensor,
                             anchorTensor, 1, borderMode),
                     nvcv::Exception);
    }
#undef NVCV_IMAGE_FORMAT_2U8
}
