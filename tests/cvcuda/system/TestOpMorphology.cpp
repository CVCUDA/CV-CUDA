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
#include "PlanarParityUtils.hpp"

#include <common/TensorDataUtils.hpp>
#include <common/ValueTests.hpp>
#include <cvcuda/OpMorphology.hpp>
#include <cvcuda/cuda_tools/TypeTraits.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorDataAccess.hpp>

#include <array>
#include <functional>
#include <random>
#include <stdexcept>

namespace test = nvcv::test;
namespace cuda = nvcv::cuda;

using uchar = unsigned char;

namespace {

bool NeedsWorkspace(NVCVMorphologyType morphType, int iteration)
{
    return morphType == NVCVMorphologyType::NVCV_OPEN || morphType == NVCVMorphologyType::NVCV_CLOSE || iteration > 1;
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
        std::vector<uint8_t> tmpDst;
        tmpDst.reserve(hSrc.size());
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
    {     25,     44,       2, NVCV_IMAGE_FORMAT_U8,          3,          3,   NVCV_BORDER_REFLECT101, NVCV_CLOSE,          2}
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

    std::default_random_engine    randEng(0);
    std::uniform_int_distribution rand(0u, 255u);

    std::ranges::generate(inVec, [&rand, &randEng]() { return rand(randEng); });

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
    {     25,     44,       2,      NVCV_IMAGE_FORMAT_U8,          3,          3,   NVCV_BORDER_REFLECT101, NVCV_CLOSE,        2}
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

    // testSet3: invalid data type
    {
        nvcv::Tensor inTensorInvalid
            = nvcv::util::CreateTensor(1, 24, 24, nvcv::ImageFormat{NVCV_IMAGE_FORMAT_RGBAf16});
        nvcv::Tensor outTensorInvalid
            = nvcv::util::CreateTensor(1, 24, 24, nvcv::ImageFormat{NVCV_IMAGE_FORMAT_RGBAf16});
        EXPECT_THROW(morphOp(nullptr, inTensorInvalid, outTensorInvalid, nvcv::OptionalTensorConstRef{nvcv::NullOpt},
                             NVCV_ERODE, maskSize, anchor, 0, borderMode),
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

    // testSet3: invalid data type
    {
        nvcv::ImageFormat        formatInvalid{NVCV_IMAGE_FORMAT_RGBAf16};
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
