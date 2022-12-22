/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <common/HashUtils.hpp>
#include <common/ValueTests.hpp>
#include <nvcv/TensorDataAccess.hpp>

namespace test = nvcv::test;
namespace t    = ::testing;

namespace {

class MyTensorDataStrided : public nvcv::ITensorDataStrided
{
public:
    MyTensorDataStrided(nvcv::TensorShape tshape, nvcv::TensorShape::ShapeType strides, void *basePtr = nullptr)
        : m_tshape(std::move(tshape))
        , m_basePtr(basePtr)
        , m_strides(std::move(strides))
    {
        assert((int)strides.size() == tshape.rank());

        NVCVTensorData &data = this->cdata();
        data.bufferType      = NVCV_TENSOR_BUFFER_STRIDED_CUDA;
        data.rank            = tshape.size();
        data.dtype           = NVCV_DATA_TYPE_U8;
        data.layout          = tshape.layout();

        const nvcv::TensorShape::ShapeType &shape = tshape.shape();
        std::copy(shape.begin(), shape.end(), data.shape);

        NVCVTensorBufferStrided &buffer = data.buffer.strided;
        buffer.basePtr                  = reinterpret_cast<NVCVByte *>(basePtr);

        std::copy(strides.begin(), strides.end(), buffer.strides);
    }

    bool operator==(const MyTensorDataStrided &that) const
    {
        return std::tie(m_tshape, m_basePtr, m_strides) == std::tie(that.m_tshape, that.m_basePtr, that.m_strides);
    }

    bool operator<(const MyTensorDataStrided &that) const
    {
        return std::tie(m_tshape, m_basePtr, m_strides) < std::tie(that.m_tshape, that.m_basePtr, that.m_strides);
    }

    friend void Update(nvcv::util::HashMD5 &hash, const MyTensorDataStrided &d)
    {
        Update(hash, d.m_tshape, d.m_basePtr, d.m_strides);
    }

    friend std::ostream &operator<<(std::ostream &out, const MyTensorDataStrided &d)
    {
        return out << d.m_tshape << ",strides=" << d.m_strides << ",basePtr=" << d.m_basePtr;
    }

private:
    nvcv::TensorShape            m_tshape;
    void                        *m_basePtr;
    nvcv::TensorShape::ShapeType m_strides;
};

} // namespace

// TensorDataAccessStrided::sampleStride ========================

// clang-format off
NVCV_TEST_SUITE_P(TensorDataAccessStrided_SampleStride_ExecTests,
      test::ValueList<test::Param<"tdata",MyTensorDataStrided>,
                      test::Param<"gold",int64_t>>
      {
        {MyTensorDataStrided({{4,34,2},"NxC"},{160,4,2}),160},
        {MyTensorDataStrided({{4,34},"xN"},{10,4}),0},
      });

// clang-format on

TEST_P(TensorDataAccessStrided_SampleStride_ExecTests, works)
{
    const MyTensorDataStrided &input = std::get<0>(GetParam());
    const int64_t             &gold  = std::get<1>(GetParam());

    auto info = nvcv::TensorDataAccessStrided::Create(input);
    ASSERT_TRUE(info);
    EXPECT_EQ(gold, info->sampleStride());
}

// TensorDataAccessStrided::sampleData ========================

static std::byte *TEST_BASE_ADDR = reinterpret_cast<std::byte *>(0x123);

// clang-format off
NVCV_TEST_SUITE_P(TensorDataAccessStrided_SampleData_ExecTests,
      test::ValueList<test::Param<"tdata",MyTensorDataStrided>,
                      test::Param<"idx",int>,
                      test::Param<"gold",void *>>
      {
        {MyTensorDataStrided({{4,34,2},"NxC"},{160,4,2},TEST_BASE_ADDR),0,TEST_BASE_ADDR+0},
        {MyTensorDataStrided({{4,34,2},"NxC"},{160,4,2},TEST_BASE_ADDR),1,TEST_BASE_ADDR+160},
        {MyTensorDataStrided({{4,34,2},"NxC"},{160,4,2},TEST_BASE_ADDR),2,TEST_BASE_ADDR+2*160},
      });

// clang-format on

TEST_P(TensorDataAccessStrided_SampleData_ExecTests, works)
{
    const MyTensorDataStrided &input = std::get<0>(GetParam());
    const int                 &idx   = std::get<1>(GetParam());
    void                      *gold  = std::get<2>(GetParam());

    auto info = nvcv::TensorDataAccessStrided::Create(input);
    ASSERT_TRUE(info);
    EXPECT_EQ(gold, info->sampleData(idx));
}

// TensorDataAccessStridedImage::chStride ========================

// clang-format off
NVCV_TEST_SUITE_P(TensorDataAccessStridedImage_ChannelStride_ExecTests,
      test::ValueList<test::Param<"tdata",MyTensorDataStrided>,
                      test::Param<"gold",int64_t>>
      {
        {MyTensorDataStrided({{4,34,2},"NWC"},{160,4,2}),2},
        {MyTensorDataStrided({{4,34},"NW"},{160,2}),0},
        {MyTensorDataStrided({{4,34,3,6},"NCHW"},{1042,324,29,12}),324},
      });

// clang-format on

TEST_P(TensorDataAccessStridedImage_ChannelStride_ExecTests, works)
{
    const MyTensorDataStrided &input = std::get<0>(GetParam());
    const int64_t             &gold  = std::get<1>(GetParam());

    auto info = nvcv::TensorDataAccessStridedImage::Create(input);
    ASSERT_TRUE(info);
    EXPECT_EQ(gold, info->chStride());
}

// TensorDataAccessStridedImage::chData ========================

// clang-format off
NVCV_TEST_SUITE_P(TensorDataAccessStrided_ChannelData_ExecTests,
      test::ValueList<test::Param<"tdata",MyTensorDataStrided>,
                      test::Param<"idx",int>,
                      test::Param<"gold",void *>>
      {
        {MyTensorDataStrided({{4,34,3},"NWC"},{160,4,2}, TEST_BASE_ADDR),0, TEST_BASE_ADDR+0},
        {MyTensorDataStrided({{4,34,3},"NWC"},{160,4,2}, TEST_BASE_ADDR),1, TEST_BASE_ADDR+2},
        {MyTensorDataStrided({{4,34,3},"NWC"},{160,4,2}, TEST_BASE_ADDR),2, TEST_BASE_ADDR+4},
      });

// clang-format on

TEST_P(TensorDataAccessStrided_ChannelData_ExecTests, works)
{
    const MyTensorDataStrided &input = std::get<0>(GetParam());
    const int                 &idx   = std::get<1>(GetParam());
    void                      *gold  = std::get<2>(GetParam());

    auto info = nvcv::TensorDataAccessStridedImage::Create(input);
    ASSERT_TRUE(info);
    EXPECT_EQ(gold, info->chData(idx));
}

// TensorDataAccessStridedImage::rowStride ========================

// clang-format off
NVCV_TEST_SUITE_P(TensorDataAccessStridedImage_RowStride_ExecTests,
      test::ValueList<test::Param<"tdata",MyTensorDataStrided>,
                      test::Param<"gold",int64_t>>
      {
        {MyTensorDataStrided({{4,6,34,2},"NHWC"},{160,32,4,2}),32},
        {MyTensorDataStrided({{4,6,2},"NWC"},{160,32,2}),0},
      });

// clang-format on

TEST_P(TensorDataAccessStridedImage_RowStride_ExecTests, works)
{
    const MyTensorDataStrided &input = std::get<0>(GetParam());
    const int64_t             &gold  = std::get<1>(GetParam());

    auto info = nvcv::TensorDataAccessStridedImage::Create(input);
    ASSERT_TRUE(info);
    EXPECT_EQ(gold, info->rowStride());
}

// TensorDataAccessStridedImage::rowData ========================

// clang-format off
NVCV_TEST_SUITE_P(TensorDataAccessStrided_RowData_ExecTests,
      test::ValueList<test::Param<"tdata",MyTensorDataStrided>,
                      test::Param<"idx",int>,
                      test::Param<"gold",void *>>
      {
        {MyTensorDataStrided({{4,6,34,2},"NHWC"},{160,32,4,2}, TEST_BASE_ADDR), 0, TEST_BASE_ADDR+0},
        {MyTensorDataStrided({{4,6,34,2},"NHWC"},{160,32,4,2}, TEST_BASE_ADDR), 1, TEST_BASE_ADDR+32},
        {MyTensorDataStrided({{4,6,34,2},"NHWC"},{160,32,4,2}, TEST_BASE_ADDR), 2, TEST_BASE_ADDR+64},
      });

// clang-format on

TEST_P(TensorDataAccessStrided_RowData_ExecTests, works)
{
    const MyTensorDataStrided &input = std::get<0>(GetParam());
    const int                 &idx   = std::get<1>(GetParam());
    void                      *gold  = std::get<2>(GetParam());

    auto info = nvcv::TensorDataAccessStridedImage::Create(input);
    ASSERT_TRUE(info);
    EXPECT_EQ(gold, info->rowData(idx));
}

// TensorDataAccessStridedImagePlanar::planeStride ========================

// clang-format off
NVCV_TEST_SUITE_P(TensorDataAccessStridedImagePlanar_planeStride_ExecTests,
      test::ValueList<test::Param<"tdata",MyTensorDataStrided>,
                      test::Param<"gold",int64_t>>
      {
        {MyTensorDataStrided({{4,6,34,2},"NHWC"},{160,32,4,2}),0},
        {MyTensorDataStrided({{4,6,2},"NCW"},{160,32,2}),32},
      });

// clang-format on

TEST_P(TensorDataAccessStridedImagePlanar_planeStride_ExecTests, works)
{
    const MyTensorDataStrided &input = std::get<0>(GetParam());
    const int64_t             &gold  = std::get<1>(GetParam());

    auto info = nvcv::TensorDataAccessStridedImagePlanar::Create(input);
    ASSERT_TRUE(info);
    EXPECT_EQ(gold, info->planeStride());
}

// TensorDataAccessStridedImagePlanar::planeData ========================

// clang-format off
NVCV_TEST_SUITE_P(TensorDataAccessStridedImagePlanar_planeData_ExecTests,
      test::ValueList<test::Param<"tdata",MyTensorDataStrided>,
                      test::Param<"idx",int>,
                      test::Param<"gold",void *>>
      {
        {MyTensorDataStrided({{4,6,2},"NCW"},{160,32,2}, TEST_BASE_ADDR),0, TEST_BASE_ADDR+0},
        {MyTensorDataStrided({{4,6,2},"NCW"},{160,32,2}, TEST_BASE_ADDR),1, TEST_BASE_ADDR+32},
        {MyTensorDataStrided({{4,6,2},"NCW"},{160,32,2}, TEST_BASE_ADDR),2, TEST_BASE_ADDR+64},
      });

// clang-format on

TEST_P(TensorDataAccessStridedImagePlanar_planeData_ExecTests, works)
{
    const MyTensorDataStrided &input = std::get<0>(GetParam());
    const int                 &idx   = std::get<1>(GetParam());
    void                      *gold  = std::get<2>(GetParam());

    auto info = nvcv::TensorDataAccessStridedImagePlanar::Create(input);
    ASSERT_TRUE(info);
    EXPECT_EQ(gold, info->planeData(idx));
}
