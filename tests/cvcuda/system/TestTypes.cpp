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

#include <cvcuda/Types.h>
#include <gtest/gtest.h>

#include <array>

namespace {

TEST(Types, c_api_constructs_and_validates_handles)
{
    NVCVBndBoxI                      bndBox       = {};
    const std::array<NVCVBndBoxI, 2> bndBoxData   = {bndBox, bndBox};
    const std::array<int32_t, 3>     bndBoxCounts = {1, 0, 1};

    NVCVBndBoxesI bndBoxes = nullptr;
    ASSERT_EQ(NVCV_SUCCESS, nvcvBndBoxesIConstruct(&bndBoxes, bndBoxData.data(), bndBoxCounts.data(), 3));
    ASSERT_NE(nullptr, bndBoxes);
    EXPECT_EQ(NVCV_SUCCESS, nvcvBndBoxesIDestroy(bndBoxes));
    EXPECT_EQ(NVCV_SUCCESS, nvcvBndBoxesIDestroy(nullptr));

    NVCVBndBoxesI invalidBndBoxes = nullptr;
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcvBndBoxesIConstruct(nullptr, nullptr, nullptr, 0));
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcvBndBoxesIConstruct(&invalidBndBoxes, nullptr, nullptr, -1));
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcvBndBoxesIConstruct(&invalidBndBoxes, nullptr, nullptr, 1));
    const std::array<int32_t, 1> negativeCount = {-1};
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcvBndBoxesIConstruct(&invalidBndBoxes, nullptr, negativeCount.data(), 1));
    const std::array<int32_t, 1> oneCount = {1};
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcvBndBoxesIConstruct(&invalidBndBoxes, nullptr, oneCount.data(), 1));

    NVCVBlurBoxI                      blurBox       = {};
    const std::array<NVCVBlurBoxI, 2> blurBoxData   = {blurBox, blurBox};
    const std::array<int32_t, 3>      blurBoxCounts = {1, 0, 1};

    NVCVBlurBoxesI blurBoxes = nullptr;
    ASSERT_EQ(NVCV_SUCCESS, nvcvBlurBoxesIConstruct(&blurBoxes, blurBoxData.data(), blurBoxCounts.data(), 3));
    ASSERT_NE(nullptr, blurBoxes);
    EXPECT_EQ(NVCV_SUCCESS, nvcvBlurBoxesIDestroy(blurBoxes));
    EXPECT_EQ(NVCV_SUCCESS, nvcvBlurBoxesIDestroy(nullptr));

    NVCVBlurBoxesI invalidBlurBoxes = nullptr;
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcvBlurBoxesIConstruct(nullptr, nullptr, nullptr, 0));
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcvBlurBoxesIConstruct(&invalidBlurBoxes, nullptr, nullptr, -1));
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcvBlurBoxesIConstruct(&invalidBlurBoxes, nullptr, nullptr, 1));
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcvBlurBoxesIConstruct(&invalidBlurBoxes, nullptr, negativeCount.data(), 1));
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcvBlurBoxesIConstruct(&invalidBlurBoxes, nullptr, oneCount.data(), 1));

    NVCVPoint                        point      = {};
    NVCVLine                         line       = {};
    NVCVRotatedBox                   rotatedBox = {};
    NVCVCircle                       circle     = {};
    NVCVArrow                        arrow      = {};
    const std::array<NVCVOSDType, 6> elementTypes
        = {NVCV_OSD_RECT, NVCV_OSD_POINT, NVCV_OSD_LINE, NVCV_OSD_ROTATED_RECT, NVCV_OSD_CIRCLE, NVCV_OSD_ARROW};
    std::array<const void *, 6>  elementPayloads = {&bndBox, &point, &line, &rotatedBox, &circle, &arrow};
    const std::array<int32_t, 2> elementCounts   = {0, 6};

    NVCVElements elements = nullptr;
    ASSERT_EQ(NVCV_SUCCESS,
              nvcvElementsConstruct(&elements, elementTypes.data(), elementPayloads.data(), elementCounts.data(), 2));
    ASSERT_NE(nullptr, elements);
    EXPECT_EQ(NVCV_SUCCESS, nvcvElementsDestroy(elements));
    EXPECT_EQ(NVCV_SUCCESS, nvcvElementsDestroy(nullptr));

    NVCVElements invalidElements = nullptr;
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcvElementsConstruct(nullptr, nullptr, nullptr, nullptr, 0));
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcvElementsConstruct(&invalidElements, nullptr, nullptr, nullptr, -1));
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcvElementsConstruct(&invalidElements, nullptr, nullptr, oneCount.data(), 1));
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcvElementsConstruct(&invalidElements, nullptr, nullptr, negativeCount.data(), 1));

    const std::array<NVCVOSDType, 1> unsupportedType = {NVCV_OSD_TEXT};
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcvElementsConstruct(&invalidElements, unsupportedType.data(),
                                                                 elementPayloads.data(), oneCount.data(), 1));

    const std::array<NVCVOSDType, 1> rectType    = {NVCV_OSD_RECT};
    std::array<const void *, 1>      nullPayload = {nullptr};
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcvElementsConstruct(&invalidElements, rectType.data(), nullPayload.data(), oneCount.data(), 1));
}

TEST(Types, c_api_rejects_null_element_counts_for_nonempty_batch)
{
    NVCVElements elements = nullptr;
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcvElementsConstruct(&elements, nullptr, nullptr, nullptr, 1));
    EXPECT_EQ(nullptr, elements);
}

TEST(Types, c_api_rejects_osd_max_sentinel)
{
    NVCVElements                 elements = nullptr;
    const NVCVOSDType            type     = NVCV_OSD_MAX;
    const NVCVPoint              point    = {};
    std::array<const void *, 1>  payloads = {&point};
    const std::array<int32_t, 1> counts   = {1};

    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcvElementsConstruct(&elements, &type, payloads.data(), counts.data(), 1));
    EXPECT_EQ(nullptr, elements);
}

} // namespace
