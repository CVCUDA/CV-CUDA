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

#include <cvcuda/OpAdaptiveThreshold.hpp>
#include <cvcuda/OpAdvCvtColor.hpp>
#include <cvcuda/OpAverageBlur.hpp>
#include <cvcuda/OpBilateralFilter.hpp>
#include <cvcuda/OpBndBox.hpp>
#include <cvcuda/OpBoxBlur.hpp>
#include <cvcuda/OpBrightnessContrast.hpp>
#include <cvcuda/OpCLAHE.hpp>
#include <cvcuda/OpCenterCrop.hpp>
#include <cvcuda/OpChannelReorder.hpp>
#include <cvcuda/OpColorTwist.hpp>
#include <cvcuda/OpComposite.hpp>
#include <cvcuda/OpConv2D.hpp>
#include <cvcuda/OpConvertTo.hpp>
#include <cvcuda/OpCopyMakeBorder.hpp>
#include <cvcuda/OpCropFlipNormalizeReformat.hpp>
#include <cvcuda/OpCustomCrop.hpp>
#include <cvcuda/OpCvtColor.hpp>
#include <cvcuda/OpErase.hpp>
#include <cvcuda/OpFindHomography.hpp>
#include <cvcuda/OpFlip.hpp>
#include <cvcuda/OpGammaContrast.hpp>
#include <cvcuda/OpGaussian.hpp>
#include <cvcuda/OpGaussianNoise.hpp>
#include <cvcuda/OpHQResize.hpp>
#include <cvcuda/OpHistogram.hpp>
#include <cvcuda/OpHistogramEq.hpp>
#include <cvcuda/OpInpaint.hpp>
#include <cvcuda/OpJointBilateralFilter.hpp>
#include <cvcuda/OpLabel.hpp>
#include <cvcuda/OpLaplacian.hpp>
#include <cvcuda/OpMedianBlur.hpp>
#include <cvcuda/OpMinAreaRect.hpp>
#include <cvcuda/OpMinMaxLoc.hpp>
#include <cvcuda/OpMorphology.hpp>
#include <cvcuda/OpNonMaximumSuppression.hpp>
#include <cvcuda/OpNormalize.hpp>
#include <cvcuda/OpOSD.hpp>
#include <cvcuda/OpPadAndStack.hpp>
#include <cvcuda/OpPairwiseMatcher.hpp>
#include <cvcuda/OpPillowResize.hpp>
#include <cvcuda/OpRandomResizedCrop.hpp>
#include <cvcuda/OpReformat.hpp>
#include <cvcuda/OpRemap.hpp>
#include <cvcuda/OpResize.hpp>
#include <cvcuda/OpResizeCropConvertReformat.hpp>
#include <cvcuda/OpRotate.hpp>
#include <cvcuda/OpSIFT.hpp>
#include <cvcuda/OpStack.hpp>
#include <cvcuda/OpThreshold.hpp>
#include <cvcuda/OpWarpAffine.hpp>
#include <cvcuda/OpWarpPerspective.hpp>

#include <type_traits>
#include <utility>

template<typename Wrapper>
static void ExpectWrapperIsMoveOnly()
{
    static_assert(!std::is_copy_constructible_v<Wrapper>, "operator wrapper must not be copy constructible");
    static_assert(!std::is_copy_assignable_v<Wrapper>, "operator wrapper must not be copy assignable");
    static_assert(std::is_move_constructible_v<Wrapper>, "operator wrapper must be move constructible");
    static_assert(std::is_move_assignable_v<Wrapper>, "operator wrapper must be move assignable");
    static_assert(std::is_nothrow_move_constructible_v<Wrapper>, "operator wrapper move constructor must be noexcept");
    static_assert(std::is_nothrow_move_assignable_v<Wrapper>, "operator wrapper move assignment must be noexcept");
}

TEST(PublicOperatorWrappers, allWrappersAreMoveOnly)
{
    ExpectWrapperIsMoveOnly<cvcuda::AdaptiveThreshold>();
    ExpectWrapperIsMoveOnly<cvcuda::AdvCvtColor>();
    ExpectWrapperIsMoveOnly<cvcuda::AverageBlur>();
    ExpectWrapperIsMoveOnly<cvcuda::BilateralFilter>();
    ExpectWrapperIsMoveOnly<cvcuda::BndBox>();
    ExpectWrapperIsMoveOnly<cvcuda::BoxBlur>();
    ExpectWrapperIsMoveOnly<cvcuda::BrightnessContrast>();
    ExpectWrapperIsMoveOnly<cvcuda::CenterCrop>();
    ExpectWrapperIsMoveOnly<cvcuda::ChannelReorder>();
    ExpectWrapperIsMoveOnly<cvcuda::CLAHE>();
    ExpectWrapperIsMoveOnly<cvcuda::ColorTwist>();
    ExpectWrapperIsMoveOnly<cvcuda::Composite>();
    ExpectWrapperIsMoveOnly<cvcuda::Conv2D>();
    ExpectWrapperIsMoveOnly<cvcuda::ConvertTo>();
    ExpectWrapperIsMoveOnly<cvcuda::CopyMakeBorder>();
    ExpectWrapperIsMoveOnly<cvcuda::CropFlipNormalizeReformat>();
    ExpectWrapperIsMoveOnly<cvcuda::CustomCrop>();
    ExpectWrapperIsMoveOnly<cvcuda::CvtColor>();
    ExpectWrapperIsMoveOnly<cvcuda::Erase>();
    ExpectWrapperIsMoveOnly<cvcuda::FindHomography>();
    ExpectWrapperIsMoveOnly<cvcuda::Flip>();
    ExpectWrapperIsMoveOnly<cvcuda::GammaContrast>();
    ExpectWrapperIsMoveOnly<cvcuda::Gaussian>();
    ExpectWrapperIsMoveOnly<cvcuda::GaussianNoise>();
    ExpectWrapperIsMoveOnly<cvcuda::Histogram>();
    ExpectWrapperIsMoveOnly<cvcuda::HistogramEq>();
    ExpectWrapperIsMoveOnly<cvcuda::HQResize>();
    ExpectWrapperIsMoveOnly<cvcuda::Inpaint>();
    ExpectWrapperIsMoveOnly<cvcuda::JointBilateralFilter>();
    ExpectWrapperIsMoveOnly<cvcuda::Label>();
    ExpectWrapperIsMoveOnly<cvcuda::Laplacian>();
    ExpectWrapperIsMoveOnly<cvcuda::MedianBlur>();
    ExpectWrapperIsMoveOnly<cvcuda::MinAreaRect>();
    ExpectWrapperIsMoveOnly<cvcuda::MinMaxLoc>();
    ExpectWrapperIsMoveOnly<cvcuda::Morphology>();
    ExpectWrapperIsMoveOnly<cvcuda::NonMaximumSuppression>();
    ExpectWrapperIsMoveOnly<cvcuda::Normalize>();
    ExpectWrapperIsMoveOnly<cvcuda::OSD>();
    ExpectWrapperIsMoveOnly<cvcuda::PadAndStack>();
    ExpectWrapperIsMoveOnly<cvcuda::PairwiseMatcher>();
    ExpectWrapperIsMoveOnly<cvcuda::PillowResize>();
    ExpectWrapperIsMoveOnly<cvcuda::RandomResizedCrop>();
    ExpectWrapperIsMoveOnly<cvcuda::Reformat>();
    ExpectWrapperIsMoveOnly<cvcuda::Remap>();
    ExpectWrapperIsMoveOnly<cvcuda::Resize>();
    ExpectWrapperIsMoveOnly<cvcuda::ResizeCropConvertReformat>();
    ExpectWrapperIsMoveOnly<cvcuda::Rotate>();
    ExpectWrapperIsMoveOnly<cvcuda::SIFT>();
    ExpectWrapperIsMoveOnly<cvcuda::Stack>();
    ExpectWrapperIsMoveOnly<cvcuda::Threshold>();
    ExpectWrapperIsMoveOnly<cvcuda::WarpAffine>();
    ExpectWrapperIsMoveOnly<cvcuda::WarpPerspective>();
}

// Runtime evidence that the move-only contract closes the original double-destroy bug
// (CWE-415): construct -> move -> destroy must transfer the handle so that the
// destination wrapper owns the original handle and destruction does not double-free it.
TEST(PublicOperatorWrappers, moveTransfersHandle)
{
    cvcuda::Resize           a;
    const NVCVOperatorHandle original = a.handle();
    ASSERT_NE(nullptr, original);

    cvcuda::Resize b{std::move(a)};
    EXPECT_EQ(original, b.handle());

    cvcuda::Resize           c;
    const NVCVOperatorHandle cOriginal = c.handle();
    ASSERT_NE(nullptr, cOriginal);
    ASSERT_NE(original, cOriginal);

    c = std::move(b);
    EXPECT_EQ(original, c.handle());
}
