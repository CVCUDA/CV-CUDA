# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Simple Joint Bilateral Filter example with CVCUDA."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cvcuda

common_dir = Path(__file__).parent.parent
sys.path.append(str(common_dir))

from common import parse_image_args, read_image, write_image  # noqa: E402


def main() -> None:
    """Apply joint bilateral filter to an image with CVCUDA."""
    # docs_tag: begin_main
    args: argparse.Namespace = parse_image_args("cat_joint_bilateral_filter.jpg")
    # docs_tag: begin_read_image
    input_image: cvcuda.Tensor = read_image(args.input)
    # docs_tag: end_read_image

    # docs_tag: begin_joint_bilateral_filter_setup
    # The joint bilateral filter smooths `src` guided by `srcColor`.
    # Using a grayscale-converted version of the image as the guidance signal
    # keeps edges defined by luminance sharp while smoothing color noise.
    # We batch the HWC tensor to NHWC so cvtcolor (which requires a batch dim) works.
    nhwc_image: cvcuda.Tensor = input_image.reshape((1, *input_image.shape), "NHWC")
    guidance_image: cvcuda.Tensor = cvcuda.cvtcolor(
        nhwc_image, cvcuda.ColorConversion.RGB2GRAY
    )
    # cvtcolor produces a 1-channel NHWC tensor; replicate to 3 channels so it
    # matches the source tensor's channel count, which joint_bilateral_filter requires.
    guidance_3ch: cvcuda.Tensor = cvcuda.cvtcolor(
        guidance_image, cvcuda.ColorConversion.GRAY2RGB
    )
    # docs_tag: end_joint_bilateral_filter_setup

    # docs_tag: begin_joint_bilateral_filter
    # diameter=9  – neighbourhood pixel diameter (must be odd and positive)
    # sigma_color  – range kernel width; larger values blend more dissimilar colours
    # sigma_space  – spatial kernel width; larger values mean farther pixels contribute
    filtered_nhwc: cvcuda.Tensor = cvcuda.joint_bilateral_filter(
        nhwc_image,
        guidance_3ch,
        diameter=9,
        sigma_color=75.0,
        sigma_space=75.0,
        border=cvcuda.Border.REFLECT,
    )
    # Drop the batch dimension added for processing back to HWC before saving.
    filtered_image: cvcuda.Tensor = filtered_nhwc.reshape(
        filtered_nhwc.shape[1:], "HWC"
    )
    write_image(filtered_image, args.output)
    # docs_tag: end_joint_bilateral_filter
    # docs_tag: end_main


if __name__ == "__main__":
    main()
