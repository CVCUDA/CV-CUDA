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
"""Simple Advanced Color Conversion example with CVCUDA."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cvcuda

common_dir = Path(__file__).parent.parent
sys.path.append(str(common_dir))

from common import parse_image_args, read_image, write_image  # noqa: E402


def main() -> None:
    """Convert an image from RGB to YUV and back to RGB using advanced color conversion."""
    # docs_tag: begin_main
    args: argparse.Namespace = parse_image_args("cat_advcvtcolor.jpg")
    # docs_tag: begin_read_image
    input_image: cvcuda.Tensor = read_image(args.input)
    # docs_tag: end_read_image

    # docs_tag: begin_advcvtcolor
    # 1. Convert the RGB image to YUV using the BT.709 color specification.
    #    BT.709 is the standard for HDTV content and is a common choice for
    #    high-quality color-space transformations.
    yuv_image: cvcuda.Tensor = cvcuda.advcvtcolor(
        input_image,
        cvcuda.ColorConversion.RGB2YUV,
        cvcuda.ColorSpec.BT709,
    )

    # 2. Convert the YUV image back to RGB so the result is a viewable image.
    #    Using the same color specification (BT.709) ensures a round-trip that
    #    closely reproduces the original colors.
    output_image: cvcuda.Tensor = cvcuda.advcvtcolor(
        yuv_image,
        cvcuda.ColorConversion.YUV2RGB,
        cvcuda.ColorSpec.BT709,
    )

    write_image(output_image, args.output)
    # docs_tag: end_advcvtcolor
    # docs_tag: end_main


if __name__ == "__main__":
    main()
