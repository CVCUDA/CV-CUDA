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
"""Simple Copy Make Border example with CVCUDA."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cvcuda

common_dir = Path(__file__).parent.parent
sys.path.append(str(common_dir))

from common import parse_image_args, read_image, write_image  # noqa: E402


def main() -> None:
    """Add a colored border to an image with CVCUDA copymakeborder."""
    # docs_tag: begin_main
    args: argparse.Namespace = parse_image_args("cat_copymakeborder.jpg")
    # docs_tag: begin_read_image
    input_image: cvcuda.Tensor = read_image(args.input)
    # docs_tag: end_read_image

    # docs_tag: begin_copymakeborder
    # Add a visible border around the image using CONSTANT mode so the border
    # is filled with a solid color rather than replicated/reflected pixels.
    # The border widths (top=30, bottom=30, left=60, right=60) make the
    # added region clearly visible in the output.
    output_image: cvcuda.Tensor = cvcuda.copymakeborder(
        src=input_image,
        top=30,
        bottom=30,
        left=60,
        right=60,
        border_mode=cvcuda.Border.CONSTANT,
        # Bright orange border (R=255, G=140, B=0) makes the padding conspicuous.
        border_value=[255, 140, 0],
    )
    write_image(output_image, args.output)
    # docs_tag: end_copymakeborder
    # docs_tag: end_main


if __name__ == "__main__":
    main()
