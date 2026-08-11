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
"""Simple Custom Crop example with CVCUDA."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cvcuda

common_dir = Path(__file__).parent.parent
sys.path.append(str(common_dir))

from common import parse_image_args, read_image, write_image  # noqa: E402


def main() -> None:
    """Crop an image to an off-center rectangle with CVCUDA."""
    # docs_tag: begin_main
    args: argparse.Namespace = parse_image_args("cat_customcrop.jpg")
    # docs_tag: begin_read_image
    input_image: cvcuda.Tensor = read_image(args.input)
    # docs_tag: end_read_image

    # docs_tag: begin_customcrop_setup
    # Determine an off-center crop rectangle.
    # cvcuda.customcrop takes (x, y, width, height) in pixel coordinates,
    # where (x, y) is the top-left corner of the crop region.
    # We choose a region that is 60% of each dimension, starting at 20% offset
    # so the crop is visually off-center and still fits within the image.
    img_h, img_w = input_image.shape[0], input_image.shape[1]
    crop_x = img_w // 5  # 20% from the left edge
    crop_y = img_h // 5  # 20% from the top edge
    crop_w = max(1, (img_w * 3) // 5)  # 60% of the image width (at least 1px)
    crop_h = max(1, (img_h * 3) // 5)  # 60% of the image height (at least 1px)

    # RectI specifies the crop region in the input image coordinate space
    rect = cvcuda.RectI(x=crop_x, y=crop_y, width=crop_w, height=crop_h)
    # docs_tag: end_customcrop_setup

    # docs_tag: begin_customcrop
    # customcrop extracts a rectangular sub-region from the input tensor.
    # The output shape will be (crop_h, crop_w, channels) for an HWC input.
    output_image: cvcuda.Tensor = cvcuda.customcrop(input_image, rect)
    write_image(output_image, args.output)
    # docs_tag: end_customcrop
    # docs_tag: end_main


if __name__ == "__main__":
    main()
