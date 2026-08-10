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
"""Simple Center Crop example with CVCUDA."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cvcuda

common_dir = Path(__file__).parent.parent
sys.path.append(str(common_dir))

from common import parse_image_args, read_image, write_image  # noqa: E402


def main() -> None:
    """Center-crop an image with CVCUDA."""
    # docs_tag: begin_main
    args: argparse.Namespace = parse_image_args("cat_center_crop.jpg")
    # docs_tag: begin_read_image
    input_image: cvcuda.Tensor = read_image(args.input)
    # docs_tag: end_read_image

    # docs_tag: begin_center_crop_setup
    # Derive a square crop size that fits within the image dimensions.
    # The operator requires [crop_height, crop_width] as a Python list.
    h, w = input_image.shape[0], input_image.shape[1]
    crop_h = min(args.height, h)
    crop_w = min(args.width, w)
    crop_size = [crop_h, crop_w]
    # docs_tag: end_center_crop_setup

    # docs_tag: begin_center_crop
    # cvcuda.center_crop symmetrically extracts a rectangular region from the
    # centre of the image, so no manual coordinate arithmetic is required.
    output_image: cvcuda.Tensor = cvcuda.center_crop(input_image, crop_size)
    write_image(output_image, args.output)
    # docs_tag: end_center_crop
    # docs_tag: end_main


if __name__ == "__main__":
    main()
