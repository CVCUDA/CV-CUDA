# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Simple stack example with CVCUDA."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cvcuda

common_dir = Path(__file__).parent.parent
sys.path.append(str(common_dir))

from common import parse_image_args, read_image  # noqa: E402


def main() -> None:
    """Stack an image with CVCUDA."""
    # docs_tag: begin_main
    args: argparse.Namespace = parse_image_args()
    # docs_tag: begin_read_image
    input_image: cvcuda.Tensor = read_image(args.input)
    height, width, channels = input_image.shape
    # docs_tag: end_read_image

    # docs_tag: begin_stack
    # 1. Generate all the images
    # We will simply use the same image for all to do the stack operation
    # in the real world, you would have different images/data to stack
    images: list[cvcuda.Tensor] = [input_image] * 3

    # 2. Perform a stack operation on the images
    stacked_images: cvcuda.Tensor = cvcuda.stack(images)
    assert stacked_images.shape == (3, height, width, channels)
    # docs_tag: end_stack
    # docs_tag: end_main


if __name__ == "__main__":
    main()
