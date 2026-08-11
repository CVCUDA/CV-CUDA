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
"""Simple Average Blur example with CVCUDA."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cvcuda

common_dir = Path(__file__).parent.parent
sys.path.append(str(common_dir))

from common import parse_image_args, read_image, write_image  # noqa: E402


def main() -> None:
    """Apply average blur to an image with CVCUDA."""
    # docs_tag: begin_main
    args: argparse.Namespace = parse_image_args("cat_averageblur.jpg")
    # docs_tag: begin_read_image
    input_image: cvcuda.Tensor = read_image(args.input)
    # docs_tag: end_read_image

    # docs_tag: begin_averageblur
    # Apply a 7x7 average (box) blur to smooth the image.
    # kernel_anchor=[-1, -1] places the anchor at the kernel center, which is
    # the conventional choice for symmetric filters.
    output_image: cvcuda.Tensor = cvcuda.averageblur(
        input_image,
        kernel_size=[7, 7],
        kernel_anchor=[-1, -1],
        border=cvcuda.Border.REFLECT101,
    )
    write_image(output_image, args.output)
    # docs_tag: end_averageblur
    # docs_tag: end_main


if __name__ == "__main__":
    main()
