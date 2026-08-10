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
"""Simple Pillow Resize example with CVCUDA."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cvcuda

common_dir = Path(__file__).parent.parent
sys.path.append(str(common_dir))

from common import parse_image_args, read_image, write_image  # noqa: E402


def main() -> None:
    """Pillow-quality resize an image with CVCUDA."""
    # docs_tag: begin_main
    args: argparse.Namespace = parse_image_args("cat_pillowresize.jpg")
    # docs_tag: begin_read_image
    input_image: cvcuda.Tensor = read_image(args.input)
    # docs_tag: end_read_image

    # docs_tag: begin_pillowresize
    # Pillow-style resize uses high-quality downsampling filters (e.g. LANCZOS)
    # that match PIL/Pillow output more closely than a plain bilinear resize.
    # The output shape must be (H, W, C) for HWC tensors.
    output_image: cvcuda.Tensor = cvcuda.pillowresize(
        input_image,
        (args.height, args.width, input_image.shape[2]),
        cvcuda.Format.RGB8,
        cvcuda.Interp.LANCZOS,
    )
    write_image(output_image, args.output)
    # docs_tag: end_pillowresize
    # docs_tag: end_main


if __name__ == "__main__":
    main()
