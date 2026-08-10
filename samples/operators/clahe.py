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
"""Simple CLAHE example with CVCUDA."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cvcuda

common_dir = Path(__file__).parent.parent
sys.path.append(str(common_dir))

from common import (  # noqa: E402
    parse_image_args,
    read_image,
    write_image,
)


def main() -> None:
    """Apply CLAHE contrast enhancement to an image with CVCUDA."""
    # docs_tag: begin_main
    args: argparse.Namespace = parse_image_args("cat_clahe.jpg")
    # docs_tag: begin_read_image
    input_image: cvcuda.Tensor = read_image(args.input)
    # docs_tag: end_read_image

    # docs_tag: begin_clahe_setup
    # CLAHE requires a single-channel (grayscale) U8 tensor.
    # Convert the RGB input to grayscale using cvtcolor, producing HWC with C=1.
    nhwc_image: cvcuda.Tensor = cvcuda.stack([input_image])
    gray_nhwc: cvcuda.Tensor = cvcuda.cvtcolor(
        nhwc_image, cvcuda.ColorConversion.RGB2GRAY
    )
    # docs_tag: end_clahe_setup

    # docs_tag: begin_clahe
    # Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).
    # clip_limit caps the contrast amplification per tile to suppress noise;
    # tile_grid_size divides the image into a grid of contextual regions.
    stream = cvcuda.Stream()
    clahe_output: cvcuda.Tensor = cvcuda.clahe(
        gray_nhwc,
        clip_limit=2.0,
        tile_grid_size=(8, 8),
        stream=stream,
    )
    stream.sync()
    # docs_tag: end_clahe

    # CLAHE output is NHWC with shape (1, H, W, 1).
    # Replicate the single channel to RGB on-device with cvtcolor, then reshape
    # to HWC so write_image can encode it.  Staying on-device avoids a host
    # round-trip and its row-pitch handling.
    rgb_nhwc: cvcuda.Tensor = cvcuda.cvtcolor(
        clahe_output, cvcuda.ColorConversion.GRAY2RGB
    )
    h, w = rgb_nhwc.shape[1], rgb_nhwc.shape[2]
    write_image(rgb_nhwc.reshape((h, w, 3), "HWC"), args.output)
    # docs_tag: end_main


if __name__ == "__main__":
    main()
