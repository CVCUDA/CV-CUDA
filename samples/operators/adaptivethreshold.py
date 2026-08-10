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
"""Simple Adaptive Threshold example with CVCUDA."""

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
)  # noqa: E402


def main() -> None:
    """Apply adaptive threshold to an image with CVCUDA."""
    # docs_tag: begin_main
    args: argparse.Namespace = parse_image_args("cat_adaptivethreshold.jpg")
    # docs_tag: begin_read_image
    input_image: cvcuda.Tensor = read_image(args.input)
    # docs_tag: end_read_image

    # docs_tag: begin_adaptivethreshold_setup
    # adaptivethreshold requires a single-channel (grayscale) U8 input.
    # Stack the HWC image into a batch (NHWC) so cvtcolor can operate on it,
    # then convert RGB to grayscale.
    nhwc_image: cvcuda.Tensor = cvcuda.stack([input_image])
    gray_nhwc: cvcuda.Tensor = cvcuda.cvtcolor(
        nhwc_image, cvcuda.ColorConversion.RGB2GRAY
    )
    # docs_tag: end_adaptivethreshold_setup

    # docs_tag: begin_adaptivethreshold
    # Apply GAUSSIAN_C adaptive threshold: for each pixel, the threshold is the
    # Gaussian-weighted average of the block_size x block_size neighbourhood minus c.
    # max_value=255 means foreground pixels are set to full white.
    # block_size must be an odd integer >= 3; c is subtracted from the local mean.
    thresholded: cvcuda.Tensor = cvcuda.adaptivethreshold(
        src=gray_nhwc,
        max_value=255.0,
        adaptive_method=cvcuda.AdaptiveThresholdType.GAUSSIAN_C,
        threshold_type=cvcuda.ThresholdType.BINARY,
        block_size=11,
        c=2.0,
    )
    # docs_tag: end_adaptivethreshold

    # The result is NHWC with a single channel.  Replicate the single channel
    # to RGB on-device with cvtcolor, then reshape to HWC so write_image (which
    # expects 3-channel uint8 HWC) can save it.  Staying on-device avoids any
    # host round-trip and the row-pitch handling it would require.
    rgb_nhwc: cvcuda.Tensor = cvcuda.cvtcolor(
        thresholded, cvcuda.ColorConversion.GRAY2RGB
    )
    h, w = rgb_nhwc.shape[1], rgb_nhwc.shape[2]
    write_image(rgb_nhwc.reshape((h, w, 3), "HWC"), args.output)
    # docs_tag: end_main


if __name__ == "__main__":
    main()
