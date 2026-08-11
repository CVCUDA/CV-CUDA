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
"""Simple Histogram Equalization example with CVCUDA."""

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
    """Apply histogram equalization to an image with CVCUDA."""
    # docs_tag: begin_main
    args: argparse.Namespace = parse_image_args("cat_histogrameq.jpg")
    # docs_tag: begin_read_image
    input_image: cvcuda.Tensor = read_image(args.input)
    # docs_tag: end_read_image

    # docs_tag: begin_histogrameq_setup
    # Histogram equalization in CVCUDA works on single-channel (grayscale) or
    # multi-channel tensors in HWC/NHWC layout with U8 dtype.
    # We batch the HWC image into NHWC so we can use cvtcolor for RGB->GRAY conversion.
    nhwc_image: cvcuda.Tensor = cvcuda.stack([input_image])
    gray_image: cvcuda.Tensor = cvcuda.cvtcolor(
        nhwc_image, cvcuda.ColorConversion.RGB2GRAY
    )
    # docs_tag: end_histogrameq_setup

    # docs_tag: begin_histogrameq
    # Apply histogram equalization: redistributes pixel intensities so that the
    # cumulative histogram of the output is approximately uniform, improving contrast.
    # The dtype keyword specifies the output element type (must be U8 for uint8 input).
    equalized: cvcuda.Tensor = cvcuda.histogrameq(src=gray_image, dtype=cvcuda.Type.U8)
    # docs_tag: end_histogrameq

    # The equalized output is NHWC with shape (1, H, W, 1).
    # Replicate the single channel across R, G, B on-device with cvtcolor, then
    # reshape to HWC so write_image can encode a viewable grayscale JPEG.
    # Staying on-device avoids a host round-trip and its row-pitch handling.
    rgb_nhwc: cvcuda.Tensor = cvcuda.cvtcolor(
        equalized, cvcuda.ColorConversion.GRAY2RGB
    )
    h, w = rgb_nhwc.shape[1], rgb_nhwc.shape[2]
    write_image(rgb_nhwc.reshape((h, w, 3), "HWC"), args.output)
    # docs_tag: end_main


if __name__ == "__main__":
    main()
