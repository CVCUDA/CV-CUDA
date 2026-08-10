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
"""Simple Convert To example with CVCUDA."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cvcuda
import numpy as np

common_dir = Path(__file__).parent.parent
sys.path.append(str(common_dir))

from common import parse_image_args, read_image, write_image  # noqa: E402


def main() -> None:
    """Convert an image dtype with scaling using CVCUDA."""
    # docs_tag: begin_main
    args: argparse.Namespace = parse_image_args("cat_convertto.jpg")
    # docs_tag: begin_read_image
    input_image: cvcuda.Tensor = read_image(args.input)
    # docs_tag: end_read_image

    # docs_tag: begin_convertto
    # 1. Convert the uint8 image to float32, applying a scale factor.
    #    scale=1/255.0 maps [0, 255] -> [0.0, 1.0] — a standard normalization step
    #    used before feeding images into neural networks.
    float_image: cvcuda.Tensor = cvcuda.convertto(
        src=input_image,
        dtype=np.float32,
        scale=1.0 / 255.0,
    )

    # 2. Convert back to uint8 by reversing the scale (multiply by 255).
    #    This round-trip demonstrates that the conversion is lossless for
    #    images with pixel values in the valid uint8 range.
    output_image: cvcuda.Tensor = cvcuda.convertto(
        src=float_image,
        dtype=np.uint8,
        scale=255.0,
    )
    # docs_tag: end_convertto

    write_image(output_image, args.output)
    # docs_tag: end_main


if __name__ == "__main__":
    main()
