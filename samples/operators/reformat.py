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
"""Simple reformat example with CVCUDA."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cvcuda

common_dir = Path(__file__).parent.parent
sys.path.append(str(common_dir))

from common import parse_image_args, read_image  # noqa: E402


def main() -> None:
    """Reformat an image with CVCUDA."""
    # docs_tag: begin_main
    args: argparse.Namespace = parse_image_args()
    # docs_tag: begin_read_image
    input_image: cvcuda.Tensor = read_image(args.input)
    height, width, channels = input_image.shape
    # docs_tag: end_read_image

    # docs_tag: begin_reformat
    # 1. Perform a reformat operation on the image to CHW
    chw_image: cvcuda.Tensor = cvcuda.reformat(input_image, "CHW")
    assert chw_image.shape == (channels, height, width)

    # 2. Perform a reformat operation on the CHW image back to HWC
    hwc_image: cvcuda.Tensor = cvcuda.reformat(chw_image, "HWC")
    assert hwc_image.shape == (height, width, channels)

    # 3. Add a batch dimension to the image and reformat to NCHW
    nhwc_image: cvcuda.Tensor = cvcuda.stack([input_image])
    nchw_image: cvcuda.Tensor = cvcuda.reformat(nhwc_image, "NCHW")
    assert nchw_image.shape == (1, channels, height, width)
    # docs_tag: end_reformat
    # docs_tag: end_main


if __name__ == "__main__":
    main()
