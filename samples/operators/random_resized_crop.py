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
"""Simple Random Resized Crop example with CVCUDA."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cvcuda

common_dir = Path(__file__).parent.parent
sys.path.append(str(common_dir))

from common import parse_image_args, read_image, write_image  # noqa: E402


def main() -> None:
    """Apply random resized crop to an image with CVCUDA."""
    # docs_tag: begin_main
    args: argparse.Namespace = parse_image_args("cat_random_resized_crop.jpg")
    # docs_tag: begin_read_image
    input_image: cvcuda.Tensor = read_image(args.input)
    # docs_tag: end_read_image

    # docs_tag: begin_random_resized_crop_setup
    # The operator works on batched (NHWC) tensors, so wrap the single HWC image
    # in a batch dimension of size 1. The output shape must also carry the batch dim.
    batched_input: cvcuda.Tensor = input_image.reshape((1, *input_image.shape), "NHWC")
    _, _, _, c = batched_input.shape
    output_h, output_w = args.height, args.width
    output_shape = (1, output_h, output_w, c)

    # Scale bounds control what fraction of the original image area the crop covers.
    # A min_scale of 0.08 and max_scale of 1.0 matches the standard torchvision
    # RandomResizedCrop defaults used in ImageNet training pipelines.
    min_scale: float = 0.08
    max_scale: float = 1.0

    # Ratio bounds set the aspect-ratio range (width/height) for the crop window
    # before it is resized to the target output dimensions.
    min_ratio: float = 0.75
    max_ratio: float = 1.3333333
    seed: int = 42
    # docs_tag: end_random_resized_crop_setup

    # docs_tag: begin_random_resized_crop
    # Apply a random crop of a random area/aspect-ratio sub-region, then resize it
    # to the requested output size — all in a single GPU kernel launch.
    output_image: cvcuda.Tensor = cvcuda.random_resized_crop(
        batched_input,
        output_shape,
        min_scale,
        max_scale,
        min_ratio,
        max_ratio,
        cvcuda.Interp.LINEAR,
        seed,
    )

    # Remove the batch dimension before writing so write_image receives an HWC tensor.
    hwc_output: cvcuda.Tensor = output_image.reshape((output_h, output_w, c), "HWC")
    write_image(hwc_output, args.output)
    # docs_tag: end_random_resized_crop
    # docs_tag: end_main


if __name__ == "__main__":
    main()
