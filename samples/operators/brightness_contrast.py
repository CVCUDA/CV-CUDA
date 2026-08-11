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
"""Simple Brightness Contrast example with CVCUDA."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cvcuda
import numpy as np

common_dir = Path(__file__).parent.parent
sys.path.append(str(common_dir))

from common import (  # noqa: E402
    parse_image_args,
    read_image,
    write_image,
    cuda_memcpy_h2d,
)


def main() -> None:
    """Apply brightness and contrast adjustment to an image with CVCUDA."""
    # docs_tag: begin_main
    args: argparse.Namespace = parse_image_args("cat_brightness_contrast.jpg")
    # docs_tag: begin_read_image
    input_image: cvcuda.Tensor = read_image(args.input)
    # docs_tag: end_read_image

    # docs_tag: begin_brightness_contrast_setup
    # The brightness_contrast operator expects per-image scalar parameters
    # supplied as 1-D tensors with layout "N" (one element per image in the batch).
    # brightness: multiplicative gain applied to the pixel values (>1 brightens).
    # contrast:   multiplier around contrast_center (<1 compresses, >1 expands).
    # brightness_shift: additive offset added after brightness scaling.
    # contrast_center: the pivot value around which contrast is computed (default 0.0).
    brightness_host = np.array([1.5], dtype=np.float32)
    contrast_host = np.array([1.4], dtype=np.float32)
    brightness_shift_host = np.array([20.0], dtype=np.float32)
    contrast_center_host = np.array([127.0], dtype=np.float32)

    brightness = cvcuda.Tensor((1,), dtype=cvcuda.Type.F32, layout="N")
    contrast = cvcuda.Tensor((1,), dtype=cvcuda.Type.F32, layout="N")
    brightness_shift = cvcuda.Tensor((1,), dtype=cvcuda.Type.F32, layout="N")
    contrast_center = cvcuda.Tensor((1,), dtype=cvcuda.Type.F32, layout="N")

    cuda_memcpy_h2d(brightness_host, brightness.cuda())
    cuda_memcpy_h2d(contrast_host, contrast.cuda())
    cuda_memcpy_h2d(brightness_shift_host, brightness_shift.cuda())
    cuda_memcpy_h2d(contrast_center_host, contrast_center.cuda())
    # docs_tag: end_brightness_contrast_setup

    # docs_tag: begin_brightness_contrast
    # Apply brightness and contrast adjustment.
    # The operator processes the HWC tensor directly; no batch dimension is required.
    # The result has the same shape, layout, and dtype as the input (uint8 HWC),
    # so it can be written directly to disk as a viewable JPEG.
    output_image: cvcuda.Tensor = cvcuda.brightness_contrast(
        src=input_image,
        brightness=brightness,
        contrast=contrast,
        brightness_shift=brightness_shift,
        contrast_center=contrast_center,
    )
    write_image(output_image, args.output)
    # docs_tag: end_brightness_contrast
    # docs_tag: end_main


if __name__ == "__main__":
    main()
