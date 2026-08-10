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
"""Simple Remap example with CVCUDA."""

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
    upload_tensor,
)


def main() -> None:
    """Remap (wave-distort) an image with CVCUDA."""
    # docs_tag: begin_main
    args: argparse.Namespace = parse_image_args("cat_remap.jpg")
    # docs_tag: begin_read_image
    input_image: cvcuda.Tensor = read_image(args.input)
    # docs_tag: end_read_image

    # docs_tag: begin_remap_setup
    # Build a sinusoidal wave-distortion map in absolute coordinates.
    # The map tensor must have shape (H, W, 1) with dtype _2F32 (two float32
    # values packed per element: [src_x, src_y]) or shape (H, W, 2) with dtype F32.
    # We use shape (H, W, 2) / dtype F32 here so each pixel stores [src_x, src_y].
    height, width, _ = input_image.shape

    # Create grid of output pixel coordinates
    ys = np.arange(height, dtype=np.float32)
    xs = np.arange(width, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)  # both (H, W)

    # Apply a sinusoidal horizontal and vertical wave displacement
    amplitude = height * 0.04  # ~4 % of image height
    freq_x = 2.0 * np.pi / width * 3  # 3 cycles across width
    freq_y = 2.0 * np.pi / height * 3  # 3 cycles across height

    # Each output pixel at (y, x) samples the source at a displaced position,
    # creating a ripple effect that is visually distinctive without clipping content.
    src_x = grid_x + amplitude * np.sin(freq_y * grid_y)
    src_y = grid_y + amplitude * np.sin(freq_x * grid_x)

    # Stack into (H, W, 2) array — channel 0 = src_x, channel 1 = src_y
    map_np = np.stack([src_x, src_y], axis=2).astype(np.float32)
    map_np = np.ascontiguousarray(map_np)

    # Allocate a GPU tensor for the map and upload it from the host.
    # Layout "HWC" matches the (H, W, 2) shape; the operator sees 2 channels of F32.
    map_tensor = cvcuda.Tensor(map_np.shape, cvcuda.Type.F32, "HWC")
    upload_tensor(map_np, map_tensor)
    # docs_tag: end_remap_setup

    # docs_tag: begin_remap
    # Apply the remap operator.
    # map_type=ABSOLUTE means map values are absolute source-pixel coordinates.
    # src_interp=LINEAR gives smooth results on continuous displacement fields.
    # border=REPLICATE pads the edges with the nearest border pixel instead of black.
    output_image: cvcuda.Tensor = cvcuda.remap(
        src=input_image,
        map=map_tensor,
        src_interp=cvcuda.Interp.LINEAR,
        map_interp=cvcuda.Interp.NEAREST,
        map_type=cvcuda.Remap.ABSOLUTE,
        border=cvcuda.Border.REPLICATE,
    )
    write_image(output_image, args.output)
    # docs_tag: end_remap
    # docs_tag: end_main


if __name__ == "__main__":
    main()
