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
"""Simple Color Twist example with CVCUDA."""

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
)  # noqa: E402


def main() -> None:
    """Apply a color twist (per-channel linear transform) to an image with CVCUDA."""
    # docs_tag: begin_main
    args: argparse.Namespace = parse_image_args("cat_color_twist.jpg")
    # docs_tag: begin_read_image
    input_image: cvcuda.Tensor = read_image(args.input)
    # docs_tag: end_read_image

    # docs_tag: begin_color_twist_setup
    # The color_twist operator expects a twist matrix of shape (3, 4) with dtype F32,
    # interpreted as "HW" layout.  Each row i defines the output for channel i:
    #   out[i] = twist[i,0]*R + twist[i,1]*G + twist[i,2]*B + twist[i,3]
    # (the +offset column allows brightness shifts per channel).
    #
    # Below we build a "warm-boost" matrix that:
    #   - scales the red channel slightly up   (row 0)
    #   - leaves the green channel unchanged   (row 1)
    #   - scales the blue channel slightly down (row 2)
    # This gives the image a warm, golden-hour tint.
    twist_np = np.array(
        [
            [1.2, 0.0, 0.0, 10.0],  # R' = 1.2*R + 10
            [0.0, 1.0, 0.0, 0.0],  # G' = G
            [0.0, 0.0, 0.8, -10.0],  # B' = 0.8*B - 10
        ],
        dtype=np.float32,
    )

    # Allocate a (3, 4) F32 tensor on device with layout "HW" and upload the matrix.
    twist_tensor = cvcuda.Tensor((3, 4), cvcuda.Type.F32, "HW")
    cuda_memcpy_h2d(twist_np, twist_tensor.cuda())
    # docs_tag: end_color_twist_setup

    # docs_tag: begin_color_twist
    # color_twist applies the 3×4 affine-per-channel matrix to every pixel.
    # The operator clips the result back into the source dtype range automatically.
    output_image: cvcuda.Tensor = cvcuda.color_twist(
        src=input_image,
        twist=twist_tensor,
    )
    write_image(output_image, args.output)
    # docs_tag: end_color_twist
    # docs_tag: end_main


if __name__ == "__main__":
    main()
