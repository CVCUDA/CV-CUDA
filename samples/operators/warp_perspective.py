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
"""Simple Warp Perspective example with CVCUDA."""

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
    """Apply a perspective warp transform to an image with CVCUDA."""
    # docs_tag: begin_main
    args: argparse.Namespace = parse_image_args("cat_warp_perspective.jpg")
    # docs_tag: begin_read_image
    input_image: cvcuda.Tensor = read_image(args.input)
    # docs_tag: end_read_image

    h, w = input_image.shape[0], input_image.shape[1]

    # docs_tag: begin_warp_perspective_setup
    # Build a perspective matrix that applies a mild keystone / tilt effect.
    # The matrix maps destination pixel (x, y) to source pixel via homogeneous
    # coordinates: [x_src, y_src, w] = M @ [x_dst, y_dst, 1].
    # We nudge the top-right and bottom-left corners inward so the image
    # appears to recede into the distance without leaving empty regions.
    src_pts = np.array(
        [[0, 0], [w, 0], [w, h], [0, h]],
        dtype=np.float32,
    )
    dst_pts = np.array(
        [
            [w * 0.1, h * 0.05],
            [w * 0.9, h * 0.1],
            [w * 0.85, h * 0.95],
            [w * 0.15, h * 0.9],
        ],
        dtype=np.float32,
    )

    # Use OpenCV-compatible 3x3 float32 perspective matrix expected by cvcuda.warp_perspective.
    # We compute it manually via the 4-point DLT (Direct Linear Transform).
    def _get_perspective_transform(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
        """Compute 3x3 perspective matrix from 4 point correspondences (DLT)."""
        A = []
        for (sx, sy), (dx, dy) in zip(src, dst, strict=True):
            A.append([-sx, -sy, -1, 0, 0, 0, dx * sx, dx * sy, dx])
            A.append([0, 0, 0, -sx, -sy, -1, dy * sx, dy * sy, dy])
        A_mat = np.array(A, dtype=np.float64)
        _, _, Vt = np.linalg.svd(A_mat)
        H = Vt[-1].reshape(3, 3)
        return (H / H[2, 2]).astype(np.float32)

    xform = _get_perspective_transform(dst_pts, src_pts)
    # docs_tag: end_warp_perspective_setup

    # docs_tag: begin_warp_perspective
    # Apply the perspective warp.  We pass xform as a plain Python list-of-lists
    # (or a numpy array); cvcuda converts it internally to float32.
    # WARP_INVERSE_MAP tells the kernel that xform maps destination → source,
    # which is the convention used by the matrix we computed above.
    output_image: cvcuda.Tensor = cvcuda.warp_perspective(
        input_image,
        xform.tolist(),
        cvcuda.Interp.LINEAR | cvcuda.Interp.WARP_INVERSE_MAP,
        border_mode=cvcuda.Border.CONSTANT,
        border_value=[0],
    )
    # docs_tag: end_warp_perspective

    write_image(output_image, args.output)
    # docs_tag: end_main


if __name__ == "__main__":
    main()
