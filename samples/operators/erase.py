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
"""Simple Erase example with CVCUDA."""

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
    """Erase rectangular regions from an image with CVCUDA."""
    # docs_tag: begin_main
    args: argparse.Namespace = parse_image_args("cat_erase.jpg")
    # docs_tag: begin_read_image
    input_image: cvcuda.Tensor = read_image(args.input)
    # docs_tag: end_read_image

    # docs_tag: begin_erase_setup
    # The erase operator works on a batched NHWC tensor, so wrap the HWC image
    # in a batch dimension of size 1.
    height, width, channels = input_image.shape
    nhwc_image: cvcuda.Tensor = input_image.reshape(
        (1, height, width, channels), "NHWC"
    )

    # Number of rectangular regions to erase.
    num_areas = 6

    # anchor: (x, y) pixel coordinates of the top-left corner of each rectangle.
    # Shape (num_areas,) with element type _2S32 (pair of int32).
    anchor_host = np.array(
        [[50, 80], [200, 150], [350, 300], [480, 80], [480, 260], [200, 400]],
        dtype=np.int32,
    )
    anchor = cvcuda.Tensor((num_areas,), cvcuda.Type._2S32, "N")
    cuda_memcpy_h2d(anchor_host, anchor.cuda())

    # erasing: (width, height, flag) for each rectangle.
    # flag is a channel bitmask: bit0=R, bit1=G, bit2=B.
    # 7 (0b111) erases all three channels (solid fill).
    # 2 (0b010) erases only the green channel, leaving R and B intact — tint.
    erasing_host = np.array(
        [
            [120, 80, 7],
            [160, 100, 7],
            [100, 120, 7],
            [120, 100, 7],
            [120, 100, 7],
            [160, 100, 2],
        ],
        dtype=np.int32,
    )
    erasing = cvcuda.Tensor((num_areas,), cvcuda.Type._3S32, "N")
    cuda_memcpy_h2d(erasing_host, erasing.cuda())

    # values: 4 float32 fill values per area (R,G,B,A), laid out flat.
    # The operator always reserves 4 slots regardless of channel count.
    values_host = np.array(
        [
            255.0,
            0.0,
            0.0,
            0.0,  # area 0: red   (solid, flag=7)
            0.0,
            255.0,
            0.0,
            0.0,  # area 1: green (solid, flag=7)
            0.0,
            0.0,
            255.0,
            0.0,  # area 2: blue  (solid, flag=7)
            255.0,
            255.0,
            255.0,
            0.0,  # area 3: white (solid, flag=7)
            0.0,
            0.0,
            0.0,
            0.0,  # area 4: black (solid, flag=7)
            0.0,
            255.0,
            0.0,
            0.0,  # area 5: green tint (G channel only, flag=2)
        ],
        dtype=np.float32,
    )
    values = cvcuda.Tensor((num_areas * 4,), cvcuda.Type.F32, "N")
    cuda_memcpy_h2d(values_host, values.cuda())

    # imgIdx: which image in the batch each rectangle belongs to.
    # All areas are applied to image index 0.
    imgIdx_host = np.array([0, 0, 0, 0, 0, 0], dtype=np.int32)
    imgIdx = cvcuda.Tensor((num_areas,), cvcuda.Type.S32, "N")
    cuda_memcpy_h2d(imgIdx_host, imgIdx.cuda())
    # docs_tag: end_erase_setup

    # docs_tag: begin_erase
    # Erase the defined rectangles from the image.
    # random=False uses the per-area fill values supplied above.
    # random=True would ignore values and fill with pseudo-random noise seeded by seed.
    out_nhwc: cvcuda.Tensor = cvcuda.erase(
        src=nhwc_image,
        anchor=anchor,
        erasing=erasing,
        values=values,
        imgIdx=imgIdx,
        random=False,
        seed=0,
    )

    # Drop the batch dimension before saving; write_image expects HWC layout.
    out_hwc: cvcuda.Tensor = out_nhwc.reshape((height, width, channels), "HWC")
    write_image(out_hwc, args.output)
    # docs_tag: end_erase
    # docs_tag: end_main


if __name__ == "__main__":
    main()
