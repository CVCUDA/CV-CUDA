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
"""Simple Morphology example with CVCUDA."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cvcuda

common_dir = Path(__file__).parent.parent
sys.path.append(str(common_dir))

from common import parse_image_args, read_image, write_image  # noqa: E402


def main() -> None:
    """Apply morphological erosion and dilation to an image with CVCUDA."""
    # docs_tag: begin_main
    args: argparse.Namespace = parse_image_args("cat_morphology.jpg")
    # docs_tag: begin_read_image
    input_image: cvcuda.Tensor = read_image(args.input)
    # docs_tag: end_read_image

    # docs_tag: begin_morphology
    # Wrap the HWC single image in a batch dimension (NHWC) so the morphology
    # operator can process it. The operator accepts both HWC and NHWC layouts.
    nhwc_image: cvcuda.Tensor = input_image.reshape((1, *input_image.shape), "NHWC")

    # A 5x5 rectangular structuring element is large enough to show a visible
    # effect on a natural image without destroying structure.
    mask_size = [5, 5]
    # anchor=[-1, -1] centres the structuring element automatically.
    anchor = [-1, -1]

    # DILATE expands bright regions — edges become thicker and fine dark lines
    # are reduced.  A workspace tensor is not required for a single iteration.
    dilated: cvcuda.Tensor = cvcuda.morphology(
        nhwc_image,
        cvcuda.MorphologyType.DILATE,
        mask_size,
        anchor,
        iteration=1,
        border=cvcuda.Border.REPLICATE,
    )

    # ERODE is the dual of dilation — it shrinks bright regions and removes
    # small bright specks.  Running erode after dilate is a CLOSE operation,
    # which suppresses small dark artifacts/holes while preserving larger
    # structures.
    workspace: cvcuda.Tensor = cvcuda.Tensor(
        nhwc_image.shape, nhwc_image.dtype, nhwc_image.layout
    )
    closed: cvcuda.Tensor = cvcuda.morphology(
        dilated,
        cvcuda.MorphologyType.ERODE,
        mask_size,
        anchor,
        iteration=1,
        border=cvcuda.Border.REPLICATE,
        workspace=workspace,
    )

    # Remove the batch dimension before writing; write_image expects HWC.
    result: cvcuda.Tensor = closed.reshape(closed.shape[1:], "HWC")
    write_image(result, args.output)
    # docs_tag: end_morphology
    # docs_tag: end_main


if __name__ == "__main__":
    main()
