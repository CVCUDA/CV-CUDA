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
"""Simple Gamma Contrast example with CVCUDA."""

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
    """Apply gamma contrast correction to an image with CVCUDA."""
    # docs_tag: begin_main
    args: argparse.Namespace = parse_image_args("cat_gamma_contrast.jpg")
    # docs_tag: begin_read_image
    input_image: cvcuda.Tensor = read_image(args.input)
    # docs_tag: end_read_image

    # docs_tag: begin_gamma_contrast_setup
    # This sample uses the ImageBatchVarShape overload (gamma_contrast also accepts a
    # plain Tensor with a gamma tensor or a host-scalar gamma/gain).
    # Wrap the single HWC tensor as a cvcuda.Image and push it into a batch.
    img_cvcuda: cvcuda.Image = cvcuda.as_image(input_image.cuda())
    batch = cvcuda.ImageBatchVarShape(1)
    batch.pushback(img_cvcuda)

    # Build a 1-D float32 gamma tensor, one value per image in the batch.
    # A gamma of 2.2 matches the standard sRGB display transfer function —
    # values < 1 brighten the image, values > 1 darken/increase contrast.
    gamma_np = np.array([2.2], dtype=np.float32)
    gamma_tensor = cvcuda.Tensor((1,), dtype=np.float32, layout="N")
    cuda_memcpy_h2d(gamma_np, gamma_tensor.cuda())
    # docs_tag: end_gamma_contrast_setup

    # docs_tag: begin_gamma_contrast
    # Apply gamma contrast: each pixel p is mapped to p^gamma (normalised to [0,1]).
    # The operator accepts ImageBatchVarShape inputs so it supports variable-size batches.
    out_batch: cvcuda.ImageBatchVarShape = cvcuda.gamma_contrast(batch, gamma_tensor)

    # Extract the single output image and convert back to an HWC Tensor for writing.
    out_img: cvcuda.Image = list(out_batch)[0]
    output_image: cvcuda.Tensor = cvcuda.as_tensor(out_img.cuda(), "HWC")
    write_image(output_image, args.output)
    # docs_tag: end_gamma_contrast
    # docs_tag: end_main


if __name__ == "__main__":
    main()
