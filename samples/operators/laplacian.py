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
"""Simple Laplacian example with CVCUDA."""

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
    download_tensor,
    upload_tensor,
)


def normalize_to_uint8(tensor: cvcuda.Tensor) -> cvcuda.Tensor:
    """Stretch the value range of a uint8 tensor to full [0, 255] for visibility.

    The Laplacian response is often concentrated in a narrow range;
    normalization ensures the output is a useful visual image.
    """
    host = download_tensor(tensor)

    lo = float(host.min())
    hi = float(host.max())
    if hi > lo:
        # Scale to [0, 255] using float32 arithmetic to avoid overflow
        stretched = ((host.astype(np.float32) - lo) / (hi - lo) * 255.0).astype(
            np.uint8
        )
    else:
        # Uniform image — return as-is
        stretched = host

    # Allocate a device tensor and upload the result (upload_tensor honours the
    # tensor's row pitch, so a padded allocation is handled correctly).
    out_tensor = cvcuda.Tensor(tensor.shape, np.uint8, tensor.layout)
    upload_tensor(np.ascontiguousarray(stretched), out_tensor)
    return out_tensor


def main() -> None:
    """Apply the Laplacian edge-detection operator to an image with CVCUDA."""
    # docs_tag: begin_main
    args: argparse.Namespace = parse_image_args("cat_laplacian.jpg")
    # docs_tag: begin_read_image
    input_image: cvcuda.Tensor = read_image(args.input)
    # docs_tag: end_read_image

    # docs_tag: begin_laplacian
    # Apply the Laplacian operator.
    # ksize=3 selects the 3×3 discrete Laplacian aperture; scale=1.0 leaves
    # the computed values unchanged before the output is saturated back to uint8.
    # REPLICATE border avoids zero-valued artifacts at the image boundary.
    output_image: cvcuda.Tensor = cvcuda.laplacian(
        input_image,
        ksize=3,
        scale=1.0,
        border=cvcuda.Border.REPLICATE,
    )

    # The raw Laplacian response occupies only a small fraction of [0, 255].
    # Stretch the histogram so the edges are clearly visible in the saved JPEG.
    output_image = normalize_to_uint8(output_image)

    write_image(output_image, args.output)
    # docs_tag: end_laplacian
    # docs_tag: end_main


if __name__ == "__main__":
    main()
