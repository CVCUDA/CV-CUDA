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
"""Simple Normalize example with CVCUDA."""

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


def main() -> None:
    """Normalize an image with CVCUDA using ImageNet mean and standard deviation."""
    # docs_tag: begin_main
    args: argparse.Namespace = parse_image_args("cat_normalize.jpg")
    # docs_tag: begin_read_image
    input_image: cvcuda.Tensor = read_image(args.input)
    # docs_tag: end_read_image

    # docs_tag: begin_normalize_setup
    # The normalize operator expects base (mean) and scale (std) tensors with
    # the same layout as the source.  We use per-channel ImageNet statistics
    # expressed as pixel values in [0, 255] so no manual pre-scaling is needed.
    # Shape (1, 1, 3) broadcasts over height and width for an HWC image.
    imagenet_mean = np.array(
        [[[123.675, 116.28, 103.53]]], dtype=np.float32
    )  # R, G, B means × 255
    imagenet_std = np.array(
        [[[58.395, 57.12, 57.375]]], dtype=np.float32
    )  # R, G, B stds × 255

    base_tensor = cvcuda.Tensor(imagenet_mean.shape, np.float32, "HWC")
    scale_tensor = cvcuda.Tensor(imagenet_std.shape, np.float32, "HWC")
    upload_tensor(imagenet_mean, base_tensor)
    upload_tensor(imagenet_std, scale_tensor)

    # Convert the uint8 HWC input to float32 so that the operator emits float32
    # output; a uint8 source would produce a clamped uint8 result that is
    # unsuitable for visualising the normalized values.
    uint8_host = download_tensor(input_image)
    float32_host = uint8_host.astype(np.float32)

    float32_tensor = cvcuda.Tensor(float32_host.shape, np.float32, "HWC")
    upload_tensor(float32_host, float32_tensor)
    # docs_tag: end_normalize_setup

    # docs_tag: begin_normalize
    # Apply mean-std normalization: out = (src - base) / scale.
    # SCALE_IS_STDDEV tells the operator to treat the scale tensor as a standard
    # deviation and apply out = (src - base) / (scale + epsilon) accordingly.
    # globalscale and globalshift are multiplicative/additive post-processing
    # factors applied after the per-pixel formula; both are 1.0/0.0 (identity)
    # here because we only want the standard ImageNet normalization.
    normalized: cvcuda.Tensor = cvcuda.normalize(
        src=float32_tensor,
        base=base_tensor,
        scale=scale_tensor,
        flags=cvcuda.NormalizeFlags.SCALE_IS_STDDEV,
        globalscale=1.0,
        globalshift=0.0,
        epsilon=1e-5,
    )
    # docs_tag: end_normalize

    # docs_tag: begin_normalize_postprocess
    # The normalized output is in roughly [-2, 2].  To produce a viewable image
    # we map that range linearly back to [0, 255] uint8 on the host, then upload
    # the result to a new GPU tensor so write_image can save it.
    norm_host = download_tensor(normalized)

    # Rescale: shift by ~2 (min of typical range) and compress to [0, 255]
    norm_min = norm_host.min()
    norm_max = norm_host.max()
    scale_range = norm_max - norm_min if norm_max != norm_min else 1.0
    vis_host = (
        ((norm_host - norm_min) / scale_range * 255.0).clip(0, 255).astype(np.uint8)
    )

    upload_tensor(np.ascontiguousarray(vis_host), input_image)
    write_image(input_image, args.output)
    # docs_tag: end_normalize_postprocess
    # docs_tag: end_main


if __name__ == "__main__":
    main()
