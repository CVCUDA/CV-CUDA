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
"""Simple Inpaint example with CVCUDA."""

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
    download_tensor,
)


def main() -> None:
    """Inpaint a masked region of an image with CVCUDA."""
    # docs_tag: begin_main
    args: argparse.Namespace = parse_image_args("cat_inpaint.jpg")
    # docs_tag: begin_read_image
    input_image: cvcuda.Tensor = read_image(args.input)
    # docs_tag: end_read_image

    height, width, _ = input_image.shape

    # docs_tag: begin_inpaint_setup
    # Reshape to NHWC and download so we can synthesise the damage on the CPU.
    # We keep a clean copy of the original to restore non-masked pixels later.
    nhwc_image: cvcuda.Tensor = input_image.reshape((1, height, width, 3), "NHWC")
    orig_np = download_tensor(input_image)  # (H, W, 3)
    damaged_np = orig_np.copy()[np.newaxis]  # (1,H,W,3)

    # Simulate salt-and-pepper sensor noise by randomly zeroing ~15% of pixels.
    # Each masked pixel is surrounded by unmasked neighbours so the inpaint
    # operator fills every corrupted pixel cleanly from its immediate context.
    rng = np.random.default_rng(42)
    noise_mask = rng.random((height, width)) < 0.15  # bool (H, W)
    mask_np = np.zeros((1, height, width, 1), dtype=np.uint8)
    mask_np[0, :, :, 0] = noise_mask.astype(np.uint8) * 255
    damaged_np[0, noise_mask, :] = 0

    # Save the noisy image for the before/after comparison in the docs.
    damaged_output = args.output.parent / (
        args.output.stem + "_damaged" + args.output.suffix
    )
    upload_tensor(np.ascontiguousarray(damaged_np), nhwc_image)
    write_image(nhwc_image.reshape((height, width, 3), "HWC"), damaged_output)

    # Re-upload damaged (write_image may have altered the tensor content)
    upload_tensor(np.ascontiguousarray(damaged_np), nhwc_image)

    mask_tensor: cvcuda.Tensor = cvcuda.Tensor(
        (1, height, width, 1), cvcuda.Type.U8, "NHWC"
    )
    upload_tensor(mask_np, mask_tensor)
    # docs_tag: end_inpaint_setup

    # docs_tag: begin_inpaint
    # Inpaint the noisy image — the operator reconstructs each corrupted pixel
    # from its neighbourhood.  inpaintRadius controls the neighbourhood size.
    inpaint_radius: float = 15.0
    output_nhwc: cvcuda.Tensor = cvcuda.inpaint(
        src=nhwc_image,
        masks=mask_tensor,
        inpaintRadius=inpaint_radius,
    )

    # The operator may modify pixels just outside the mask boundary.
    # Restore the original content everywhere outside the mask so only the
    # noisy pixels differ from the input.
    out_np = download_tensor(output_nhwc)  # (1, H, W, 3)
    mask_bool = mask_np.astype(bool)
    final_np = np.where(mask_bool, out_np, orig_np[np.newaxis]).astype(np.uint8)

    # Upload final result back into nhwc_image (preserves tensor format for
    # write_image) then reshape to HWC for output.
    upload_tensor(np.ascontiguousarray(final_np), nhwc_image)
    # docs_tag: end_inpaint

    output_hwc: cvcuda.Tensor = nhwc_image.reshape((height, width, 3), "HWC")
    write_image(output_hwc, args.output)
    # docs_tag: end_main


if __name__ == "__main__":
    main()
